from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import bisect
import os
import torch
from functools import reduce
from torch.utils.data import DataLoader, TensorDataset
import psutil
import math
import copy
import time

# TODO: this batching thing is not being used so far, it is just inherited from the kitana code.
# will finally want to implement this in case data is huge

def memory_usage():
    # This function returns the current process's memory usage in MB
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def cleanup(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            del arg
        elif isinstance(arg, dict):
            for key, val in arg.items():
                if isinstance(arg, torch.Tensor):
                    del arg
    torch.cuda.empty_cache()


def linear_regression_residuals(df, X_columns, Y_column):
    # Ensure that X_columns exist in the dataframe
    if not all(item in df.columns for item in X_columns):
        raise ValueError('Not all specified X_columns are in the dataframe.')
    if Y_column not in df.columns:
        raise ValueError('The Y_column is not in the dataframe.')

    # Prepare the feature matrix X by selecting the X_columns and adding an intercept term
    X = df[X_columns].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept term

    # Extract the target variable vector Y
    Y = df[Y_column].values

    theta = np.linalg.pinv(X.T @ X) @ X.T @ Y

    Y_pred = X @ theta

    residuals = Y - Y_pred

    # Add residuals to the dataframe
    df['residuals'] = residuals

    SS_res = (residuals ** 2).sum()
    SS_tot = ((Y - np.mean(Y)) ** 2).sum()
    R_squared = 1 - SS_res / SS_tot
    return df, R_squared


def calculate_total_tensor_size(*args):
    total_size = 0
    for tensor in args:
        total_size += tensor.nelement() * tensor.element_size()

    return total_size

    
class SemiRing:
    # For a join sketch, we may compute different sets of aggregation that forms semi-rings
    # This class covers some common semi-ring that we will use in our sketches
    def __init__(self, c, s, Q, agg_min, agg_max, device='cpu'):
        self.c = c.to(device)
        self.s = s.to(device)
        self.Q = Q.to(device)
        self.min = agg_min.to(device)
        self.max = agg_max.to(device)
        self.device = device
    
    # return a semi_ring by getting columns 0:ind.
    # update the current SemiRing class to columns ind+1: 
    # don't need the to(device) because already on device before slicing
    def concat_semi_ring(self, semi_ring):
        # Assume the parameter semi_rings data is on the same device as self.device
        self.c = torch.cat([self.c, semi_ring.c], dim=1)
        self.s = torch.cat([self.s, semi_ring.s], dim=1)
        self.Q = torch.cat([self.Q, semi_ring.Q], dim=1)
        self.min = torch.cat([self.min, semi_ring.min], dim=1)
        self.max = torch.cat([self.max, semi_ring.max], dim=1)
    
    def slice_semi_ring(self, ind):
        new_semi_ring = SemiRing(
            c=self.c[:, :ind],
            s=self.s[:, :ind],
            Q=self.Q[:, :ind],
            agg_min=self.min[:, :ind],
            agg_max=self.max[:, :ind],
            device=self.device
        )
        self.c = self.c[:, ind:]
        self.s = self.s[:, ind:]
        self.Q = self.Q[:, ind:]
        self.min = self.min[:, ind:]
        self.max = self.max[:, ind:]
        return new_semi_ring
    
    def ind_semi_ring(self, inds):
        return SemiRing(
            c=self.c[:, inds],
            s=self.s[:, inds],
            Q=self.Q[:, inds],
            agg_min=self.min[:, inds],
            agg_max=self.max[:, inds],
            device=self.device
        )
    
    def set_ind_zero(self, ind):
        self.s[:, ind:ind+1] = 0
        self.Q[:, ind:ind+1] = 0
        self.min[:, ind:ind+1] = 0
        self.max[:, ind:ind+1] = 0
        
        
class SketchLoader:
    def __init__(self, batch_size, is_buyer=False, device='cpu', disk_dir='sketches/'):
        self.batch_size = batch_size
        
        # {batch_id: {moment: tensor}}
        # This is dynamic and will be updated across search iterations
        self.batch_sketches = {}
        self.is_buyer = is_buyer
        self.device = device
        self.num_batches = 0
        self.disk_dir = disk_dir
        
    def update_sketch(self, update_sketch, batch_id):
        self.batch_sketches[batch_id] = update_sketch

    def load_sketches(self, semi_ring, feature_index_map, seller_id, cur_df_offset=0):
        if self.is_buyer:
            self.num_batches = 1
            self.batch_sketches[0] = semi_ring
            return
        
        if not self.batch_sketches:
            self.batch_sketches[0] = semi_ring.slice_semi_ring(
                min(self.batch_size, semi_ring.c.size(1)))
            feature_index_map[0] = [(0, seller_id, 0)]
            cur_df_offset = self.batch_size
        else:
            # Find the last batch number
            last_batch_num = max(self.batch_sketches.keys())
            cur_space = self.batch_sketches[last_batch_num].c.size(1)
            remaining_space = self.batch_size - cur_space
            amount_to_append = min(remaining_space, semi_ring.c.size(1))
            if remaining_space > 0:
                new_semi_ring = semi_ring.slice_semi_ring(amount_to_append)
                self.batch_sketches[last_batch_num].concat_semi_ring(new_semi_ring)
                bisect.insort(feature_index_map[last_batch_num], (
                    cur_space, seller_id, cur_df_offset))
                cur_df_offset += remaining_space
            else:
                last_batch_num += 1
                self.batch_sketches[last_batch_num] = semi_ring.slice_semi_ring(
                    min(self.batch_size, semi_ring.c.size(1)))
                feature_index_map[last_batch_num] = [(0, seller_id, cur_df_offset)]
                cur_df_offset += self.batch_size
                
        self.num_batches = len(self.batch_sketches.keys())
        if semi_ring.c.size(1) > 0:           
            self.load_sketches(semi_ring, feature_index_map, seller_id, cur_df_offset)
        
    def get_sketches(self, batch_id, inds=None):
        if inds is not None:
            return self.batch_sketches[batch_id].ind_semi_ring(inds)
        else:
            return self.batch_sketches[batch_id]
        

class JoinSketch:
    # join_key-based sketches, a dataset can be registered with respect to many join sketch.
    # join_key_domain must be a dictionary with each key be the name of the column and the values
    # be an array of the possible values of the column, based on the assumption that the domain shall not be too big
    def __init__(self, join_key_domain, device='cpu', is_buyer=False, join_key_index=None):
        self.join_keys = sorted(list(join_key_domain.keys()))
        self.join_key_domain = join_key_domain
        index_ranges = [self.join_key_domain[col] for col in self.join_keys]
        if join_key_index is None:
            self.join_key_index = pd.MultiIndex.from_product(index_ranges, names=self.join_keys)
        else:
            self.join_key_index = join_key_index
        self.is_buyer = is_buyer
        # This will hold tuples of batch_ind: [(batch_start_ind, seller_id, seller_offset)...]
        self.feature_index_mapping = {}
        # Additional structure to store the seller datasets' feature names mapped by seller_id
        self.seller_features = {}
        self.device = device
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory
            self.gpu_free_mem = gpu_total_mem - torch.cuda.memory_allocated(0)
        else:
            self.gpu_free_mem = None
        
        # number of columns that may fit into gpu/cpu
        self.gpu_batch_size, self.ram_batch_size = self._estimate_batch_size()
        # sketch loader only needs to fully utilize gpu memory
        self.sketch_loader = SketchLoader(self.gpu_batch_size, is_buyer, device, 
                                          disk_dir='sketches/' + '_'.join(self.join_keys) + '/')

    def _estimate_batch_size(self):
        bytes_per_element = 4
        tensor_width = reduce(lambda x, y: x * len(y), 
                              self.join_key_domain.values(), 1)
        memory = psutil.virtual_memory()
        # TODO: 2 is a workaround
        available_memory = memory.available // 2
        ram_batch_size = available_memory // (bytes_per_element * 5 * tensor_width)
        if not self.gpu_free_mem or not torch.cuda.is_available():
            gpu_batch_size = ram_batch_size
        else:
            gpu_batch_size = self.gpu_free_mem // (bytes_per_element * 5 * tensor_width)
        return gpu_batch_size, ram_batch_size
    
    def sketch_to_df(self, conf_feature, batch_ind, feature_ind):
        sketch_col = self.sketch_loader.get_sketches(batch_ind, [feature_ind])
        conf_df = pd.DataFrame(index=self.join_key_index)
        if self.device != 'cpu':
            conf_df[conf_feature] = sketch_col[1].cpu().numpy().flatten()
        else:
            conf_df[conf_feature] = sketch_col[1].numpy().flatten()
        return conf_df
    
    def get_cross_term(self, df, col1, col2):
        if not set(self.join_keys + [col1, col2]).issubset(set(df.columns)):
            raise ValueError('Features not present in dataset.')
        df[f"{col1}_{col2}"] = df[col1] * df[col2]
        cross_sum = df[[f"{col1}_{col2}"] + self.join_keys].groupby(
            self.join_keys
        ).sum()
        if not isinstance(cross_sum.index, pd.MultiIndex):
            cross_sum.index = pd.MultiIndex.from_arrays(
                [cross_sum.index], names=self.join_keys)
        
        cross_sum = cross_sum.reindex(self.join_key_index, fill_value=0)
        df.drop(f"{col1}_{col2}", axis=1, inplace=True)
        return torch.tensor(
            cross_sum[cross_sum.index.isin(self.join_key_index)].values, 
            dtype=torch.float32).to(self.device)

    # register a dataframe with this sketch
    def register_df(self, df_id, df, features, agg='mean'):
        if not set(self.join_keys).issubset(set(df.columns)):
            raise ValueError('Join key does not present in dataset.')
            
        # Exclude join key columns from feature count
        num_features = len(features)

        # TODO: if data cannot even fit on ram, write to disk, now assume always CPU ram fits everything
        if len(df.columns) > self.ram_batch_size:
            features_per_partition = self.ram_batch_size - 1
            # Splitting the DataFrame into partitions
            num_partitions = ((len(df.columns) - len(self.join_keys)) // features_per_partition) + 1
            for i in range(num_partitions):
                start_col = len(self.join_keys) + i * features_per_partition # 1 is added to skip the join_key column
                end_col = start_col + features_per_partition

                # Selecting the columns for this partition
                cur_features = list(df.columns[start_col:end_col])
                cols = self.join_keys + cur_features

                # Creating the partition DataFrame
                partition_df = df[cols]
                semi_ring = self._lift_and_calibrate(
                    df_id, 
                    partition_df, 
                    len(cur_features),
                    agg=agg
                )
                self.sketch_loader.load_sketches(
                    semi_ring, 
                    self.feature_index_mapping, 
                    df_id, 
                    cur_df_offset=start_col-len(self.join_keys)
                )
        else:
            semi_ring = self._lift_and_calibrate(
                df_id, 
                df, 
                num_features, 
                agg=agg
            )
            self.sketch_loader.load_sketches(
                semi_ring, self.feature_index_mapping, df_id)
            
    # compute the semi-ring structure with respect to each join key
    def _lift_and_calibrate(self, df_id, df, num_features, agg):
        join_keys_df = df[self.join_keys]
        data_df = df.drop(columns=self.join_keys)
        ordered_columns = list(data_df.columns)
        if self.is_buyer:
            seller_agg = df
        # if seller, we consider aggregate based on join key
        elif agg=='mean':
            seller_agg = df.groupby(self.join_keys).mean()
        else:
            raise Exception(f"{agg} is not supported so far")
            
        seller_agg = seller_agg.reset_index()
        if df_id in self.seller_features:
            self.seller_features[df_id] += ordered_columns
        else:
            self.seller_features[df_id] = ordered_columns
        seller_count = seller_agg.groupby(self.join_keys).size().to_frame('count')
        
        seller_moments = {0: seller_count}
        
        # compute the variance semi-ring
        for i in range(1, 3):
            df_m = seller_agg[ordered_columns]**i
            df_m[self.join_keys] = seller_agg[self.join_keys]

            seller_moments[i] = df_m.groupby(
                self.join_keys).sum()[ordered_columns]
        
        # compute the min-max semi-ring
        agg_min = seller_agg.groupby(self.join_keys).agg(['min'])[
            ordered_columns]
        agg_max = seller_agg.groupby(self.join_keys).agg(['max'])[
            ordered_columns]
        
        # indexing for join key spanning more than a single column
        for i in range(3):
            seller_moments[i].index = pd.MultiIndex.from_arrays(
                [seller_moments[i].index], names=self.join_keys)
        agg_min.index = pd.MultiIndex.from_arrays(
            [agg_min.index], names=self.join_keys)
        agg_max.index = pd.MultiIndex.from_arrays(
            [agg_max.index], names=self.join_keys)

        # Temporary DataFrame to facilitate 'inner' join
        temp_df = pd.DataFrame(index=self.join_key_index)

        for i in range(3):
            seller_moments[i] = seller_moments[i].reindex(self.join_key_index, fill_value=0)
            if i == 0:
                seller_moments[i] = torch.tensor(
                    seller_moments[i][seller_moments[i].index.isin(temp_df.index)].values, 
                    dtype=torch.int).view(-1, 1)
                seller_moments[i] = seller_moments[i].expand(-1, num_features)
            else:
                seller_moments[i] = torch.tensor(
                    seller_moments[i][seller_moments[i].index.isin(temp_df.index)].values, 
                    dtype=torch.float32)
        
        agg_min = agg_min.reindex(self.join_key_index, fill_value=0)
        agg_min = torch.tensor(
            agg_min[agg_min.index.isin(temp_df.index)].values, 
            dtype=torch.float32
        )
        agg_max = agg_max.reindex(self.join_key_index, fill_value=0)
        agg_max = torch.tensor(
            agg_max[agg_max.index.isin(temp_df.index)].values, 
            dtype=torch.float32
        )
        return SemiRing(seller_moments[0], seller_moments[1], 
                        seller_moments[2], agg_min, agg_max, self.device)
    
    
    def get_seller_by_feature_index(self, batch_id, feature_index):
        # Perform a binary search to find the right interval
        # bisect.bisect returns the insertion point which gives us the index where the feature_index would be inserted to maintain order.
        # We subtract one to get the tuple corresponding to the start index of the range that the feature_index falls into.
        def bisect(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:  # Compare with the first element of the tuple at mid
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        index = bisect(self.feature_index_mapping[batch_id], feature_index) - 1
        start_index, seller_id, offset = self.feature_index_mapping[batch_id][index]
        # Calculate the local feature index within the seller's dataset
        local_feature_index = feature_index - start_index + offset
        return seller_id, self.seller_features[seller_id][local_feature_index]


class DataMarket:
    def __init__(self, device='cpu'):
        self.seller_count = 1
        self.buyer_count = 1
        self.seller_datasets = {}
        self.join_key_domains = {}
        self.seller_join_sketches = {} 
        self.device = device
        
    def get_seller_join_sketch(self, join_key):
        return self.seller_join_sketches[join_key]

    def get_seller_data(self, seller_id):
        return self.seller_datasets[seller_id][0]

    def add_seller(self, seller, seller_name, join_keys, join_key_domains, seller_features, agg='mean'):
        if not seller_features: return
        try:
            if not isinstance(seller, str):
                seller_df = seller
            elif ".parquet" in seller:
                seller_df = pd.read_parquet(seller)
            else:
                seller_df = pd.read_csv(seller)
        except:
            raise Exception(seller + " is too large, consider splitting or sampling.")
        
        for join_key in join_keys:
            cur_domain = {}
            for col in join_key:
                cur_domain[col] = join_key_domains[col]
                if col not in self.join_key_domains:
                    self.join_key_domains[col] = join_key_domains[col]
            join_key_ind = tuple(sorted(join_key))
            if join_key_ind in self.seller_join_sketches:
                self.seller_join_sketches[join_key_ind].register_df(
                    self.seller_count, 
                    seller_df[join_key + seller_features],
                    seller_features, 
                    agg=agg
                )
            else:
                join_sketch_instance = JoinSketch(join_key_domain=cur_domain, device=self.device)
                join_sketch_instance.register_df(
                    self.seller_count, 
                    seller_df[join_key + seller_features], 
                    seller_features,
                    agg=agg
                )
                self.seller_join_sketches[join_key_ind] = join_sketch_instance
        
        self.seller_datasets[self.seller_count] = (seller_df, seller_name)
        self.seller_count += 1