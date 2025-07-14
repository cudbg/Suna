import pandas as pd
import bisect
import os
import torch
from functools import reduce
import psutil
from semi_ring import MomentSemiRing


# TODO: this batching thing is not being used so far, it is just inherited from
# the kitana code.
# Will finally want to implement this in case data is huge
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


def calculate_total_tensor_size(*args):
    total_size = 0
    for tensor in args:
        total_size += tensor.nelement() * tensor.element_size()

    return total_size


class SketchLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        # {batch_id: {moment: tensor}}
        # This is dynamic and will be updated across search iterations
        self.batch_sketches = {}
        self.num_batches = 0

    def update_sketch(self, update_sketch, batch_id):
        self.batch_sketches[batch_id] = update_sketch

    def load_sketches(self, semi_ring, feature_index_map, seller_id, cur_df_offset=0):
        m = semi_ring.moments[0].size(1)
        if not self.batch_sketches:
            self.batch_sketches[0] = semi_ring.slice_semi_ring(
                min(self.batch_size, m))
            feature_index_map[0] = [(0, seller_id, 0)]
            cur_df_offset = self.batch_size
        else:
            # Find the last batch number
            last_batch_num = max(self.batch_sketches.keys())
            cur_space = self.batch_sketches[last_batch_num].moments[0].size(1)
            remaining_space = self.batch_size - cur_space
            amount_to_append = min(remaining_space, m)
            if remaining_space > 0:
                new_semi_ring = semi_ring.slice_semi_ring(amount_to_append)
                self.batch_sketches[last_batch_num].concat_semi_ring(new_semi_ring)
                bisect.insort(feature_index_map[last_batch_num], (
                    cur_space, seller_id, cur_df_offset))
                cur_df_offset += remaining_space
            else:
                last_batch_num += 1
                self.batch_sketches[last_batch_num] = semi_ring.slice_semi_ring(
                    min(self.batch_size, m))
                feature_index_map[last_batch_num] = [(0, seller_id, cur_df_offset)]
                cur_df_offset += self.batch_size

        self.num_batches = len(self.batch_sketches.keys())
        if semi_ring.moments[0].size(1) > 0:
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
        self.is_buyer = is_buyer
        if join_key_index is None:
            self.join_key_index = pd.MultiIndex.from_product(
                index_ranges, names=self.join_keys)
        else:
            self.join_key_index = join_key_index
        # This will hold tuples of batch_ind:
        # [(batch_start_ind, seller_id, seller_offset)...]
        self.feature_index_mapping = {}
        # Store the seller datasets' feature names mapped by seller_id
        self.seller_features = {}
        self.device = device
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.init()
            gpu_total_mem = torch.cuda.get_device_properties(0).total_memory
            self.gpu_free_mem = gpu_total_mem - torch.cuda.memory_allocated(0)
        else:
            self.gpu_free_mem = None

        self.gpu_batch_size, self.ram_batch_size = self._estimate_batch_size()
        self.sketch_loader = SketchLoader(self.gpu_batch_size)

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

    # register a dataframe with this sketch
    def register_df(self, df_id, df, features, deg=1, agg=None):
        if not set(self.join_keys).issubset(set(df.columns)):
            raise ValueError('Join key does not present in dataset.')
        # Exclude join key columns from feature count
        num_features = len(features)

        if agg is None:
            pass
        elif agg == 'mean':
            df = df.groupby(self.join_keys).mean().reset_index()
        else:
            raise Exception(f"{agg} is not supported so far")

        if len(df.columns) > self.ram_batch_size:
            print("Are we here?????")
            features_per_partition = self.ram_batch_size - 1
            # Splitting the DataFrame into partitions
            num_partitions = ((len(df.columns) - len(
                self.join_keys)) // features_per_partition) + 1
            for i in range(num_partitions):
                start_col = len(self.join_keys) + i * features_per_partition
                end_col = start_col + features_per_partition

                # Selecting the columns for this partition
                cur_features = list(df.columns[start_col:end_col])
                cols = self.join_keys + cur_features

                # Creating the partition DataFrame
                partition_df = df[cols]
                # s = time.time()
                msr = self._lift_and_calibrate(
                    df_id,
                    partition_df,
                    len(cur_features),
                    deg=deg
                )
                # t = time.time()
                self.sketch_loader.load_sketches(
                    msr,
                    self.feature_index_mapping,
                    df_id,
                    cur_df_offset=start_col-len(self.join_keys)
                )
        else:
            msr = self._lift_and_calibrate(
                df_id,
                df,
                num_features,
                deg=deg
            )
            self.sketch_loader.load_sketches(
                msr, self.feature_index_mapping, df_id)
        return df

    # compute the semi-ring structure with respect to each join key
    def _lift_and_calibrate(self, df_id, df, num_features, deg):
        ordered_columns = list(df.drop(columns=self.join_keys).columns)

        if df_id in self.seller_features:
            self.seller_features[df_id] += ordered_columns
        else:
            self.seller_features[df_id] = ordered_columns

        seller_count = df.groupby(self.join_keys).size().to_frame('count')
        seller_moments = {
            0: seller_count
        }
        seller_moments[0].index = pd.MultiIndex.from_arrays(
            [seller_count.index], names=self.join_keys)

        for i in range(1, 2*deg + 1):
            df_m = df[ordered_columns]**i
            df_m[self.join_keys] = df[self.join_keys]
            seller_moments[i] = df_m.groupby(
                self.join_keys).sum()[ordered_columns]
            seller_moments[i].index = pd.MultiIndex.from_arrays(
                [seller_moments[i].index], names=self.join_keys)
        # print(f"This is seller_moments: {seller_moments[0]}")

        # Temporary DataFrame to facilitate 'inner' join
        temp_df = pd.DataFrame(index=self.join_key_index)

        for i in range(2*deg + 1):
            if self.is_buyer:
                seller_moments[i] = seller_moments[i].reindex(
                    self.join_key_index, fill_value=0)
            # seller_moments[i] = seller_moments[i].reindex(
            #     self.join_key_index, fill_value=0)
            if i == 0:
                seller_moments[i] = seller_moments[i].reindex(
                    self.join_key_index, fill_value=1)
                
                seller_moments[i] = torch.tensor(
                    seller_moments[i][seller_moments[i].index.isin(temp_df.index)].values, 
                    dtype=torch.int, device=self.device).view(-1, 1)
                seller_moments[i] = seller_moments[i].expand(-1, num_features)
            else:
                seller_moments[i] = seller_moments[i].reindex(
                    self.join_key_index).fillna(seller_moments[i].mean())
                
                seller_moments[i] = torch.tensor(
                    seller_moments[i][seller_moments[i].index.isin(temp_df.index)].values, 
                    dtype=torch.float, device=self.device)
        # print(seller_moments)
        return MomentSemiRing(moments=seller_moments, device=self.device)

    def get_seller_by_feature_index(self, batch_id, feature_index):
        # Perform a binary search to find the right interval
        # bisect.bisect returns the insertion point which gives us the index
        # where the feature_index would be inserted to maintain order.
        # We subtract one to get the tuple corresponding to the start index of
        # the range that the feature_index falls into.
        def bisect(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid][0]:
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
        self.seller_sketches = {}
        self.device = device

    def get_seller_join_sketch(self, join_key):
        return self.seller_sketches[join_key]

    def get_seller_data(self, seller_id):
        return self.seller_datasets[seller_id][0]

    def add_seller(self, seller, seller_name, join_keys,
                   join_key_domains, seller_features, deg=1):
        if not seller_features:
            return
        try:
            if not isinstance(seller, str):
                seller_df = seller
            elif ".parquet" in seller:
                seller_df = pd.read_parquet(seller)
            else:
                seller_df = pd.read_csv(seller)
        except ...:
            raise Exception(seller + " is too large, consider splitting or sampling.")

        for join_key in join_keys:
            cur_domain = {}
            for col in join_key:
                cur_domain[col] = join_key_domains[col]
                if col not in self.join_key_domains:
                    self.join_key_domains[col] = join_key_domains[col]
            join_key_ind = tuple(sorted(join_key))
            if join_key_ind in self.seller_sketches:
                agg_df = self.seller_sketches[join_key_ind].register_df(
                    self.seller_count,
                    seller_df[join_key + seller_features],
                    seller_features,
                    deg=deg
                )
            else:
                join_sketch_instance = JoinSketch(
                    join_key_domain=cur_domain, device=self.device)
                agg_df = join_sketch_instance.register_df(
                    self.seller_count,
                    seller_df[join_key + seller_features],
                    seller_features,
                    deg=deg
                )
                self.seller_sketches[join_key_ind] = join_sketch_instance

        self.seller_datasets[self.seller_count] = (agg_df, seller_name)
        self.seller_count += 1





















