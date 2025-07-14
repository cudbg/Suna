import warnings
warnings.filterwarnings('ignore')
import math
import sys
import torch
import copy
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from data_profile import DataProfile
from sketch_search_v0 import JoinSketch, DataMarket, cleanup, SemiRing
from sklearn.linear_model import LinearRegression
import time


def causal_effect(data, X: str, Y: str, adjustment_set={}):
    """
    Calculate the linar treatment effect of X on Y on input dataset, using adjustment_set.
    """
    # Ensure X, Y, and adjustment variables are in the DataFrame
    data = data[[X, Y] + list(adjustment_set)]
    data = data.dropna()

    if len(data) == 0:
        return -1
    if X not in data.columns or Y not in data.columns or not set(adjustment_set).issubset(data.columns):
        raise ValueError(
            "Input DataFrame does not contain all of the following: treatment X, outcome Y, and backdoor sets.")

    # Select explanatory variables (X and adjustment variables)
    explanatory_vars = [X] + list(adjustment_set)
    X_data = data[explanatory_vars]

    # Select outcome variable
    Y_data = data[Y]

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_data, Y_data)

    # Extract the coefficient of X
    x_coef_index = explanatory_vars.index(X)
    linear_causal_effect = model.coef_[x_coef_index]

    return linear_causal_effect


def prune_confounders(buyer_df, cd, dm, target_ce, treatment, outcome, k=1):
    seller_feature_inds = {}
    for ele in cd.conf_set[(treatment, outcome)]:
        join_key, dataset, feature, f_batch, f_ind = ele[0], dm.seller_datasets[
            ele[1]][1], ele[2], ele[3], ele[4]
        if join_key in seller_feature_inds:
            if f_batch in seller_feature_inds[join_key]:
                seller_feature_inds[
                    join_key][f_batch]['features'].append(f'{dataset}:{feature}')
                seller_feature_inds[
                    join_key][f_batch]['inds'].append(f_ind)
            else:
                seller_feature_inds[join_key][f_batch] = {
                    'features': [f'{dataset}:{feature}'],
                    'inds': [f_ind]
                }
        else:
            seller_feature_inds[join_key] = {
                f_batch: {
                    'features': [f'{dataset}:{feature}'],
                    'inds': [f_ind]
                }
            }
    
    confs = []
    for i in range(k):
        dist_to_targ = math.inf
        best_buyer, best_feature = buyer_df, None
        best_batch_id, best_join_key, best_ind = None, None, None
        for join_key, batch_inds in seller_feature_inds.items():
            search_sketch = dm.seller_join_sketches[join_key]
            for batch_id in batch_inds.keys():
                inds = torch.tensor(seller_feature_inds[
                    join_key][batch_id]['inds'])
                s_sketch = search_sketch.sketch_loader.get_sketches(
                    batch_id, inds
                ).s

                cols = seller_feature_inds[
                    join_key][batch_id]['features']

                # Create DataFrame
                conf_df = pd.DataFrame(
                    data=s_sketch.cpu().numpy(),
                    columns=cols,
                    index=search_sketch.join_key_index
                )

                join_df = buyer_df.merge(
                    conf_df, on=search_sketch.join_keys, how='left'
                )

                for ind, col in enumerate(cols):
                    ce = causal_effect(
                        join_df[[treatment, outcome, col] + confs], 
                        X=treatment, Y=outcome, 
                        adjustment_set=set([col] + confs)
                    )
                    # print(f"Feature: {col}, Effect: {ce}")
                    cur_dist = abs(target_ce - ce)
                    if cur_dist < dist_to_targ:
                        best_buyer = join_df[list(buyer_df.columns) + [col]]
                        best_feature = col
                        best_batch_id = batch_id
                        best_join_key = join_key
                        dist_to_targ = cur_dist
                        best_ind = inds[ind]
                        
        tmp = seller_feature_inds[best_join_key][best_batch_id]
        new_ind, new_features = [], []
        for j, j_ind in enumerate(tmp['inds']):
            if j_ind != best_ind:
                new_ind.append(j_ind)
                new_features.append(tmp['features'][j])
        seller_feature_inds[best_join_key][best_batch_id] = {
            'features': new_features,
            'inds': new_ind
        }
        confs.append(best_feature)
        buyer_df = best_buyer
        # print("X"*100)
    return confs, causal_effect(
        buyer_df, 
        X=treatment, Y=outcome, 
        adjustment_set=set(confs)
    )


class ConDiscovery:
    def __init__(
        self, 
        dm,
        err=0.001, 
        mi_threshold=0.05, 
        device='cpu', 
        diagram_name='diagram', 
        approx=False,
        hist=False,
        verbose=False,
        factorized_hist=False,
        factor=1,
        test_histogram=False,
    ):
        self.err = err
        self.mi_threshold = mi_threshold
        self.device = device
        self.bin_num = None
        self.approx=approx
        self.hist=hist
        self.verbose=verbose
        self.factorized_hist=factorized_hist
        self.factor = factor
        self.test_histogram = test_histogram
        
        # Assume that this only contains treatment and outcome
        self.dm = dm
        self.cur_data_in = None
        self.seller_count = 1
        
        self.treat_out_sketches = {}
        self.cross_t_o = {}
        
        # Util parameters for iterative search
        self.buyer_join_keys = None  # only consider single column join key for now
        self.conf_set = {}
        self.exclude_ind = {}
        self.coeff_w_o_adj = 0
        self.treatment = None
        self.outcome = None
        self.conf_type = None
        
        self.treat_vecs = {}
        self.out_vecs = {}
        self.trans_mats = {}
        self.t_o_mi = 0
    
    def _align_treat_out_vec(
        self, 
        join_key, 
        treat_out_instance
    ):
        # get the transformation matrix
        if isinstance(join_key, tuple):
            join_key = list(join_key)
        # initialize transformation matrices
        align_df = self.cur_data_in[
            join_key + [self.treatment, self.outcome]].set_index(join_key)
        align_df.index = pd.MultiIndex.from_arrays(
            [align_df.index], names=join_key)

        desired_order = treat_out_instance.join_key_index.to_frame().reset_index(
            drop=True)
        merged = pd.merge(
            desired_order, 
            self.cur_data_in.reset_index(), 
            on=join_key, 
            how='left')

        merged = merged.sort_index()

        self.cur_data_in = merged.loc[:, self.cur_data_in.columns]

        align_df = self.cur_data_in[join_key + [self.treatment, self.outcome]]

        align_treat_vec = torch.tensor(
            align_df[self.treatment].values, dtype=torch.float32)
        align_out_vec = torch.tensor(
            align_df[self.outcome].values, dtype=torch.float32)

        self.treat_vecs[tuple(join_key)] = align_treat_vec.view(-1, 1).to(self.device)
        self.out_vecs[tuple(join_key)] = align_out_vec.view(-1, 1).to(self.device)
    
    # TODO: this function will be removed after doing the system optimization
    def _get_exp_res(self, exp_sketch, targ_sketch, exp_vec, targ_vec):
        # Opt TODO: The slope and intercept only needs to be computed once
        # because in the normalized case, the slope is just the correlation, and intercept is 0
        c_x, s_x, Q_x = exp_sketch.c, exp_sketch.s, exp_sketch.Q
        c_y, s_y, Q_y = targ_sketch.c, targ_sketch.s, targ_sketch.Q

        y_y = torch.sum(Q_y * c_x, dim=0)
        y = torch.sum(s_y * c_x, dim=0)
        c = torch.sum(c_y * c_x, dim=0)
        x_y = torch.sum(s_x * s_y, dim=0)
        x_x = torch.sum(Q_x * c_y, dim=0)
        x = torch.sum(s_x * c_y, dim=0)
        
        x_mean = x / c
        y_mean = y / c
        sd_x = torch.nan_to_num(
            torch.sqrt((x_x - (x ** 2) / c) / (c-1)), nan=1)
        sd_y = torch.nan_to_num(
            torch.sqrt((y_y - (y ** 2) / c) / (c-1)), nan=1)
        # last iteration's discovered vector will become all 0, so sd = 0
        sd_x = torch.where(sd_x == 0, torch.tensor(1), sd_x)
        sd_y = torch.where(sd_y == 0, torch.tensor(1), sd_y)
        x_x_std = (x_x - 2*x_mean*x + c*x_mean**2)/(sd_x**2)
        x_y_std = (x_y - y*x_mean - x*y_mean + c*x_mean*y_mean)/(sd_x*sd_y)
        slope = torch.nan_to_num(
            torch.where(x_x_std != 0, x_y_std / x_x_std, torch.tensor(0.0)), nan=0)
        slope = torch.where(torch.abs(sd_x) > 1, torch.tensor(1), slope)

        res_vec = (targ_vec - torch.mean(targ_vec))/torch.std(targ_vec) - slope * exp_vec
        cleanup(x_x_std, x_y_std)
        return res_vec
    
    # In this function, the exp_sketch is 1D, representing the fact table, 
    # the targ_sketch are dimension tables, which are often of high dimension.
    # explanatory sketch and target sketch
    # returns the difference of the mutual information
    # mutual information of using exp v.s. the prediction residual of targ (mi)
    # and the targ v.s. the prediction residual of exp (mi_r)
    def bivariate_causal_discovery(self, t_sketch, seller_sketch, join_key, mfactor=1):
        treat_vec = self.treat_vecs[join_key]
        c_x, s_x, Q_x = t_sketch.c, t_sketch.s, t_sketch.Q
        c_y, s_y, Q_y = seller_sketch.c, seller_sketch.s, seller_sketch.Q
        c, x_y = c_x * c_y, s_x * s_y
        x, x_x = s_x * c_y, Q_x * c_y
        y, y_y = s_y * c_x, Q_y * c_x

        agg_c = torch.sum(c, dim=0)
        agg_x_y = torch.sum(x_y, dim=0)
        agg_x = torch.sum(x, dim=0)
        agg_x_x = torch.sum(x_x, dim=0)
        agg_y = torch.sum(y, dim=0)
        agg_y_y = torch.sum(y_y, dim=0)
        
        # print(f"agg_c in bcd: {agg_c}")
        # print(f"agg_x in bcd: {agg_x}")
        # print(f"agg_y in bcd: {agg_y}")
        
        x_mean = agg_x / agg_c
        y_mean = agg_y / agg_c

        sd_x = torch.nan_to_num(
            torch.sqrt((agg_x_x - (agg_x ** 2) / agg_c) / (agg_c-1)), nan=1)
        sd_y = torch.nan_to_num(
            torch.sqrt((agg_y_y - (agg_y ** 2) / agg_c) / (agg_c-1)), nan=1)
        # last iteration's discovered vector will become all 0, so sd = 0
        sd_x = torch.where(sd_x == 0, torch.tensor(1), sd_x)
        sd_y = torch.where(sd_y == 0, torch.tensor(1), sd_y)
        # sum of square for normalized x and y both equal to agg_c
        agg_x_y_std = (
            agg_x_y - agg_y*x_mean - agg_x*y_mean + agg_c*x_mean*y_mean)/(sd_x*sd_y)

        # the forward slope and the backward slope shall be the same
        slope = torch.nan_to_num(agg_x_y_std / (agg_c-1), nan=0)
        # print(f"agg_x_y_std in bcd: {agg_x_y_std}")
        # print(f"slope in bcd: {slope}")
        
        # std_treat_vec stays 1D
        std_treat_vec = (treat_vec - treat_vec.mean())/treat_vec.std()
        
        seller_std = (seller_sketch.s-y_mean)/sd_y
        
        cleanup(sd_x, sd_y, agg_x_y_std)
        
        if self.factorized_hist:
            def get_res_min_max(x_min, x_max, y_min, y_max):
                beta_x_min, beta_x_max = slope*x_min, slope*x_max
                res_min = y_min + torch.min(-beta_x_min, -beta_x_max)
                res_max = y_max + torch.max(-beta_x_min, -beta_x_max)
                return torch.min(res_min, dim=0)[0], torch.max(res_max, dim=0)[0]
            
            def assign_bins(x, width):
                bin_indices = (x*(
                    self.bin_num*mfactor- 1)/width).floor().to(torch.int64)
                return bin_indices

            # returns 2 matrices, 
            # the first matrix covers all indices with non-zero counts of each join key
            # the second matrix is the count of each index of corresponding entry of the first matrix in each join key
            # the two matrix has the same size
            # Note that non-zero entries for each join key might not be the same, need to pad some zeros
            def treat_to_hist(c, treat_inds):
                c = c.flatten()
                treat_inds = treat_inds.flatten()
                min_treat = treat_inds.min()
                treat_inds -= min_treat
                total_bins = treat_inds.max() + 1

                partition_indices = torch.repeat_interleave(
                    torch.arange(len(c)).to(self.device), c
                )
                combined_indices = partition_indices * total_bins + treat_inds
                unique_combined, counts = combined_indices.unique(
                    return_counts=True
                )
                
                partition_indices = unique_combined // total_bins
                bin_indices = unique_combined % total_bins
                partition_sizes = torch.bincount(partition_indices)
                max_nonzero = partition_sizes.max().item()
                
                num_partitions = len(c)
                padded_bin_indices = torch.zeros(
                    (num_partitions, max_nonzero), 
                    dtype=torch.int64
                ).to(self.device)
                
                padded_counts = torch.zeros(
                    (num_partitions, max_nonzero), 
                    dtype=torch.int64
                ).to(self.device)

                cumulative_sizes = torch.cat(
                    [torch.tensor([0], device=self.device), 
                     partition_sizes.cumsum(0)[:-1]]
                )

                # Create index tensors for scatter
                row_indices = torch.arange(len(partition_indices)).to(self.device)
                col_indices = row_indices - cumulative_sizes[partition_indices]

                # Perform scatter operations
                padded_bin_indices[partition_indices, col_indices] = bin_indices
                padded_counts[partition_indices, col_indices] = counts
                return (padded_bin_indices + min_treat), padded_counts

            def _merge_histograms(bin_matrices, t_counts, m):
                d = t_sketch.c.size(0) * bin_matrices.size(2)
                bin_matrices = bin_matrices.reshape(m, d).permute(1, 0)
                t_counts = t_counts.reshape(d, 1).expand(-1, m)
                joint_hist = torch.zeros(
                    ((self.bin_num*mfactor)**2, m), 
                    device=self.device, dtype=torch.int64)

                joint_hist.scatter_add_(
                    0, bin_matrices.to(torch.int64), 
                    t_counts)
                if mfactor > 1:
                    # visualize histogram in 2D shape
                    joint_hist = joint_hist.t().reshape(
                        m, self.bin_num*mfactor, self.bin_num*mfactor)
                    # divide 2D histogram into small chunks
                    joint_hist = joint_hist.reshape(
                        m, self.bin_num, mfactor, self.bin_num, mfactor).permute(0, 1, 3, 2, 4)
                    # aggregation the small chunks
                    joint_hist = joint_hist.sum(dim=(3, 4))
                    marg_hist1 = joint_hist.sum(dim=-1).t()
                    marg_hist2 = joint_hist.sum(dim=-2).t()
                    # transform back to original shape
                    joint_hist = joint_hist.transpose(0, 1).reshape(
                        self.bin_num**2, m)
                else:
                    grid_hist = joint_hist.t().reshape(
                        m, self.bin_num*mfactor, self.bin_num*mfactor)
                    marg_hist1 = grid_hist.sum(dim=-1).t()
                    marg_hist2 = grid_hist.sum(dim=-2).t()

                return joint_hist, marg_hist1, marg_hist2
            
            def _outlier_inds(v):
                return torch.where(v <= 14)[0].to(self.device)
            
            t_std_min_join = (t_sketch.min - x_mean) / sd_x
            t_std_max_join = (t_sketch.max - x_mean) / sd_x

            seller_std_min_join = (seller_sketch.min - y_mean) / sd_y
            seller_std_max_join = (seller_sketch.max - y_mean) / sd_y
            
            z_res_min, z_res_max = get_res_min_max(
                t_std_min_join, t_std_max_join, 
                seller_std_min_join, seller_std_max_join
            )

            t_res_min, t_res_max = get_res_min_max(
                seller_std_min_join, seller_std_max_join, 
                t_std_min_join, t_std_max_join
            )
            
            t_std_min, t_std_max = torch.min(
                std_treat_vec), torch.max(std_treat_vec)
            
            seller_std_min = torch.min(seller_std_min_join, dim=0)[0]
            seller_std_max = torch.max(seller_std_max_join, dim=0)[0]
            cleanup(
                t_std_min_join, 
                t_std_max_join,
                seller_std_min_join, 
                seller_std_max_join
            )

            # z_res_min and z_res_max are min and max for z-slope*t
            # bin1(t) * bin_num + bin2(z-slope*t) = bin1(t) * bin_num - bin2(slope*t) + bin2(z)
            z_res_inds = _outlier_inds(z_res_max - z_res_min)
            seller_std_inds = _outlier_inds(
                seller_std_max - seller_std_min)
            t_res_inds = _outlier_inds(
                t_res_max - t_res_min)
            
            concatenated = torch.cat(
                [z_res_inds, seller_std_inds, t_res_inds])

            unique, counts = torch.unique(concatenated, return_counts=True)
            intersect = unique[counts == 3]
            
            w_fwd = torch.max(
                torch.max(z_res_max[intersect] - z_res_min[intersect]), 
                torch.max(t_std_max - t_std_min)
            )
            
            w_bwd = torch.max(
                torch.max(seller_std_max[intersect] - seller_std_min[intersect]), 
                torch.max(t_res_max[intersect] - t_res_min[intersect])
            )
            
            def _entropy_from_hist(hist):
                hist = hist / agg_c[intersect]
                hist[hist == 0] = 1
                return -torch.sum(hist * torch.log(hist), dim=0)
            
            z_res_min = z_res_min[intersect]
            t_res_min = t_res_min[intersect]
            seller_std_min = seller_std_min[intersect]
            slope = slope[intersect]
            
            bin_t = assign_bins(std_treat_vec - t_std_min, w_fwd)
            bin_z = assign_bins(
                seller_std[:, intersect] - slope*t_std_min - z_res_min, w_fwd)
            
            t_bin_matrix, counts_matrix = treat_to_hist(t_sketch.c, bin_t)
            b_slope = self.bin_num*mfactor - slope
#             approx_bin_matrices_fwd = torch.floor(
#                 bin_t*b_slope + torch.matmul(self.trans_mats[join_key], bin_z))
#             print(f'''
# Approximated bins fwd: 
# {approx_bin_matrices_fwd}
# ''')
            
            bin_matrices = torch.floor(
                t_bin_matrix * b_slope.view(-1, 1, 1) + bin_z.t().unsqueeze(-1))
    
            bin_matrices.clamp_(min=0, max=(self.bin_num * mfactor)**2 - 1)
            joint_hist, marg_hist1, marg_hist2 = _merge_histograms(
                bin_matrices, counts_matrix, len(intersect))
            if self.test_histogram:
                with open(f'fhm_fwd_{len(treat_vec)}.pkl', 'wb') as file:
                    pickle.dump(joint_hist, file)
            mi_fwd_ = _entropy_from_hist(marg_hist1) + _entropy_from_hist(
                marg_hist2) - _entropy_from_hist(joint_hist)
            
            mi_fwd = torch.zeros(
                seller_std.shape[1], dtype=mi_fwd_.dtype, device=self.device)
            mi_fwd[intersect] = mi_fwd_
            
            cleanup(
                b_slope, bin_t, bin_z, t_bin_matrix, counts_matrix, 
                bin_matrices, joint_hist, marg_hist1, marg_hist2)

            # bin1(z)*bin_num + bin2(t-slope*z) = bin1(z)*bin_num + bin2(t) - bin2(slope*z)
            bin_t = assign_bins(std_treat_vec, w_bwd)
            bin_z = assign_bins(
                seller_std[:, intersect] - seller_std_min, w_bwd)
            bin_slope_z = assign_bins(
                slope*seller_std[:, intersect] + t_res_min, w_bwd)
            # This has shape (k, m), k is join key domain, m is confounders
            bwd_z_bins = bin_z*self.bin_num*mfactor - bin_slope_z
            t_bin_matrix, counts_matrix = treat_to_hist(t_sketch.c, bin_t)
            
            bin_matrices = t_bin_matrix.unsqueeze(0) + bwd_z_bins.t().unsqueeze(-1)
            bin_matrices.clamp_(min=0, max=(self.bin_num * mfactor)**2 - 1)
#             approx_bin_matrices_bwd = bin_t + torch.matmul(self.trans_mats[join_key], bwd_z_bins)
#             print(f'''
# Approximated bins bwd: 
# {approx_bin_matrices_bwd}
# ''')
            joint_hist, marg_hist1, marg_hist2 = _merge_histograms(
                bin_matrices, counts_matrix, len(intersect))
            if self.test_histogram:
                with open(f'fhm_bwd_{len(treat_vec)}.pkl', 'wb') as file:
                    pickle.dump(joint_hist, file)
            mi_bwd_ = _entropy_from_hist(marg_hist1) + _entropy_from_hist(
                marg_hist2) - _entropy_from_hist(joint_hist)
            mi_bwd = torch.zeros(
                seller_std.shape[1], dtype=mi_fwd_.dtype, device=self.device)
            mi_bwd[intersect] = mi_bwd_

            cleanup(
                bin_t, bin_z, t_bin_matrix, counts_matrix, b_slope, 
                bin_matrices, joint_hist, marg_hist1, marg_hist2)
        else:
            seller_std_vec = torch.matmul(self.trans_mats[join_key].float(), seller_std)
            # z - slope * t
            res_vec_fwd = seller_std_vec - slope*std_treat_vec
            # t - slope * z
            res_vec_bwd = std_treat_vec - slope*seller_std_vec
            
            std_res_fwd = torch.std(res_vec_fwd, dim=0)
            std_res_bwd = torch.std(res_vec_bwd, dim=0)
            std_res_fwd = torch.where(std_res_fwd == 0, torch.tensor(1), std_res_fwd)
            std_res_bwd = torch.where(std_res_bwd == 0, torch.tensor(1), std_res_bwd)

            if self.approx:
                mi_fwd = self._get_entropies(
                    std_treat_vec, 
                    res_vec_fwd/std_res_fwd
                )
                mi_bwd = self._get_entropies(
                    seller_std_vec, 
                    res_vec_bwd/std_res_bwd
                )
            elif self.hist:
                if self.test_histogram:
                    mi_fwd = self._get_mutual_info(
                        std_treat_vec, 
                        res_vec_fwd,
                        suf=f"fwd_{len(treat_vec)}"
                    )
                    mi_bwd = self._get_mutual_info(
                        seller_std_vec, 
                        res_vec_bwd,
                        suf=f"bwd_{len(treat_vec)}"
                    )
                else:
                    mi_fwd = self._get_mutual_info(
                        std_treat_vec, 
                        res_vec_fwd
                    )
                    mi_bwd = self._get_mutual_info(
                        seller_std_vec, 
                        res_vec_bwd
                    )
            else:
                raise Exception("Please specify a method to estimate Mutual Information")
        # since mutual information shall never be smaller than 0, we cap by 0 to be more accurate
        mi_bwd = mi_bwd.clamp_(min=0)
        mi_diffs = mi_fwd - mi_bwd
        return mi_fwd, mi_bwd, mi_diffs, seller_std
    
    def preprocess_compute_te(self, df, join_keys, treatment, outcome):
        self.conf_set[(treatment, outcome)] = set()
        self.buyer_join_keys = join_keys
        self.treatment = treatment
        self.outcome = outcome
        self.cur_data_in = df
        # Currently based on Rice Rule
        # TODO: if there is outlier in the data, may want to other ways of doing so i.e. Freedman-Diaconis Rule
        self.bin_num = int(2*len(df)**(1/3))
        
        # Align join key domains to that of the treatment
        def _get_trans_matrix(c):
            num_rows = torch.sum(c).item()
            num_columns = c.size()[0]
            trans_matrix = torch.zeros(
                num_rows, num_columns, dtype=torch.int64
            )
            c = torch.cumsum(c, dim=0)
            row_indices = torch.arange(num_rows)
            for i in range(num_columns):
                start_index = c[i-1].item() if i > 0 else 0
                end_index = c[i].item()
                trans_matrix[start_index:end_index, i] = 1
            return trans_matrix.to(self.device)
        
        for join_key in join_keys:
            cur_domain = {}
            for col in join_key:
                cur_domain[col] = self.dm.join_key_domains[col]
            if tuple(join_key) in self.dm.seller_join_sketches:
                treat_out_instance = JoinSketch(
                    join_key_domain=cur_domain, is_buyer=True, device=self.device, 
                    join_key_index=self.dm.seller_join_sketches[tuple(join_key)].join_key_index)
                treat_out_instance.register_df(
                    0, 
                    self.cur_data_in[join_key + [treatment, outcome]], 
                    [treatment, outcome]
                )
                self.treat_out_sketches[tuple(join_key)] = treat_out_instance
                self.cross_t_o[tuple(join_key)] = treat_out_instance.get_cross_term(
                    self.cur_data_in,
                    treatment,
                    outcome
                )
                
                self._align_treat_out_vec(
                    join_key, 
                    treat_out_instance
                )
                # Now the treat and out vector are aligned
                # we want to compute the trans matrix by the count in the sketch
                c = treat_out_instance.sketch_loader.batch_sketches[0].c[:, 0]
                self.trans_mats[tuple(join_key)] = _get_trans_matrix(c)
    
    def compute_treatment_effect(self, df, join_keys, treatment, outcome, conf_type=None, search_iters=None):
        # s = time.time()
        if conf_type is not None:
            if conf_type not in {'pos', 'neg'}:
                raise Exception(f"{conf_type} is not a valid confounder type")
            self.conf_type = conf_type
        
        t = time.time()
        self.preprocess_compute_te(df, join_keys, treatment, outcome)
        preprocess_time = time.time() - t
        
        t = time.time()
        conf_size = len(self.conf_set[(treatment, outcome)])
        # print(f"Total time spent on initializing: {t-s}")
        end_to_end_time = 0
        cum_time = []
        update_df_time, update_t_o_time, update_cor_time, search_time = 0, 0, 0, 0
        ate_iters = []
        while True:
            s = time.time()
            def std_v(v):
                mean = v.mean()
                std = v.std()
                return (v - mean) / std
            if self.verbose:
                print(f"Current coeff is {causal_effect(self.cur_data_in, treatment, outcome)}")
            ate_iters.append(causal_effect(self.cur_data_in, treatment, outcome))
            self.t_o_mi = self._get_mutual_info(
                std_v(
                    torch.tensor(self.cur_data_in[treatment].values.reshape(-1, 1), 
                                 dtype=torch.float32).to(self.device)), 
                std_v(torch.tensor(self.cur_data_in[outcome].values.reshape(-1, 1), 
                             dtype=torch.float32).to(self.device))
            )[0]

            cur_dft, cur_tot, cur_cort, cur_st = self.search_one_iter(treatment, outcome)
            if self.test_histogram:
                return 0, 0, 0, 0, 0, 0, 0

            update_df_time += cur_dft
            update_t_o_time += cur_tot
            update_cor_time += cur_cort
            search_time += cur_st
            if (len(self.conf_set[(treatment, outcome)]) == conf_size) or (
                search_iters is not None and len(
                    self.conf_set[(treatment, outcome)]) == search_iters):
                break
            conf_size += 1
            cum_time.append(time.time() - s)
        end_to_end_time = time.time() - t
        
        if self.verbose:
            if len(self.conf_set[(treatment, outcome)]) != 0:
                print(f"Avg update df time: {update_df_time/len(self.conf_set[(treatment, outcome)])}")
                print(f"Avg update TO time: {update_t_o_time/len(self.conf_set[(treatment, outcome)])}")
                print(f"Avg update corpus time: {update_cor_time/len(self.conf_set[(treatment, outcome)])}")
            else:
                print(f"Avg update df time: {update_df_time}")
                print(f"Avg update TO time: {update_t_o_time}")
                print(f"Avg update corpus time: {update_cor_time}")
            print(f"Avg search time: {search_time/(len(self.conf_set[(treatment, outcome)])+1)}")
            print(f"Discovered set of confounders: {self.conf_set[(treatment, outcome)]}")
        return causal_effect(self.cur_data_in, treatment, outcome), preprocess_time, \
               end_to_end_time, search_time, update_cor_time, cum_time, ate_iters
        
    def search_one_iter(self, treatment, outcome):
        s = time.time()
        conf_join_key, f_opt_batch_id, f_opt_ind = self.discover_confounder()
        t = time.time()
        # TODO: What if there is no causal effect between X and Y?
        if conf_join_key is None or self.test_histogram: 
            return 0, 0, 0, t-s
        if f_opt_batch_id not in self.exclude_ind:
            self.exclude_ind[f_opt_batch_id] = {f_opt_ind}
        else:
            self.exclude_ind[f_opt_batch_id].add(f_opt_ind)
        seller_id, cur_feature = self.dm.seller_join_sketches[
            conf_join_key].get_seller_by_feature_index(f_opt_batch_id, f_opt_ind)
        
        conf_sketch = self.dm.seller_join_sketches[conf_join_key]
        conf_vec = conf_sketch.sketch_loader.get_sketches(
            f_opt_batch_id
        ).s[:, f_opt_ind]
        
        self.conf_set[(treatment, outcome)].add(
            (conf_join_key, seller_id, cur_feature, f_opt_batch_id, f_opt_ind))
        update_df_time, update_t_o_time, update_corpus_time = self._update_corpus_to_res(
            seller_id, 
            cur_feature, 
            conf_join_key, 
            f_opt_batch_id, 
            f_opt_ind,
            conf_vec
        )
        return update_df_time, update_t_o_time, update_corpus_time, t - s  

    # get a varibale Z that 
    # (1) is an ancestor of T
    # (2) has no confounder between T 
    # (3) conditioning on Z reduces mutual information between T and O
    def discover_confounder(self):
        def score_based_select(
            join_key, 
            batch_id, 
            fil_mi_diff_inds, 
            fil_mi_diff_sgf, 
            fil_conf_vecs
        ):
            if len(fil_mi_diff_sgf) > 0:
                max_mi_diff_ind = torch.argmax(fil_mi_diff_sgf)
                return fil_mi_diff_inds[max_mi_diff_ind], fil_mi_diff_sgf[
                    max_mi_diff_ind], fil_conf_vecs[:, max_mi_diff_ind]
            else:
                return -1, -1, -1, -1

        def get_t_o_res_vec(t_sketch, o_sketch, cand_conf_sketch, cand_conf_vecs):
            t_res_vecs = self._get_exp_res(
                cand_conf_sketch, t_sketch, cand_conf_vecs, self.treat_vecs[join_key])
            o_res_vecs = self._get_exp_res(
                cand_conf_sketch, o_sketch, cand_conf_vecs, self.out_vecs[join_key])
            mi = self._get_mutual_info(t_res_vecs, o_res_vecs)
            return mi
                
        
        def get_source(proj_x, sketch):
            def _split_sketch(i):
                exp_sketch, targ_sketch = {}, {}
                for key, val in sketch.items():
                    exp_sketch[key] = val[:, i:i+1]
                    targ_sketch[key] = torch.cat((val[:, :i], val[:, i+1:]), dim=1)
                return exp_sketch, targ_sketch

            sketch_dim = proj_x.size(1)
            score_list = torch.zeros(sketch_dim)
            for i in range(sketch_dim):
                exp_sketch, targ_sketch = _split_sketch(i)
                res_vec_std = self._get_exp_res(
                    exp_sketch, targ_sketch, 
                    proj_x[:, i:i+1], 
                    torch.cat((proj_x[:, :i], proj_x[:, i+1:]), dim=1)
                )
                r_res_vec_std = self._get_exp_res(
                    targ_sketch, exp_sketch, 
                    torch.cat((proj_x[:, :i], proj_x[:, i+1:]), dim=1), 
                    proj_x[:, i:i+1]
                )
                if not self.approx:
                    mi = self._get_mutual_info(proj_x[:, i:i+1], res_vec_std)
                    mi_r = self._get_mutual_info(
                        torch.cat((proj_x[:, :i], proj_x[:, i+1:]), dim=1), 
                        r_res_vec_std
                    )
                else:
                    mi = self._get_entropies(proj_x[:, i:i+1], res_vec_std)
                    mi_r = self._get_entropies(
                        torch.cat((proj_x[:, :i], proj_x[:, i+1:]), dim=1), 
                        r_res_vec_std
                    )
                mi_diff = mi - mi_r
                mi_diff[mi_diff < 0] = 0
                score_list[i] += torch.sum(mi_diff**2).item()
            return score_list

        conf_join_key, max_batch_id, max_ind, max_mi_diff = None, -1, -1, 0
        min_score = math.inf
        for join_key, treat_out_sketch in self.treat_out_sketches.items():
            t_sketch = treat_out_sketch.sketch_loader.get_sketches(0, inds=[0])
            if join_key not in self.dm.seller_join_sketches: continue
            else:
                search_sketch = self.dm.seller_join_sketches[join_key]
                for batch_id in range(search_sketch.sketch_loader.num_batches):
                    s_sketch = search_sketch.sketch_loader.get_sketches(batch_id)

                    mi, mi_r, mi_diffs, seller_std = self.bivariate_causal_discovery(
                        t_sketch, s_sketch, join_key, mfactor=self.factor
                    )
                    
                    if self.verbose:
                        print("="*50)
                        for i in range(len(mi)):
                            conf = self.dm.seller_join_sketches[
                                join_key].get_seller_by_feature_index(batch_id, i)
                            print(f'''
Dataset is {self.dm.seller_datasets[conf[0]][1]}, 
Confounder is {conf[1]}, 
MI fwd is {mi[i]},
MI bwd is {mi_r[i]},
MI difference is {mi_diffs[i]}
''')

                    mi_diff_inds = torch.where(mi_diffs > self.mi_threshold)[0]
                    mi_diff_sgf = mi_diffs[mi_diff_inds]
                    
                    if len(mi_diff_inds) == 0:
                        continue
                    
                    # sgf_sketch = search_sketch.sketch_loader.get_sketches(batch_id, mi_diff_inds)

                    conf_vecs = torch.matmul(
                        self.trans_mats[join_key].float(), seller_std[:, mi_diff_inds])
                    cmi = get_t_o_res_vec(
                        treat_out_sketch.sketch_loader.get_sketches(0, inds=[0]), 
                        treat_out_sketch.sketch_loader.get_sketches(0, inds=[1]), 
                        search_sketch.sketch_loader.get_sketches(
                            batch_id, inds=mi_diff_inds
                        ), 
                        conf_vecs
                    )
                    
                    if self.verbose:
                        print("="*50)
                        print(f"MI Treatmen/Outcome w/o condition: {self.t_o_mi}")
                        for i in range(len(cmi)):
                            conf = self.dm.seller_join_sketches[
                                join_key].get_seller_by_feature_index(batch_id, mi_diff_inds[i])
                            print(f'''
Dataset is {self.dm.seller_datasets[conf[0]][1]}, 
Confounder is {conf[1]}, 
MI conditioned on Confounder is {cmi[i]}
''')

                    epsilon = 1e-8  # small value to avoid nan
                    condition = torch.where(
                        torch.abs(cmi - self.t_o_mi) / (
                            abs(self.t_o_mi) + epsilon) > self.err
                    )[0]
                    if len(condition) == 0:
                        continue

                    # get valid var ind and their slope
                    cur_ind, cur_mi_diff, cur_conf_vec = score_based_select(
                        join_key, batch_id, 
                        mi_diff_inds[condition], 
                        mi_diff_sgf[condition],
                        conf_vecs[:, condition]
                    )

                    if cur_ind == -1:
                        continue

                    if cur_mi_diff >= max_mi_diff:
                        conf_join_key = join_key
                        max_mi_diff = cur_mi_diff
                        max_batch_id = batch_id
                        max_ind = cur_ind

        if self.verbose and conf_join_key is not None:
            conf = self.dm.seller_join_sketches[
                conf_join_key].get_seller_by_feature_index(max_batch_id, max_ind)
            print(f'''Dataset is {self.dm.seller_datasets[conf[0]][1]}, 
            Confounder is {conf[1]}''')

        return conf_join_key, max_batch_id, max_ind
    
    # Assumption: we assume the scope of datasets to be considered are the ones can be integrated with
    # the requestor dataset by a PK-FK join, and join keys are clusterd.
    # Here, we take the column of the discovered confounder, join with treatment sketch
    def _update_corpus_to_res(self, seller_id, conf_feature, conf_join_key, conf_batch_id, ind, conf_vec):
        update_df_time, update_t_o_time, update_corpus_time = 0, 0, 0
        # we assume the join over n-to-1 fanout can be modeled with an ANM
        def _get_residual(slope, intercept, y_sketch, x_sketch, is_seller=False):
            x = {0: x_sketch.c, 1: x_sketch.s, 2: x_sketch.Q}
            y = {0: y_sketch.c, 1: y_sketch.s, 2: y_sketch.Q}
            res = {0: y[0]}
            res[1] = torch.nan_to_num(
                (y[1]*x[0]-slope*x[1]*y[0]-x[0]*y[0]*intercept)/(x[0]), 
                nan=0)
            res[2] = res[1]**2
            return SemiRing(res[0], res[1], res[2], res[1], res[1], self.device)
        
        # this function updates the seller sketch
        def _get_res_sketch(target_sketch, exp_sketch, conf_batch_id, ind, is_seller=False):
            e_sketch = exp_sketch.sketch_loader.get_sketches(conf_batch_id, [ind])
            for batch_id in range(target_sketch.sketch_loader.num_batches):
                targ_sketch = target_sketch.sketch_loader.get_sketches(batch_id)
                slope, intercept = self._get_slope_intercept(
                    targ_sketch, e_sketch
                )
                # print(f"slope in update residual: {slope}")
                res_sketch = _get_residual(slope, intercept, targ_sketch, e_sketch, is_seller=True)
                
                if target_sketch is exp_sketch and batch_id == conf_batch_id:
                    res_sketch.set_ind_zero(ind)
                target_sketch.sketch_loader.update_sketch(res_sketch, batch_id)
        
        def _update_df_treat_out(df, conf_col):
            X = df[[conf_col, self.treatment, self.outcome]].dropna().values
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            XTX = X.T @ X
            
            conf_sum, t_sum, o_sum = XTX[0, 1], XTX[0, 2], XTX[0, 3]
            t_conf_sum, o_conf_sum = XTX[1, 2], XTX[1, 3]
            c, conf_2_sum = XTX[0, 0], XTX[1, 1]
            
            conf_mean = conf_sum / c
            t_mean = t_sum / c
            o_mean = o_sum / c
            
            S_xx = conf_2_sum - 2*conf_mean*conf_sum + c*conf_mean**2
            S_xt = t_conf_sum - conf_mean*t_sum - conf_sum*t_mean + c*conf_mean*t_mean
            S_xo = o_conf_sum - conf_mean*o_sum - conf_sum*o_mean + c*conf_mean*o_mean
            
            slope_xt = np.nan_to_num(S_xt / S_xx, nan=0)
            intercept_xt = t_mean - slope_xt*conf_mean
            slope_xo = np.nan_to_num(S_xo / S_xx, nan=0)
            intercept_xo = o_mean - slope_xo*conf_mean
            
            df[self.treatment] = df[self.treatment] - slope_xt*df[
                conf_col] - intercept_xt
            df[self.outcome] = df[self.outcome] - slope_xo*df[
                conf_col] - intercept_xo
        
        if conf_join_key not in self.dm.seller_join_sketches:
            raise Exception(f"Join key cluster {conf_join_key} not found")
        s = time.time()
        conf_sketch = self.dm.seller_join_sketches[conf_join_key]
        # Must use join here because buyer sketch might contain more than one join key
        # join the semi-ring and do update
        conf_df = pd.DataFrame(
            {conf_feature: conf_vec.cpu().numpy()}, 
            index=conf_sketch.join_key_index)
        
        # There is space for optimization here, since the data is aligned, 
        # we can just concatenate data together instead of performing a join
        join_df = self.cur_data_in.merge(
            conf_df, on=conf_sketch.join_keys, how='left'
        )
        _update_df_treat_out(join_df, conf_feature)
        
        self.cur_data_in = join_df[self.cur_data_in.columns]
        update_df_time += time.time() - s
        
        # update clustered sketches in the data corpus
        # only consider those joinable to the input dataset
        for join_key in self.treat_out_sketches.keys():
            s = time.time()
            if join_key not in self.dm.seller_join_sketches: continue
            
            treat_out_instance = JoinSketch(
                join_key_domain=self.treat_out_sketches[join_key].join_key_domain, 
                is_buyer=True, 
                device=self.device
            )
            treat_out_instance.register_df(
                0, 
                self.cur_data_in[
                    list(join_key) + [self.treatment, self.outcome]], 
                [self.treatment, self.outcome]
            )
            self.treat_out_sketches[join_key] = treat_out_instance
            # print(f"Updated: {treat_out_instance.sketch_loader.batch_sketches[0].s}")
            self.cross_t_o[join_key] = treat_out_instance.get_cross_term(
                self.cur_data_in,
                self.treatment,
                self.outcome
            )

            self._align_treat_out_vec(
                join_key, 
                treat_out_instance
            )

            update_t_o_time += time.time() - s
            # Update corpus sketches, including itself to 0
            s = time.time()
            if join_key == conf_join_key:
                _get_res_sketch(
                    self.dm.seller_join_sketches[join_key], 
                    conf_sketch, 
                    conf_batch_id, 
                    ind,
                    is_seller=True
                )
            else:
                cur_domain = {}
                for col in join_key:
                    cur_domain[col] = self.dm.join_key_domains[col]
                conf_update_sketch = JoinSketch(join_key_domain=cur_domain, device=self.device)
                conf_update_sketch.register_df(
                    0, 
                    join_df[list(join_key) + [conf_feature]], 
                    [conf_feature],
                    agg='mean'
                )
                _get_res_sketch(
                    self.dm.seller_join_sketches[join_key], 
                    conf_update_sketch, 
                    0, 
                    0,
                    is_seller=True
                )
            update_corpus_time += time.time() - s
        return update_df_time, update_t_o_time, update_corpus_time
    
    # v and u are original data vectors, this function estimates the differential entropy of a variable
    def _get_entropies(self, v, u):
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        const = torch.tensor((1 + np.log(2 * np.pi)) / 2, device = self.device)
        def _taylor_approx(agg_sketch, func="log_cosh"):
            if func not in {"log_cosh", "x_exp"}:
                raise ValueError("Invalid function to Approx.")
            if func == "log_cosh":
                taylor_coeff = {
                    2: 1/2,
                    4: -1/12,
                    6: 1/45
                }
            else:
                taylor_coeff = {
                    1: 1,
                    3: -1/2,
                    5: 1/8
                }
            res = 0
            for key, val in agg_sketch.items():
                if key in taylor_coeff:
                    res += taylor_coeff[key] * val
                else: continue
            return res
            
        def _entropy_semi_ring(sketch):
            agg_sketch = {}
            for key, val in sketch.items():
                agg_sketch[key] = torch.sum(val, dim=0)
            log_cosh_term = -k1*(_taylor_approx(
                agg_sketch, func="log_cosh")/agg_sketch[0]-gamma)**2
            x_exp_term = -k2*(_taylor_approx(
                agg_sketch, func="x_exp")/agg_sketch[0])**2
            return const + x_exp_term + log_cosh_term
        
        def _entropy(x):
            return const - k1*(
                torch.mean(torch.log(torch.cosh(x)), dim=0)-gamma)**2 - k2*(
                torch.mean(x * torch.exp(-x**2 / 2), dim=0))**2
        return _entropy(u) + _entropy(v)
    
    def _get_mutual_info(self, v, u, suf=None):
        def _mutual_info_from_hist(hist, c):
            hist = hist / c
            # Avoid log 0
            hist[hist == 0] = 1
            return -torch.sum(hist * torch.log(hist), dim=0)
        
        def _assign_bins(x, min_val, width):
            bin_indices = ((x - min_val) * (
                self.bin_num - 1)/width).floor().to(torch.int64)
            return bin_indices

        v_max, v_min = torch.max(v, dim=0)[0], torch.min(v, dim=0)[0]
        u_max, u_min = torch.max(u, dim=0)[0], torch.min(u, dim=0)[0]
        u_width = torch.max(u_max - u_min)
        v_width = torch.max(v_max - v_min)
        width = max(u_width, v_width)
        # print(f"v_max: {v_max}")
        # print(f"v_min: {v_min}")
        # print(f"u_max: {u_max}")
        # print(f"u_min: {u_min}")

        if u_width == 0 or v_width == 0:
            return torch.zeros(
                max(u_width.shape[1], v_width.shape[1]))

        seller_inds = _assign_bins(v, v_min, width)
        
        res_inds = _assign_bins(u, u_min, width)
        
        aligned_inds = seller_inds * self.bin_num + res_inds
        d = aligned_inds.shape[-1]

        x_hist = torch.zeros((self.bin_num, seller_inds.shape[1]), 
                             device=self.device, dtype=torch.float)
        y_hist = torch.zeros((self.bin_num, res_inds.shape[1]), 
                             device=self.device, dtype=torch.float)
        joint_hist = torch.zeros(
            (self.bin_num * self.bin_num, d), device=self.device, dtype=torch.float)

        x_hist.scatter_add_(
            0, seller_inds, torch.ones_like(seller_inds, dtype=torch.float))
        y_hist.scatter_add_(
            0, res_inds, torch.ones_like(res_inds, dtype=torch.float))

        joint_hist.scatter_add_(
            0, aligned_inds, 
            torch.ones_like(aligned_inds, dtype=torch.float))
        if suf is not None:
            with open(f'hist_{suf}.pkl', 'wb') as file:
                pickle.dump(joint_hist, file)
        c = len(v)
        H_XY = _mutual_info_from_hist(joint_hist, c)
        H_X = _mutual_info_from_hist(x_hist, c)
        H_Y = _mutual_info_from_hist(y_hist, c)

        return H_X + H_Y - H_XY
    
    @staticmethod
    # in this function, all nan's will be filled with 0, this has no problem
    def _get_slope_intercept(y_sketch, x_sketch):
        c_x, s_x, Q_x = x_sketch.c, x_sketch.s, x_sketch.Q
        c_y, s_y, Q_y = y_sketch.c, y_sketch.s, y_sketch.Q

        y_y = torch.sum(Q_y * c_x, dim=0)
        y = torch.sum(s_y * c_x, dim=0)
        c = torch.sum(c_y * c_x, dim=0)
        x_y = torch.sum(s_x * s_y, dim=0)
        x_x = torch.sum(Q_x * c_y, dim=0)
        x = torch.sum(s_x * c_y, dim=0)
        
        x_mean = x / c
        y_mean = y / c

        S_xx = x_x - 2 * x_mean * x + c * x_mean ** 2
        S_xy = x_y - x_mean * y - x * y_mean + c * x_mean * y_mean
        
        slope = torch.nan_to_num(
            torch.where(S_xx != 0, S_xy / S_xx, torch.tensor(0.0)), nan=0)
        intercept = y_mean - slope * x_mean
        
        cleanup(S_xx, S_xy, x, x_x, x_y, c, y, y_y)
        return slope, intercept
    
    def _get_cross_term_residual(
        self, 
        t_slope, 
        t_intercept, 
        o_slope, 
        o_intercept, 
        cross_t_o,
        t_sketch, 
        o_sketch, 
        c_sketch
    ):
        return cross_t_o*c_sketch[0] - o_slope*t_sketch[1]*c_sketch[1] - \
               o_intercept*t_sketch[1]*c_sketch[0] - t_slope*o_sketch[1]*c_sketch[1] + \
               t_slope*o_slope*t_sketch[0]*c_sketch[2] + t_slope*o_intercept*t_sketch[0]*c_sketch[1] - \
               t_intercept*o_sketch[1]*c_sketch[0] + t_intercept*o_slope*t_sketch[0]*c_sketch[1] + \
               t_intercept*o_intercept*t_sketch[0]*c_sketch[0]
    
    # def prune_confounders(self, buyer_df, dm, k):        
    #     seller_feature_inds = {}
    #     for ele in cd.conf_set[(self.treatment, self.outcome)]:
    #         join_key, dataset, feature, f_batch, f_ind = ele[0], ele[1], ele[2], ele[3], ele[4]
    #         if join_key in seller_feature_inds:
    #             if f_batch in seller_feature_inds[join_key]:
    #                 seller_feature_inds[
    #                     join_key][f_batch]['features'].append(f'{dataset}:{feature}')
    #                 seller_feature_inds[
    #                     join_key][f_batch]['inds'].append(f_ind)
    #             else:
    #                 seller_feature_inds[join_key][f_batch] = {
    #                     'feature': [f'{dataset}:{feature}'],
    #                     'inds': [f_ind]
    #                 }
    #         else:
    #             seller_feature_inds[join_key] = {
    #                 f_batch: {
    #                     'feature': [f'{dataset}:{feature}'],
    #                     'inds': [f_ind]
    #                 }
    #             }
        
    #     for i in range(k):
    #         max_change = 0
    #         for join_key, batch_inds in seller_feature_inds.items():
    #             search_sketch = dm.seller_join_sketches[join_key]
    #             for batch_id in batch_inds.keys():
    #                 inds = torch.tensor(seller_feature_inds[
    #                     join_key][batch_id]['inds']).to(self.device)
    #                 s_sketch = search_sketch.sketch_loader.get_sketches(
    #                     batch_id, inds
    #                 ).s
                    
    #                 cols = seller_feature_inds[
    #                     join_key][batch_id]['feature']

    #                 # Create DataFrame
    #                 conf_df = pd.DataFrame(
    #                     data=s_sketch.cpu().numpy(),
    #                     columns=cols,
    #                     index=conf_sketch.join_key_index
    #                 )
                    
    #                 join_df = buyer_df.merge(
    #                     conf_df, on=conf_sketch.join_keys, how='left'
    #                 )


if __name__ == "__main__":
    num_nodes = [100]
    runs = 100
    mi_threshold = 0.02
    for num_node in num_nodes:
        se = 0
        for run in range(runs):
            with open(
                f'experiment/datasets/synthetic/data_{num_node}_{run}.pkl',
                'rb'
            ) as file:
                dp, treatment, target = pickle.load(file)
                dm = DataMarket()
                dm.add_seller(dp.data_corpus, "synthetic", [[dp.join_key]], dp.join_key_domain, 
                              [col for col in dp.data_corpus if col != dp.join_key])
                cd = ConDiscovery(dm, hist=True, mi_threshold=mi_threshold)
                est_suna, prep_time, e2e_time, search_time, update_time, _, _ = cd.compute_treatment_effect(
                    dp.data_in, [['join_key']], treatment, target)
                parents = set(list(dp.G.predecessors(treatment)))
                gt = causal_effect(dp.D, treatment, target, parents)
                print(f"Treatment: {treatment}, Outcome: {target}, Error: {(gt - est_suna) ** 2}, Estimation: {est_suna}, Ground Truth: {gt}")
                se += (gt - est_suna) ** 2
        print(f"This is SE: {se}")







