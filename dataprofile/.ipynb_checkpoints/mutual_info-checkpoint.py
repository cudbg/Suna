from abc import ABC, abstractmethod
import torch

eps = 1e-5


# Assumes that the X and Y vectors are sorted according to the join key values
# and are captured by the c field in their semi-ring structures.
class MIEstimator(ABC):
    """Abstract base class for mutual information estimators for BCD."""

    def __init__(self, device: str = 'cpu'):
        self.is_fitted = False
        self.device = device

    @staticmethod
    def _assign_bins(x, min_val, width):
        bin_indices = ((x - min_val) // width).to(torch.int64)
        return bin_indices

    def _entropy_from_inds(
        self, hist_size, ind_unique, ind_inverse, counts, pad_inds, save=False
    ):
        hist = torch.zeros((hist_size, ind_inverse.shape[1]),
                           device=self.device, dtype=torch.float)
        hist.scatter_add_(0, ind_inverse, counts)
        # filter_ = torch.arange(hist.shape[0], device=self.device)
        # mask = filter_[:, None] > pad_inds[None, :]
        if save:
            import pickle
            n = int(torch.sum(hist[:, 0]).item())
            itm = eps + hist / torch.sum(hist[:, 0])
            ent = -torch.sum(itm * torch.log(itm), dim=0)
            with open(f'hist_{n}.pkl', 'wb') as file:
                pickle.dump((hist, ent), file)
        hist = eps + hist / torch.sum(hist[:, 0])
        # hist[mask] = 1
        return -torch.sum(hist * torch.log(hist), dim=0)

    @abstractmethod
    def fit(self, be, coeff_xy, coeff_yx):
        """
        Estimate the mutual information between X and Y.

        Returns:
            float: Estimated mutual information value
        """

    @abstractmethod
    def compute_mi_diff(self) -> float:
        """
        Estimate the mutual information between X and Y.

        Returns:
            float: Estimated mutual information value
        """
        if not self.is_fitted:
            raise ValueError(
                "Estimator must be fitted before calling compute_mi()")


class HistMI(MIEstimator):
    """Naive histogram-based mutual information estimator."""

    def __init__(self, device: str = 'cpu'):
        super().__init__(device)

    def _get_join_rows(self, X, c_x, c_y):
        bounds = torch.cumsum(c_x, 0)
        starts = torch.cat([torch.tensor([0], device=self.device), bounds[:-1]])

        indices = torch.cat(
            [torch.arange(start, end).repeat(rep) for start, end, rep in zip(
                starts, bounds, c_y
            )]).to(self.device)
        return torch.index_select(X, 0, indices)

    # Materializes X and Y and computes prediction residuals
    def fit(self, be, coeff_xy, coeff_yx, std=True):
        # Assume that the dimension of the coeff (k, m) is a polynomial regression
        # with degree k - 1
        deg = coeff_xy.shape[0] - 1
        X_join = self._get_join_rows(be.X, be.c_x, be.c_y)
        Y_join = torch.repeat_interleave(
            be.Y, torch.repeat_interleave(be.c_x, be.c_y), dim=0)

        X_powers = torch.stack([X_join ** i for i in range(deg + 1)], dim=-1)
        Y_powers = torch.stack([Y_join ** i for i in range(deg + 1)], dim=-1)

        pred_y = torch.sum(X_powers * coeff_xy.t().unsqueeze(0), dim=2)
        pred_x = torch.sum(Y_powers * coeff_yx.t().unsqueeze(0), dim=2)

        self.x = X_join
        self.y = Y_join
        self.res_y = Y_join - pred_y
        self.res_x = X_join - pred_x
        # print(f"True res residual: {torch.std(self.res, unbiased=True, dim=0)}")
        if std:
            self.res_x = torch.nan_to_num(
                self.res_x / torch.std(self.res_x, unbiased=True, dim=0), nan=1)
            self.res_y = torch.nan_to_num(
                self.res_y / torch.std(self.res_y, unbiased=True, dim=0), nan=0)
        self.is_fitted = True

    def _mi_from_data(self, X, Y):
        x_min = torch.min(X, dim=0)[0]
        y_min = torch.min(Y, dim=0)[0]
        c = 3.49 * (len(X)**(-1/3))
        std_x = torch.std(X, dim=0)
        std_y = torch.std(Y, dim=0)

        x_width, y_width = c * std_x, c * std_y
        x_width[x_width == 0] = 1
        y_width[y_width == 0] = 1

        x_inds = self._assign_bins(X, x_min, x_width)
        y_inds = self._assign_bins(Y, y_min, y_width)
        x_ind_unique, x_ind_inverse = torch.unique(x_inds, return_inverse=True)
        y_ind_unique, y_ind_inverse = torch.unique(y_inds, return_inverse=True)
        x_bins = torch.max(x_inds, dim=0)[0] - torch.min(x_inds, dim=0)[0] + 1
        y_bins = torch.max(y_inds, dim=0)[0] - torch.min(y_inds, dim=0)[0] + 1

        x_hist_size = torch.max(x_bins)
        y_hist_size = torch.max(y_bins)

        H_x = self._entropy_from_inds(
            x_hist_size, x_ind_unique, x_ind_inverse,
            torch.ones_like(x_ind_inverse, dtype=torch.float), x_bins)
        H_y = self._entropy_from_inds(
            y_hist_size, y_ind_unique, y_ind_inverse,
            torch.ones_like(y_ind_inverse, dtype=torch.float), y_bins)

        comb_inds = x_inds * y_bins + y_inds
        comb_ind_unique, comb_ind_inverse = torch.unique(
            comb_inds, return_inverse=True)
        H_xy = self._entropy_from_inds(
            x_hist_size*y_hist_size, comb_ind_unique, comb_ind_inverse,
            torch.ones_like(comb_ind_inverse, dtype=torch.float),
            torch.max(comb_inds, dim=0)[0] - torch.min(comb_inds, dim=0)[0] + 1
        )
        return H_x, H_y, H_xy

    def compute_mi_diff(self):
        super().compute_mi_diff()
        H_x, H_res_y, H_x_res_y = self._mi_from_data(self.x, self.res_y)
        H_y, H_res_x, H_y_res_x = self._mi_from_data(self.y, self.res_x)
        # print(f"This is MI(X, res_y): {H_x + H_res_y - H_x_res_y}")
        # print(f"This is MI(Y, res_x): {H_y + H_res_x - H_y_res_x}")
        # return H_x + H_res_y - H_y - H_res_x
        return H_x + H_res_y - H_x_res_y - (H_y + H_res_x - H_y_res_x)


class NNMI(MIEstimator):
    """Naive nearest-neighbor-based mutual information estimator."""

    def compute_mi_diff(self) -> float:
        super().compute_mi_diff()  # Check if fitted
        # Implementation for naive NN MI estimation
        # This would involve finding nearest neighbors
        # and computing MI using NN statistics
        return 0.0  # Placeholder


# Specialized class to support linear regression + std, could extend
# to polynomial regression (requires some engineering, but it will work)
class FactorizedLinearHistMI(MIEstimator):
    """Factorized histogram-based mutual information estimator specialized for linear models."""

    def __init__(self, device: str = 'cpu', mult: int = 1):
        super().__init__(device)
        self.mult = mult

    def _get_min_max(self, X, c):
        num_groups = len(c)
        key_inds = torch.repeat_interleave(
            torch.arange(num_groups).to(self.device),
            c
        )

        X_max = torch.full(
            (num_groups, X.shape[1]),
            float('-inf'), dtype=X.dtype, device=self.device)
        X_min = torch.full(
            (num_groups, X.shape[1]),
            float('inf'), dtype=X.dtype, device=self.device)

        expanded_keys = key_inds.unsqueeze(1).expand(-1, X.shape[1])
        X_max = X_max.scatter_reduce(0, expanded_keys, X, reduce='amax')
        X_min = X_min.scatter_reduce(0, expanded_keys, X, reduce='amin')
        return X_min, X_max

    def treat_to_hist(self, c, treat_inds):
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
        row_indices = torch.arange(
            len(partition_indices)).to(
            self.device)
        col_indices = row_indices - cumulative_sizes[partition_indices]

        # Perform scatter operations
        padded_bin_indices[partition_indices,
                           col_indices] = bin_indices
        padded_counts[partition_indices, col_indices] = counts
        return (padded_bin_indices + min_treat), padded_counts

    @staticmethod
    def get_res_min_max(x_min, x_max, y_min, y_max, beta, sd):
        beta_x_min, beta_x_max = beta * x_min, beta * x_max
        res_y_min = y_min + torch.min(-beta_x_min, -beta_x_max)
        res_y_max = y_max + torch.max(-beta_x_min, -beta_x_max)
        return torch.min(res_y_min, dim=0)[0] / sd, \
            torch.max(res_y_max, dim=0)[0] / sd

    # assume be.X is the many and be.Y is the one
    def fit(self, be, coeff_xy, coeff_yx):
        c = 3.49 * (torch.sum(be.c_x * be.c_y) ** (-1/3))
        self.c_y = be.c_x.reshape(-1, 1).expand(
            -1, coeff_xy.shape[1]).to(torch.float)
        beta = coeff_xy[1]
        sd_res = torch.sqrt(1 - beta ** 2)
        x_min, x_max = self._get_min_max(be.X, be.c_x)
        y_min, y_max = self._get_min_max(be.Y, be.c_y)
        res_y_min, res_y_max = self.get_res_min_max(
            x_min, x_max, y_min, y_max, beta, sd_res
        )
        res_x_min, res_x_max = self.get_res_min_max(
            y_min, y_max, x_min, x_max, beta, sd_res
        )

        x_min_ = torch.min(x_min, dim=0)[0]
        y_min_ = torch.min(y_min, dim=0)[0]

        # For the Y - beta * X direction
        beta_ = beta / sd_res
        factor = torch.max(torch.abs(beta_))
        factor_ = torch.max(1 / sd_res)

        # Turn self.X and self.Y into bins
        self.x_inds = ((be.X - x_min_) // c).to(torch.int64)
        self.y_inds = ((be.Y - y_min_) // c).to(torch.int64)

        x_inds_f = ((be.X * factor) // (c / self.mult)).to(torch.int64)
        y_inds_f = ((be.Y / sd_res - res_y_min) // (c / self.mult)).to(torch.int64)
        x_inds_b = ((be.X * factor_) // (c / self.mult)).to(torch.int64)
        y_inds_b = ((beta * be.Y / sd_res + res_x_min) // (c / self.mult)).to(torch.int64)

        x_hist_f, x_counts_f = self.treat_to_hist(be.c_x, x_inds_f)
        x_hist_b, x_counts_b = self.treat_to_hist(be.c_x, x_inds_b)

        hist_res_y = torch.floor((y_inds_f.t().unsqueeze(-1) - (
            beta_ / factor).view(-1, 1, 1) * x_hist_f) // self.mult)
        hist_res_x = torch.floor(((1 / (sd_res * factor_)).view(
            -1, 1, 1) * x_hist_b - y_inds_b.t().unsqueeze(-1)) // self.mult)
        hist_res_y = hist_res_y.clamp_(
            min=0
        ).to(torch.int64)
        hist_res_x = hist_res_x.clamp_(
            min=0
        ).to(torch.int64)

        self.hist_res_y = hist_res_y.reshape(hist_res_y.shape[0], -1).t()
        self.hist_res_x = hist_res_x.reshape(hist_res_x.shape[0], -1).t()
        self.res_y_count = x_counts_f.reshape(-1, 1).to(torch.float)
        self.res_x_count = x_counts_b.reshape(-1, 1).to(torch.float)
        self.is_fitted = True

    def compute_mi_diff(self):
        super().compute_mi_diff()  # Check if fitted
        res_y_ind_uniq, res_y_ind = torch.unique(
            self.hist_res_y, return_inverse=True)
        res_x_ind_uniq, res_x_ind = torch.unique(
            self.hist_res_x, return_inverse=True)
        x_ind_uniq, x_ind = torch.unique(self.x_inds, return_inverse=True)
        y_ind_uniq, y_ind = torch.unique(self.y_inds, return_inverse=True)

        x_bins = torch.max(self.x_inds, dim=0)[0]-torch.min(
            self.x_inds, dim=0)[0] + 1
        y_bins = torch.max(self.y_inds, dim=0)[0]-torch.min(
            self.y_inds, dim=0)[0] + 1
        res_x_bins = torch.max(self.hist_res_x, dim=0)[0]-torch.min(
            self.hist_res_x, dim=0)[0] + 1
        res_y_bins = torch.max(self.hist_res_y, dim=0)[0]-torch.min(
            self.hist_res_y, dim=0)[0] + 1

        x_hist_size = torch.max(x_bins)
        y_hist_size = torch.max(y_bins)
        res_x_hist_size = torch.max(res_x_bins)
        res_y_hist_size = torch.max(res_y_bins)

        H_x = self._entropy_from_inds(
            x_hist_size, x_ind_uniq, x_ind,
            torch.ones_like(x_ind, dtype=torch.float), x_bins)

        H_y = self._entropy_from_inds(
            y_hist_size, y_ind_uniq, y_ind, self.c_y, y_bins)

        H_res_y = self._entropy_from_inds(
            res_y_hist_size, res_y_ind_uniq, res_y_ind,
            self.res_y_count.expand(-1, res_y_ind.shape[1]), res_y_bins, True)

        H_res_x = self._entropy_from_inds(
            res_x_hist_size, res_x_ind_uniq, res_x_ind,
            self.res_x_count.expand(-1, res_x_ind.shape[1]), res_x_bins)
        return H_x + H_res_y - (H_y + H_res_x)


# class FactorizedHistMI(MIEstimator):
#     """Factorized histogram-based mutual information estimator."""

#     def __init__(self, device: str = 'cpu', n_bins: int = 10):
#         super().__init__(device)
#         self.n_bins = n_bins

#     @staticmethod
#     def _get_min_max(X, c):
#         num_groups = len(c)
#         key_inds = torch.repeat_interleave(torch.arange(num_groups), c)

#         X_max = torch.full((num_groups, X.shape[1]),
#                            float('-inf'), dtype=torch.float32)
#         X_min = torch.full((num_groups, X.shape[1]),
#                            float('inf'), dtype=torch.float32)

#         expanded_keys = key_inds.unsqueeze(1).expand(-1, X.shape[1])
#         X_max = X_max.scatter_reduce(0, expanded_keys, X, reduce='amax')
#         X_min = X_min.scatter_reduce(0, expanded_keys, X, reduce='amin')
#         return X_min, X_max

#     # return two matrix, index matrix and count matrix
#     # this is because for each bin index, there's always some zeros
#     # we only care about non-zero indices
#     def _get_marginal_hist(self, marg_bins, c):
#         marg_bin_min, marg_bin_max = torch.min(marg_bins), torch.max(marg_bins)
#         tot_marg_bin = marg_bin_max - marg_bin_min + 1

#         parts = len(c)
#         unique_inds, ind_inverse = torch.unique(marg_bins, return_inverse=True)

#         jk_inds = torch.repeat_interleave(torch.arange(parts), c)
#         result = torch.zeros(marg_bins.size(1), parts,
#                              tot_marg_bin, dtype=torch.float32)

#         values = marg_bins - marg_bin_min

#         col_inds = torch.arange(
#             marg_bins.size(1)).repeat_interleave(marg_bins.size(0))

#         part_inds = jk_inds.repeat(marg_bins.size(1))

#         val_inds = values.t().reshape(-1)

#         result.index_put_((col_inds, part_inds, val_inds),
#                           torch.ones(col_inds.shape[0], dtype=torch.float32),
#                           accumulate=True)
#         mask = result != 0
#         max_unique_bins = torch.max(mask.sum(dim=2))

#         indices = torch.arange(
#             result.shape[2], device=self.device)[None, None, :]
#         masked_indices = indices * mask + (~mask) * (-1)

#         sorted_indices, _ = torch.sort(masked_indices, dim=2, descending=True)

#         hist_inds = sorted_indices[:, :, :max_unique_bins]
#         hist_mask = hist_inds == -1
#         hist_inds[hist_mask] = 0

#         batch_idx = torch.arange(
#             result.shape[0], device=self.device)[:, None, None]
#         batch_idx = batch_idx.expand(-1, result.shape[1], max_unique_bins)

#         row_idx = torch.arange(
#             result.shape[1], device=self.device)[None, :, None]
#         row_idx = row_idx.expand(result.shape[0], -1, max_unique_bins)
#         values = result[
#             batch_idx, row_idx, hist_inds]

#         hist_inds += marg_bin_min
#         hist_inds[hist_mask] = 0
#         values[hist_mask] = 0
#         return hist_inds, values

#     # inds and counts are tensors of the same shape,
#     # (i, j)-th position of counts is the count for bin inds[i, j]
#     def _entropy_from_multi_set(self, inds, counts):
#         ind_unique, ind_inverse = torch.unique(inds, return_inverse=True)
#         hist = torch.zeros((len(inds), len(ind_unique)), dtype=counts.dtype)
#         hist.scatter_add_(1, ind_inverse, counts)

#         hist /= self.n_rows
#         hist[hist == 0] = 1
#         return -torch.sum(hist * torch.log(hist), dim=1)

#     def fit(self, be, coeff):
#         """
#         Fit the estimator with the given semi-ring.
#         """

#         deg = coeff.shape[0] - 1
#         X_powers = torch.stack([be.X ** i for i in range(deg + 1)], dim=-1)

#         pred = torch.sum(X_powers * coeff.t().unsqueeze(0), dim=2)
#         self.exp_min = torch.min(be.X, dim=0).values
#         self.exp_max = torch.max(be.X, dim=0).values
#         y_min, y_max = self._get_min_max(be.Y, be.c_y)
#         pred_min, pred_max = self._get_min_max(pred, be.c_x)
#         self.res_min = torch.min(y_min - pred_max, dim=0).values
#         self.res_max = torch.max(y_max - pred_min, dim=0).values
#         exp_widths = (self.exp_max + eps - self.exp_min) / self.n_bins
#         res_widths = (self.res_max + eps - self.res_min) / self.n_bins

#         exp_bins = self._assign_bins(be.X, self.exp_min, exp_widths)
#         y_bins = self._assign_bins(be.Y, self.res_min, res_widths)
#         pred_bins = self._assign_bins(pred, 0, res_widths)
#         x_bins = exp_bins * self.n_bins - pred_bins
#         self.n_rows = torch.sum(be.c_x * be.c_y)
#         # print(f"Time used before computing marginal histogram: {t - s}")
#         self.x_inds, self.x_cts = self._get_marginal_hist(x_bins, be.c_x)

#         self.y_inds, self.y_cts = self._get_marginal_hist(y_bins, be.c_y)
#         self.is_fitted = True

#     def compute_mi_diff(self):
#         super().compute_mi_diff()  # Check if fitted
#         # marghist: shape (m, |J|, k), k: largest # of unique bin inds per jk
#         # final ind = n_bins * exp_hist_ind - pred_hist_inds + y_hist_inds
#         comb_inds = self.x_inds.unsqueeze(-1) + self.y_inds.unsqueeze(-2)
#         comb_cts = self.x_cts.unsqueeze(-1) * self.y_cts.unsqueeze(-2)

#         m = self.x_inds.shape[0]
#         comb_inds = comb_inds.reshape(m, -1)
#         comb_cts = comb_cts.reshape(m, -1)
#         comb_inds.clamp_(min=0, max=self.n_bins**2 - 1)
#         marg_ind_x = comb_inds // self.n_bins
#         marg_ind_y = comb_inds % self.n_bins
#         H_x = self._entropy_from_multi_set(marg_ind_x, comb_cts)
#         H_y = self._entropy_from_multi_set(marg_ind_y, comb_cts)
#         H_xy = self._entropy_from_multi_set(comb_inds, comb_cts)
#         return H_x + H_y - H_xy


class FactorizedNNMI(MIEstimator):
    """Factorized nearest-neighbor-based mutual information estimator."""

    def fit(self, be, coeff):
        # Implementation for factorized NN estimation
        self.is_fitted = True
        return self

    def compute_mi_diff(self) -> float:
        super().compute_mi_diff()  # Check if fitted
        return 0.0  # Placeholder
