import math
import torch
import time
import numpy as np
from typing import Dict
from sketch_search import JoinSketch
from mutual_info import HistMI, MIEstimator, FactorizedLinearHistMI
from test_utils import generate_df, hist_mi_gt, linear_std_mi_gt
from semi_ring import MomentSemiRing


# Bivariate Causal Discovery using Polynomial Regression
class BivariateEstimator:
    def __init__(self, degree: int, method: MIEstimator, device: str = 'cpu'):
        self.deg = degree
        self.method = method
        self.device = device

    def _get_intersect_rows(self, X, c, inds):
        c0 = torch.zeros(len(c), device=self.device)
        c0[inds] = 1
        x_inds = torch.repeat_interleave(c0, c)
        return X[x_inds == 1]

    def _get_poly_reg_moments(
        self,
        msr_x: MomentSemiRing,
        msr_y: MomentSemiRing,
        deg: int,
        std: bool
    ):
        # Check if x_moments has all required moments (0 to 2*degree)
        required_moments = set(range(2 * deg + 1))
        available_moments = set(msr_x.moments.keys())

        if not required_moments.issubset(available_moments):
            raise ValueError(f'''
            Missing required moments for degree {deg} polynomial regression''')

        c = msr_x.moments[0] * msr_y.moments[0]
        inds = torch.nonzero(c[:, 0]).squeeze()
        X_dim, Y_dim = self.X.shape[1], self.Y.shape[1]
        self.X = self._get_intersect_rows(
            self.X, msr_x.moments[0][:, 0], inds
        )
        self.Y = self._get_intersect_rows(
            self.Y, msr_y.moments[0][:, 0], inds
        )
        m_x, m_y = {0: {}, 1: {}}, {0: {}, 1: {}}
        d = max(msr_x.moments[1].shape[1], msr_y.moments[1].shape[1])
        for i in range(2 * deg + 1):
            m_y[0][i] = torch.sum(
                msr_x.moments[0][inds] * msr_y.moments[i][inds], dim=0)
            m_y[1][i] = torch.sum(
                msr_x.moments[1][inds] * msr_y.moments[i][inds], dim=0)
            m_x[0][i] = torch.sum(
                msr_x.moments[i][inds] * msr_y.moments[0][inds], dim=0)
            m_x[1][i] = torch.sum(
                msr_x.moments[i][inds] * msr_y.moments[1][inds], dim=0)
        if std:
            x_bar = m_x[0][1] / m_x[0][0]
            y_bar = m_y[0][1] / m_y[0][0]
            sd_x = torch.nan_to_num(torch.sqrt((m_x[0][2]-(
                m_x[0][1]**2) / m_x[0][0])/(m_x[0][0]-1)), nan=1)
            sd_y = torch.nan_to_num(torch.sqrt((m_y[0][2]-(
                m_y[0][1]**2) / m_y[0][0])/(m_y[0][0]-1)), nan=1)
            self.X = torch.nan_to_num((self.X - x_bar) / sd_x, nan=0)
            self.Y = torch.nan_to_num((self.Y - y_bar) / sd_y, nan=0)
            if X_dim == 1:
                self.X = self.X[:, 0:1]
            if Y_dim == 1:
                self.Y = self.Y[:, 0:1]

            std_m_x, std_m_y = {0: {}, 1: {}}, {0: {}, 1: {}}
            for i in range(2 * deg + 1):
                std_m_x[0][i] = torch.zeros(d, device=self.device)
                std_m_x[1][i] = torch.zeros(d, device=self.device)
                std_m_y[0][i] = torch.zeros(d, device=self.device)
                std_m_y[1][i] = torch.zeros(d, device=self.device)

                for j in range(i + 1):
                    coef = torch.tensor(
                        (-1)**j*math.comb(i, j), device=self.device)
                    std_m_x[0][i] += coef * m_x[0][i-j] * x_bar**j
                    std_m_x[1][i] += coef * m_x[1][i-j] * x_bar**j - \
                        coef * m_x[0][i-j] * x_bar**j * y_bar

                    std_m_y[0][i] += coef * m_y[0][i-j] * y_bar**j
                    std_m_y[1][i] += coef * m_y[1][i-j] * y_bar**j - \
                        coef * m_y[0][i-j] * y_bar**j * x_bar
                std_m_x[0][i] /= (sd_x ** i)
                std_m_x[1][i] /= (sd_y * sd_x ** i)
                std_m_y[0][i] /= (sd_y ** i)
                std_m_y[1][i] /= (sd_x * sd_y ** i)
            return std_m_x, std_m_y, msr_x.moments[0][:, 0][inds], \
                msr_y.moments[0][:, 0][inds]
        else:
            return m_x, m_y, msr_x.moments[0][:, 0][inds], \
                msr_y.moments[0][:, 0][inds]

    def _polynomial_regression(
        self,
        m_x: Dict,
        m_y: Dict,
        degree: int = 1,
        std: bool = False
    ):
        d = degree + 1
        m = max(len(m_y[0][1]), len(m_x[0][1]))
        if std and degree == 1:
            agg_x_y_std = m_x[1][1]
            coeffs = torch.nan_to_num(agg_x_y_std / (m_x[0][0] - 1), nan=0)
            coeffs = torch.stack([torch.zeros_like(coeffs), coeffs])
        elif degree == 1:
            S_xx = m_x[0][2] - m_x[0][1] ** 2 / m_x[0][0]
            S_xy = m_x[1][1] - m_x[0][1] * m_y[0][1] / m_x[0][0]
            slope = torch.nan_to_num(S_xy / S_xx, nan=0)
            intercept = torch.nan_to_num(
                m_y[0][1] / m_y[0][0] - slope * m_x[0][1] / m_x[0][0], nan=0)
            coeffs = torch.stack([intercept, slope], dim=0)
        else:
            xtx = torch.zeros(m, d, d, device=self.device)
            xty = torch.zeros(m, d, device=self.device)

            for i in range(d):
                for j in range(i, d):
                    xtx[:, i, j] = xtx[:, j, i] = m_x[0][i + j]

            for i in range(d):
                xty[:, i] = m_x[1][i]

            coeffs = (torch.linalg.inv(xtx) @ xty.unsqueeze(-1)).squeeze().t()
            coeffs[torch.isnan(coeffs)] = 0
        TSS = m_y[0][2] - m_y[0][1] ** 2 / m_y[0][0]
        RSS = m_y[0][2]
        for i in range(d):
            RSS = RSS - 2 * coeffs[i] * m_x[1][i]
            for j in range(d):
                RSS = RSS + coeffs[i] * coeffs[j] * m_x[0][i + j]
        return coeffs, torch.nan_to_num(1 - RSS / TSS, nan=0)

    def compute_mi(self, msr_x: MomentSemiRing, msr_y: MomentSemiRing,
                   X: torch.Tensor, Y: torch.Tensor, std: bool = False):
        if self.deg*2 not in msr_x.moments or self.deg*2 not in msr_y.moments:
            raise Exception(
                f'''Current estimator supports up to degree {
                    min(
                        max(msr_x.moments.keys()) / 2,
                        max(msr_y.moments.keys()) / 2
                    )}''')
        self.X, self.Y = X, Y
        m_x, m_y, self.c_x, self.c_y = self._get_poly_reg_moments(
            msr_x, msr_y, self.deg, std)

        if self.deg == 1 and std:
            # we can reuse coeff
            coeff, r2 = self._polynomial_regression(
                m_x, m_y, degree=self.deg, std=std)

            self.method.fit(self, coeff, coeff)
            mi_diff = self.method.compute_mi_diff()
            return mi_diff, r2, r2
        else:
            coeff_xy, r2_xy = self._polynomial_regression(
                m_x, m_y, degree=self.deg, std=std)
            coeff_yx, r2_yx = self._polynomial_regression(
                m_y, m_x, degree=self.deg, std=std)
            self.method.fit(self, coeff_xy, coeff_yx)
            mi_diff = self.method.compute_mi_diff()
            return mi_diff, r2_xy, r2_yx


if __name__ == "__main__":
    deg = 1
    import pickle
    from scipy import stats
    from data_profile import *
    # df_domain = np.arange(10000)
    # df1, att1 = generate_df(df_domain, 1, num_attributes=1,
    #                         random_seed=0, min_rows_per_key=1, prefix='V')
    # df2, att2 = generate_df(df_domain, 1, num_attributes=10,
    #                         random_seed=0, min_rows_per_key=1, prefix='F')

    treatment, outcome = 'V7', 'V43'
    num_samples = [100, 1000, 5000, 10000, 50000]
    for num_sample in num_samples:
        dp = DataProfile(seed=0)
        dp.generate_G(50)
        dp.generate_D_from_G(num_samples=num_sample)

    # with open(f'experiment/datasets/synthetic/data_{50}_{0}.pkl', 'rb') as file:
    #     res = pickle.load(file)
    #     dp, treatment, outcome = res[0], res[1], res[2]
        att1 = [treatment]
        att2 = [att for att in dp.D.columns if att not in {treatment, outcome, 'join_key'}][:10]
        sketch_1 = JoinSketch(join_key_domain=dp.join_key_domain)
        sketch_1.register_df(1, dp.D[att1 + ['join_key']], att1, deg=2)
        sketch_2 = JoinSketch(join_key_domain=dp.join_key_domain)
        sketch_2.register_df(2, dp.D[att2 + ['join_key']], att2, deg=2)
        msr1 = sketch_1.sketch_loader.batch_sketches[0]
        msr2 = sketch_2.sketch_loader.batch_sketches[0]
        X = torch.tensor(dp.D[att1].values, dtype=torch.float32)
        Y = torch.tensor(dp.D[att2].values, dtype=torch.float32)
        linearHistMI = FactorizedLinearHistMI()
        be = BivariateEstimator(degree=deg, method=linearHistMI)
        print("----------------Linear Fact Hist & STD Data Test Block----------------")
        mi_diff, r2, _ = be.compute_mi(msr1, msr2, X, Y, std=True)
        H_x, H_res_y, _, _, gt_hist = hist_mi_gt(dp.D[att1 + ['join_key']], dp.D[att2 + ['join_key']], deg, std=True)
        # print(f"Est MI Diff: {mi_diff}, Est r2: {r2}")
        # print(f"GT MI Diff: {H_x + H_res_y - H_y - H_res_x}, GT r2: {r2_gt}")
        # print(f"GT H_x: {H_x}, GT H_res_y: {H_res_y}")
        with open(f'hist_{len(dp.D)}.pkl', 'rb') as file:
            est_hist, est_ent = pickle.load(file)
        # print(est_ent)
        # print(H_res_y)

        # print(est_hist)
        # print(gt_hist)
        results = []
        for i in range(len(gt_hist)):
            tensor_hist = est_hist[:, i].numpy()
            array_hist = gt_hist[i]
            
            # Step 1: Pad the shorter histogram with zeros
            if len(tensor_hist) < len(array_hist):
                tensor_hist = np.pad(tensor_hist, (0, len(array_hist) - len(tensor_hist)))
            elif len(array_hist) < len(tensor_hist):
                array_hist = np.pad(array_hist, (0, len(tensor_hist) - len(array_hist)))
            
            # Step 2: Normalize the histograms
            tensor_hist = tensor_hist / np.sum(tensor_hist)
            array_hist = array_hist / np.sum(array_hist)
            
            # Add small epsilon to avoid division by zero or log(0)
            epsilon = 1e-10
            tensor_hist = np.clip(tensor_hist, epsilon, None)
            array_hist = np.clip(array_hist, epsilon, None)
            
            # Step 3: Calculate KL divergence using scipy's entropy function
            kl_div = stats.entropy(tensor_hist, array_hist)
            results.append(kl_div)
        print(np.mean(results))
        print(np.mean(np.abs(est_ent.numpy() - H_res_y)))
        print("----------------Linear Fact Hist & STD Data Test Block----------------")

    # sketch_1 = JoinSketch(join_key_domain={'join_key': df_domain})
    # sketch_1.register_df(1, df1, att1, deg=2)
    # sketch_2 = JoinSketch(join_key_domain={'join_key': df_domain})
    # sketch_2.register_df(2, df2, att2, deg=2)
    # msr1 = sketch_1.sketch_loader.batch_sketches[0]
    # msr2 = sketch_2.sketch_loader.batch_sketches[0]

    # X = torch.tensor(df1[att1].values, dtype=torch.float32)
    # Y = torch.tensor(df2[att2].values, dtype=torch.float32)
    # linearHistMI = FactorizedLinearHistMI()
    # be = BivariateEstimator(degree=deg, method=linearHistMI)
    # print("----------------Linear Fact Hist & STD Data Test Block----------------")
    # mi_diff, r2, _ = be.compute_mi(msr1, msr2, X, Y, std=True)
    # H_x, H_y, H_res_y, H_res_x, r2_gt = linear_std_mi_gt(df1, df2)
    # # print(f"Est MI Diff: {mi_diff}, Est r2: {r2}")
    # # print(f"GT MI Diff: {H_x + H_res_y - H_y - H_res_x}, GT r2: {r2_gt}")
    # print(f"GT H_x: {H_x}, GT H_y: {H_y}, GT H_res_y: {H_res_y}, GT H_res_x: {H_res_x}")
    # print("----------------Linear Fact Hist & STD Data Test Block----------------")
    # histMI = HistMI()
    # be = BivariateEstimator(degree=deg, method=histMI)
    # print("----------------Naive Hist Test Block----------------")
    # mi_diff, r2_fwd, r2_bwd = be.compute_mi(msr1, msr2, X, Y)
    # print(f"Est MI Diff: {mi_diff}, Est r2 fwd: {r2_fwd}, Est r2 bwd: {r2_bwd}")
    # H_x, H_y_res, H_x_y_res, r2_fwd = hist_mi_gt(df1, df2, deg)
    # H_y, H_x_res, H_y_x_res, r2_bwd = hist_mi_gt(df2, df1, deg)
    # print(f"GT MI Diff: {H_x + H_y_res - H_x_y_res - (H_y + H_x_res - H_y_x_res)}")
    # print(f"GT r2 fwd: {r2_fwd}, GT r2 bwd: {r2_bwd}")
    # print("----------------Naive Hist Test Block----------------")
    # print("\n")
    # print("----------------Naive Hist & STD Data Test Block----------------")
    # mi_fwd, mi_bwd, _, _ = be.compute_mi(msr1, msr2, X, Y, std=True)
    # print(f"forward MI: {mi_fwd}, backward MI: {mi_bwd}")
    # H_xy_fwd, H_sum_fwd, r2_fwd = hist_mi_gt(df1, df2, deg, std=True)
    # H_xy_bwd, H_sum_bwd, r2_bwd = hist_mi_gt(df2, df1, deg, std=True)
    # print(f"GT forward MI: {H_sum_fwd - H_xy_fwd}, GT backward MI: {H_sum_bwd - H_xy_bwd}")
    # print(f"GT r2 forward: {r2_fwd}, GT r2 backward: {r2_bwd}")
    # print("----------------Naive Hist & STD Data Test Block----------------")
    # print("\n")
    # factorizedMI = FactorizedHistMI(n_bins=10)
    # be = BivariateEstimator(degree=deg, method=factorizedMI)
    # print("----------------Factorized Hist Test Block----------------")
    # mi_fwd, mi_bwd, _, _ = be.compute_mi(msr1, msr2, X, Y)
    # print(f"forward MI: {mi_fwd}, backward MI: {mi_bwd}")
    # mi_hist_gt_fwd, r2_fwd = hist_mi_gt(df1, df2, deg, n_bins=10, factorized=True)
    # mi_hist_gt_bwd, r2_bwd = hist_mi_gt(df2, df1, deg, n_bins=10, factorized=True)
    # print(f"GT forward MI: {mi_hist_gt_fwd}, GT backward MI: {mi_hist_gt_bwd}")
    # print(f"GT r2 forward: {r2_fwd}, GT r2 backward: {r2_bwd}")
    # print("----------------Factorized Hist Test Block----------------")
    # print("\n")
    # print("----------------Factorized Hist & STD Data Test Block----------------")
    # mi_fwd, mi_bwd, _, _ = be.compute_mi(msr1, msr2, X, Y, std=True)
    # print(f"forward MI: {mi_fwd}, backward MI: {mi_bwd}")
    # mi_hist_gt_fwd, r2_fwd = hist_mi_gt(df1, df2, deg, n_bins=10, std=True, factorized=True)
    # mi_hist_gt_bwd, r2_bwd = hist_mi_gt(df2, df1, deg, n_bins=10, std=True, factorized=True)
    # print(f"GT forward MI: {mi_hist_gt_fwd}, GT backward MI: {mi_hist_gt_bwd}")
    # print(f"GT r2 forward: {r2_fwd}, GT r2 backward: {r2_bwd}")
    # print("----------------Factorized Hist & STD Data Test Block----------------")



