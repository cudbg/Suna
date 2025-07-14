import time
from sklearn.linear_model import LinearRegression
from sketch_search import JoinSketch, DataMarket, cleanup
from sklearn.feature_selection import mutual_info_regression
from semi_ring import MomentSemiRing
from mutual_info import MIEstimator, HistMI, FactorizedLinearHistMI
from bivariate_estimator import BivariateEstimator
import pandas as pd
import numpy as np
import torch
import math
import warnings
warnings.filterwarnings('ignore')
torch.set_printoptions(sci_mode=False, precision=4)
torch.manual_seed(0)


eps = 1e-5


# import pickle
# with open(
#     f'experiment/datasets/synthetic/data_{500}_{13}.pkl', 'rb'
# ) as file:
#     dp, treatment, target = pickle.load(file)


def causal_effect(data, X: str, Y: str, adjustment_set: list = []):
    """
    Calculate the linar treatment effect of X on Y on input dataset, using adjustment_set.
    """
    # Ensure X, Y, and adjustment variables are in the DataFrame
    data = data[[X, Y] + adjustment_set]
    data = data.dropna()

    if len(data) == 0:
        return -1
    if X not in data.columns or Y not in data.columns or not set(
            adjustment_set).issubset(data.columns):
        raise ValueError(
            "Input DataFrame does not contain all of the following: treatment X, outcome Y, and backdoor sets.")

    # Select explanatory variables (X and adjustment variables)
    explanatory_vars = [X] + adjustment_set
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


def get_coeff(X: torch.Tensor, y: torch.Tensor, std: bool = False):
    if std:
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True)
        y_std = y.std(dim=0, keepdim=True)
        X_ = (X - X_mean) / (X_std + eps)
        y_ = (y - y_mean) / (y_std + eps)
    else:
        X_ = torch.cat([
            torch.ones(X.shape[0], 1, device=X.device),
            X], dim=1)
        y_ = y
    # Calculate condition number threshold like sklearn
    cond = max(X_.shape) * torch.finfo(X_.dtype).eps
    coeffs = torch.linalg.lstsq(X_, y_, rcond=cond).solution
    return coeffs


class ConDiscovery:
    def __init__(
        self,
        dm,
        method: MIEstimator,
        err=0.01,
        mi_threshold=0.02,
        r2_threshold=0.02,
        device='cpu',
        verbose=False,
        bootstrap=False,
    ):
        self.err = err
        self.mi_threshold = mi_threshold
        self.device = device
        self.verbose = verbose
        self.method = method
        self.bootstrap = bootstrap
        self.r2_threshold = r2_threshold

        # Assume that this only contains treatment and outcome
        self.dm = dm
        self.cur_data_in = None

        self.treat_sketches = {}
        # Util parameters for iterative search
        self.conf_set = {}
        self.exclude_ind = {}

        self.treat_vecs = {}
        self.bts_treat_sketches = {}
        self.bts_treat_vecs = {}

    def _align_treat_vec(
        self,
        df,
        join_key,
        treat_out_instance
    ):
        if isinstance(join_key, tuple):
            join_key = list(join_key)
        align_df = df[join_key + [self.treatment]].set_index(join_key)
        align_df.index = pd.MultiIndex.from_arrays(
            [align_df.index], names=join_key)

        desired_order = treat_out_instance.join_key_index.to_frame().reset_index(
            drop=True)
        merged = pd.merge(
            desired_order,
            df.reset_index(),
            on=join_key,
            how='left')

        align_df = merged[join_key + [self.treatment]].dropna()
        align_treat_vec = torch.tensor(
            align_df[self.treatment].values, dtype=torch.float)
        # align_out_vec = torch.tensor(
        #     align_df[self.outcome].values, dtype=torch.float32)
        return align_treat_vec

    def register_buyer(self, df, join_key, jk_domain, jk_index):
        treat_instance = JoinSketch(
            join_key_domain=jk_domain,
            device=self.device,
            is_buyer=True,
            join_key_index=jk_index)

        treat_instance.register_df(
            0,
            df[list(join_key) + [self.treatment]],
            [self.treatment]
        )
        # Make sure order aligns
        treat_vec = self._align_treat_vec(df, join_key, treat_instance)
        return treat_instance, treat_vec

    def preprocess_compute_te(self, df, join_keys, treatment, outcome):
        self.conf_set[(treatment, outcome)] = []
        self.treatment = treatment
        self.outcome = outcome
        self.input_w_confs = df

        # implement the bootstrap approach to get a confidence interval
        self.bts_samples = 1

        jk_cols = set()
        if self.bootstrap:
            self.bts_samples = 10
        for join_key in join_keys:
            cur_domain = {}
            for col in join_key:
                cur_domain[col] = self.dm.join_key_domains[col]
                if col not in jk_cols:
                    jk_cols.add(col)
            if tuple(join_key) in self.dm.seller_sketches:
                if self.bootstrap:
                    self.bts_treat_vecs[tuple(join_key)] = {}
                    self.bts_treat_sketches[tuple(join_key)] = {}
                    for i in range(self.bts_samples):
                        cur_df = df.sample(n=len(df), replace=True)
                    
                        treat_out_instance, treat_vec = self.register_buyer(
                            df=cur_df,
                            join_key=join_key,
                            jk_domain=cur_domain,
                            jk_index=self.dm.seller_sketches[tuple(
                                join_key)].join_key_index
                        )
                        self.bts_treat_sketches[
                            tuple(join_key)][i] = treat_out_instance
                        self.bts_treat_vecs[tuple(
                            join_key)][i] = treat_vec.view(-1, 1).to(self.device)

                treat_out_instance, treat_vec = self.register_buyer(
                    df=df,
                    join_key=join_key,
                    jk_domain=cur_domain,
                    jk_index=self.dm.seller_sketches[tuple(
                        join_key)].join_key_index
                )
                # print(f"shape of buyer sketch: {treat_out_instance.sketch_loader.batch_sketches[0].moments[1].shape}")
                # print(f"shape of buyer sketch: {treat_out_instance.sketch_loader.batch_sketches[0].moments[0]}")
                self.treat_sketches[tuple(join_key)] = treat_out_instance
                self.treat_vecs[tuple(
                    join_key)] = treat_vec.view(-1, 1).to(self.device)
        self.cur_data_in = df[list(jk_cols) + [self.treatment]]

    def compute_treatment_effect(
            self, df, join_keys, treatment, outcome, search_iters=None):

        proc_start = time.time()
        self.preprocess_compute_te(df, join_keys, treatment, outcome)
        proc_end = time.time()
        preprocess_time = proc_end - proc_start

        conf_size = len(self.conf_set[(treatment, outcome)])
        t = time.time()
        update_df_time, update_cor_time, search_time = 0, 0, 0
        ate_iters = []
        while True:
            cur_coeff = causal_effect(
                self.input_w_confs,
                self.treatment,
                self.outcome,
                [x[2] for x in self.conf_set[(treatment, outcome)]]
            )
            ate_iters.append(cur_coeff)
            cur_dft, cur_cort, cur_st = self.search_one_iter(treatment, outcome)

            update_df_time += cur_dft
            update_cor_time += cur_cort
            search_time += cur_st
            if (len(self.conf_set[(treatment, outcome)]) == conf_size) or (
                search_iters is not None and len(
                    self.conf_set[(treatment, outcome)]) == search_iters):
                break
            conf_size += 1
        e2e_time = time.time() - t

        if self.verbose:
            print(f'''
Discovered set of confounders: {[ele[2] for ele in self.conf_set[(treatment, outcome)]]}
''')

        confs = self.conf_set[(treatment, outcome)]
        coefs = get_coeff(
            torch.tensor(
                self.input_w_confs[[x[2] for x in confs]].values,
                dtype=torch.float64,
                device=self.device
            ),
            torch.tensor(
                self.input_w_confs[treatment].values,
                dtype=torch.float64,
                device=self.device
            ).reshape(-1, 1)
        ).flatten()
        mask = torch.abs(coefs[1:]) > self.err
        self.conf_set[(treatment, outcome)] = [
            conf for conf, keep in zip(confs, mask) if keep]

        return causal_effect(
            self.input_w_confs, self.treatment,
            self.outcome, [x[2] for x in self.conf_set[(treatment, outcome)]]
        ), preprocess_time, e2e_time, search_time, update_cor_time, ate_iters

    def search_one_iter(self, treatment, outcome):
        s = time.time()
        conf_join_key, f_opt_batch_id, f_opt_ind, \
            conf_vec = self.discover_confounder()
        t = time.time()

        if conf_join_key is None:
            return 0, 0, t - s
        if f_opt_batch_id not in self.exclude_ind:
            self.exclude_ind[f_opt_batch_id] = {f_opt_ind}
        else:
            self.exclude_ind[f_opt_batch_id].add(f_opt_ind)
        seller_id, cur_feature = self.dm.seller_sketches[
            conf_join_key].get_seller_by_feature_index(f_opt_batch_id, f_opt_ind)

        orig_conf_df = self.dm.seller_datasets[seller_id][0][
            list(conf_join_key) + [cur_feature]]
        seller_name = self.dm.seller_datasets[seller_id][1]
        orig_conf_df.rename(
            columns={f'{cur_feature}': f'{seller_name}:{cur_feature}'},
            inplace=True)

        update_df_time, update_corpus_time = self._update_corpus_to_res(
            conf_join_key,
            f_opt_batch_id,
            f_opt_ind,
            conf_vec,
            f'{seller_name}:{cur_feature}',
            orig_conf_df
        )
        self.conf_set[(treatment, outcome)].append(
            (conf_join_key, seller_id, f'{seller_name}:{cur_feature}', f_opt_batch_id, f_opt_ind))
        return update_df_time, update_corpus_time, t - s

    @staticmethod
    def _get_res(X, Z):
        c = len(X)
        s_x = torch.sum(X, dim=0)
        s_z = torch.sum(Z, dim=0)
        Q_z = torch.sum(Z * Z, dim=0)
        Q_xz = torch.sum(X * Z, dim=0)

        z_bar = s_z / c
        x_bar = s_x / c

        S_xx = Q_z - s_z * z_bar
        S_xy = Q_xz - s_z * x_bar

        slope = torch.nan_to_num(S_xy / S_xx, nan=0)
        intercept = torch.nan_to_num(x_bar - slope * z_bar, nan=0)
        return X - slope * Z - intercept

    # get a varibale Z that
    # (1) is an ancestor of T
    # (2) has no confounder between T
    # (3) conditioning on Z reduces mutual information between T and O
    def discover_confounder(self):
        conf_vec = None
        conf_join_key, max_batch_id, max_ind, max_score = None, -1, -1, 0
        for join_key in self.treat_sketches.keys():
            if join_key not in self.dm.seller_sketches:
                continue
            else:
                search_sketch = self.dm.seller_sketches[join_key]
                for batch_id in range(search_sketch.sketch_loader.num_batches):
                    s_sketch = search_sketch.sketch_loader.get_sketches(
                        batch_id)
                    deg = 1
                    Z = s_sketch.moments[1]
                    if self.bootstrap:
                        mi_diffs_samples = torch.empty(
                            0, Z.shape[1], device=self.device
                        )
                        for i in range(self.bts_samples):
                            T = self.bts_treat_vecs[join_key][i]
                            t_sketch = self.bts_treat_sketches[
                                join_key][i].sketch_loader.get_sketches(0)
                            be = BivariateEstimator(
                                degree=deg, method=self.method, device=self.device
                            )
                            cur_mi_diffs, r2_fwd, r2_bwd = be.compute_mi(
                                t_sketch, s_sketch, T, Z, std=True
                            )
                            mi_diffs_samples = torch.vstack(
                                (mi_diffs_samples, cur_mi_diffs))

                        z_score = 1.96
                        means = torch.mean(mi_diffs_samples, dim=0)
                        stds = torch.std(mi_diffs_samples,
                                         dim=0, unbiased=True)
                        standard_error = stds / torch.sqrt(
                            torch.tensor(mi_diffs_samples.shape[0]))
                        # Compute confidence intervals
                        mi_diffs = means - z_score * standard_error
                    else:
                        T = self.treat_vecs[join_key]
                        t_sketch = self.treat_sketches[
                            join_key].sketch_loader.get_sketches(0)
                        be = BivariateEstimator(
                            degree=deg, method=self.method, device=self.device)
                        mi_diffs, r2_fwd, r2_bwd = be.compute_mi(
                            t_sketch, s_sketch, T, Z, std=True
                        )

                    r2_mask = (r2_fwd >= self.r2_threshold)
                    mi_diff_inds = torch.where(
                        (mi_diffs > self.mi_threshold) & r2_mask
                    )[0]

                    mi_diff_sgf = mi_diffs[mi_diff_inds]
                    r2_sgf = r2_fwd[mi_diff_inds]
                    if self.verbose:
                        print("-" * 50)
                        for i, mi_diff in enumerate(mi_diffs):
                            conf = self.dm.seller_sketches[
                                join_key].get_seller_by_feature_index(batch_id, i)
                            print(f'''
Dataset is {self.dm.seller_datasets[conf[0]][1]},
Confounder is {conf[1]},
MI diff LB is {mi_diffs[i]},
r2 fwd is {r2_fwd[i]},
r2 bwd is {r2_bwd[i]}
''')

                    if len(mi_diff_inds) == 0:
                        continue

                    Z_join = torch.repeat_interleave(
                        Z[:, mi_diff_inds],
                        self.treat_sketches[
                            join_key].sketch_loader.get_sketches(0).moments[0][:, 0],
                        dim=0
                    )
                    coeffs = torch.abs(
                        get_coeff(Z_join, self.treat_vecs[join_key])[1:].flatten()
                    )
                    sgf_coeff_inds = coeffs > self.err

                    if not sgf_coeff_inds.any():
                        continue

                    scores = torch.tanh(10 * mi_diff_sgf) * torch.tanh(10 * r2_sgf)
                    scores[~sgf_coeff_inds] = 0

                    max_score_ind = torch.argmax(scores)
                    print(f"r2 is {r2_sgf[max_score_ind]}, mi_diff is {mi_diff_sgf[max_score_ind]}")
                    cur_max_ind = mi_diff_inds[max_score_ind]
                    cur_conf_vec = Z[:, mi_diff_inds][:, max_score_ind]
                    if self.verbose:
                        print(f"This is mi_diff_sgf: {mi_diff_sgf}")
                        print(f"Coeffs is {coeffs}")

                    if scores[max_score_ind] >= max_score:
                        conf_join_key = join_key
                        max_batch_id = batch_id
                        max_ind = cur_max_ind
                        conf_vec = cur_conf_vec

                    if self.verbose and conf_join_key is not None:
                        conf = self.dm.seller_sketches[
                            conf_join_key].get_seller_by_feature_index(max_batch_id, max_ind)
                        print(f'''Dataset is {self.dm.seller_datasets[conf[0]][1]},
                        Discovered Confounder is {conf[1]}''')

            return conf_join_key, max_batch_id, max_ind, conf_vec

    def _update_df_treat(self, df, conf_col):
        X = df[[conf_col, self.treatment]].dropna().values
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        XTX = X.T @ X

        conf_sum, t_sum = XTX[0, 1], XTX[0, 2]
        t_conf_sum = XTX[1, 2]
        c, conf_2_sum = XTX[0, 0], XTX[1, 1]

        conf_mean = conf_sum / c
        t_mean = t_sum / c

        S_xx = conf_2_sum - 2 * conf_mean * conf_sum + c * conf_mean**2
        S_xt = t_conf_sum - conf_mean * t_sum - \
            conf_sum * t_mean + c * conf_mean * t_mean

        slope_xt = np.nan_to_num(S_xt / S_xx, nan=0)
        intercept_xt = np.nan_to_num(t_mean - slope_xt * conf_mean, nan=0)

        df[self.treatment] = df[self.treatment] - slope_xt * df[
            conf_col] - intercept_xt

    # Assumption: we assume the scope of datasets to be considered are the ones can be integrated with
    # the requestor dataset by a PK-FK join, and join keys are clusterd.
    # Here, we take the column of the discovered confounder, join with
    # treatment sketch

    # TODO: maintain a table with original treatment, outcome and augmented confounders, 
    # throw away confounders along the way.
    def _update_corpus_to_res(
        self, join_key, conf_batch_id, ind, conf_vec, cur_feature, orig_conf_df
    ):
        if join_key not in self.dm.seller_sketches:
            raise Exception(f"Join key cluster {join_key} not found")
        s1 = time.time()
        conf_sketch = self.dm.seller_sketches[join_key]
        # Must use join here because buyer sketch might contain more than one join key
        # join the semi-ring and do update
        conf_df = pd.DataFrame(
            {cur_feature: conf_vec.cpu().numpy()},
            index=conf_sketch.join_key_index)

        join_df = self.cur_data_in.merge(
            conf_df, on=conf_sketch.join_keys, how='left'
        )
        self.input_w_confs = self.input_w_confs.merge(
            orig_conf_df, on=conf_sketch.join_keys, how='left'
        )
        self.input_w_confs[cur_feature] = self.input_w_confs[
            cur_feature].fillna(self.input_w_confs[cur_feature].mean())

        # update join_df to residuals
        self._update_df_treat(join_df, cur_feature)
        self.cur_data_in = join_df
        s2 = time.time()

        # update clustered sketches in the data corpus
        # only consider those joinable to the input dataset
        for cur_join_key in self.treat_sketches.keys():
            if cur_join_key not in self.dm.seller_sketches:
                continue

            if self.bootstrap:
                for i in range(self.bts_samples):
                    cur_df = self.cur_data_in.sample(
                        n=len(self.cur_data_in), replace=True)
                    treat_out_instance, treat_vec = self.register_buyer(
                        df=cur_df,
                        join_key=cur_join_key,
                        jk_domain=self.treat_sketches[
                            join_key].join_key_domain,
                        jk_index=self.dm.seller_sketches[tuple(
                            cur_join_key)].join_key_index
                    )
                    self.bts_treat_sketches[tuple(
                        cur_join_key)][i] = treat_out_instance
                    self.bts_treat_vecs[tuple(
                        cur_join_key)][i] = treat_vec.view(-1, 1).to(self.device)

            treat_out_instance, treat_vec = self.register_buyer(
                df=self.cur_data_in,
                join_key=cur_join_key,
                jk_domain=self.treat_sketches[
                    join_key].join_key_domain,
                jk_index=self.dm.seller_sketches[tuple(
                    cur_join_key)].join_key_index
            )
            self.treat_sketches[tuple(cur_join_key)] = treat_out_instance
            self.treat_vecs[tuple(cur_join_key)
                            ] = treat_vec.view(-1, 1).to(self.device)

            # Update corpus sketches, including itself to 0
            if cur_join_key == join_key:
                res_Z = self._get_res(
                    conf_sketch.sketch_loader.get_sketches(conf_batch_id).moments[1],
                    conf_sketch.sketch_loader.get_sketches(conf_batch_id, [ind]).moments[1]
                )
                for batch in range(conf_sketch.sketch_loader.num_batches):
                    msr = conf_sketch.sketch_loader.batch_sketches[batch].moments
                    res_moments = {}
                    for k in msr.keys():
                        res_moments[k] = (res_Z ** k).to(
                            dtype=msr[k].dtype
                        )
                    self.dm.seller_sketches[cur_join_key].sketch_loader.update_sketch(
                        MomentSemiRing(res_moments, self.device), batch
                    )
            else:
                update_sketch = JoinSketch(
                    join_key_domain=conf_sketch.join_key_domain, device=self.device)
                update_sketch.register_df(
                    0,
                    join_df[list(join_key) + ['Z']],
                    ['Z'],
                    agg='mean'
                )
                res_Z = self._get_res(
                    update_sketch.sketch_loader.get_sketches(0).moments[1],
                    self.dm.seller_sketches[join_key].
                    sketch_loader.get_sketches(conf_batch_id, [ind]).moments[1]
                )
                for batch in range(conf_sketch.sketch_loader.num_batches):
                    msr = conf_sketch.sketch_loader.batch_sketches[batch].moments
                    res_moments = {}
                    for k in msr.keys():
                        res_moments[k] = (res_Z ** k).to(
                            dtype=msr[k].dtype
                        )
                    self.dm.seller_sketches[join_key].sketch_loader.update_sketch(
                        MomentSemiRing(res_moments, self.device), batch
                    )
        s3 = time.time()
        return s2 - s1, s3 - s2


def warmup_gpu():
    device = 'cuda'
    batch_size = 1000
    feature_dim = 10
    num_groups = 5
    with torch.no_grad():
        X = torch.randn(batch_size, feature_dim, device=device)
        c_x = torch.randint(1, 10, (num_groups,), device=device)
        key_inds = torch.repeat_interleave(torch.arange(num_groups, device=device), c_x)
        X_max = torch.full((num_groups, feature_dim), float('-inf'), device=device)
        X_min = torch.full((num_groups, feature_dim), float('inf'), device=device)
        expanded_keys = key_inds.unsqueeze(1).expand(-1, feature_dim)
        _ = X_max.scatter_reduce(0, expanded_keys, X, reduce='amax')
        _ = X_min.scatter_reduce(0, expanded_keys, X, reduce='amin')

        treat_inds = torch.randint(0, 100, (batch_size,), device=device)
        _ = treat_inds.unique(return_counts=True)

        x_inds = torch.randint(0, 50, (batch_size, feature_dim), device=device)
        _ = torch.unique(x_inds, return_inverse=True)
        hist_size = 50
        ind_inverse = torch.randint(0, hist_size, (batch_size, feature_dim), device=device)
        counts = torch.ones_like(ind_inverse, dtype=torch.float, device=device)
        hist = torch.zeros((hist_size, ind_inverse.shape[1]), device=device)
        _ = hist.scatter_add_(0, ind_inverse, counts)
        torch.cuda.synchronize()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed(0)
        warmup_gpu()
    parents = list(dp.G.predecessors(treatment))
    gt = causal_effect(dp.D, treatment, target, parents)

    dp.generate_partitions_from_D(treatment, target, ['join_key'])
    dm = DataMarket(device=device)
    dm.add_seller(dp.data_corpus, "synthetic", [[dp.join_key]], dp.join_key_domain,
                  [col for col in dp.data_corpus if col != dp.join_key])

    # method = HistMI(device=device)
    method = FactorizedLinearHistMI(mult=2, device=device)
    cd = ConDiscovery(dm, method, verbose=False, bootstrap=True, device=device)
    est_suna, preprocess_time, end_to_end_time, search_time, update_cor_time, _ = cd.compute_treatment_effect(
        dp.data_in, [['join_key']], treatment, target)
    print(f'''
treatment: {treatment}, outcome: {target},
Treatment parents: {parents},
adjustment set is: {[x[2] for x in cd.conf_set[(treatment, target)]]},
est_suna: {est_suna},
gt is: {gt},
preprocess_time: {preprocess_time},
end_to_end_time: {end_to_end_time},
search_time: {search_time},
update_cor_time: {update_cor_time}
''')
