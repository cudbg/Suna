import warnings
import random
import os
from discovery import ConDiscovery
from data_profile import DataProfile
from sketch_search import DataMarket
import pickle
import multiprocessing
from sklearn.linear_model import LinearRegression
from mutual_info import MIEstimator, HistMI, FactorizedLinearHistMI
warnings.filterwarnings('ignore')


def random_pair(n):
    i = random.randint(min(10, n//10), min(50, n//3))
    j = random.randint(i+1, n-1)
    return i, j


def causal_effect(data, X: str, Y: str, adjustment_set):
    """
    Calculate the linar treatment effect of X on Y on input dataset, using adjustment_set.
    """
    # Ensure X, Y, and adjustment variables are in the DataFrame
    data = data[[X, Y] + list(adjustment_set)]
    data = data.dropna()
    # print(len(data))

    if len(data) == 0:
        return -1
    if X not in data.columns or Y not in data.columns or not set(
        adjustment_set
    ).issubset(data.columns):
        raise ValueError(
            f"Data does not contain all of: {X}, {Y}, and {adjustment_set}.")

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


def accuracy_exp_iteration(
    dp, treatment, target, mi_thresholds, method, bootstrap, join_key, device
):
    parents = set(list(dp.G.predecessors(treatment)))
    gt = causal_effect(dp.D, treatment, target, parents)
    results = {}
    for mi_threshold in mi_thresholds:
        dm = DataMarket(device)
        dm.add_seller(dp.data_corpus, "synthetic", [[dp.join_key]], dp.join_key_domain, 
                      [col for col in dp.data_corpus if col != dp.join_key])
        cd = ConDiscovery(
            dm, method=method, mi_threshold=mi_threshold,
            bootstrap=bootstrap, verbose=False, device=device
        )
        est_suna, prep_time, e2e_time, search_time, update_time, _ = cd.compute_treatment_effect(
            dp.data_in, [['join_key']], treatment, target)
        print(f"Treatment parents: {parents}")
        print(f"Treatment: {treatment}, Outcome: {target}, Error: {(gt - est_suna) ** 2}, Estimation: {est_suna}, Ground Truth: {gt}, e2e_time: {e2e_time}")
        conf_set = cd.conf_set[(treatment, target)]
        results[mi_threshold] = {
            'est': est_suna,
            'gt': gt,
            'size': len(conf_set),
            'preprocess': prep_time,
            'end_to_end': e2e_time,
            'search': search_time,
            'update_res': update_time,
        }
    return results, gt


def accuracy_exp(
    runs, num_nodes=[100], mi_thresholds=[0.02], hist=False,
    fhm=False, bootstrap=False, gpu=False
):
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
    join_key = ['join_key']
    res = {}
    iteration_pairs = [
        (num_node, run_num) for num_node in num_nodes for run_num in range(runs)
    ]

    for num_node, run_num in iteration_pairs:
        with open(
            f'experiment/datasets/synthetic/data_{num_node}_{run_num}.pkl',
            'rb'
        ) as file:
            dp, treatment, target = pickle.load(file)
            if hist:
                method = HistMI(device=device)
            elif fhm:
                method = FactorizedLinearHistMI(device=device, mult=2)
            else:
                raise ValueError("Hist and FHM both False.")

            result_suna, gt = accuracy_exp_iteration(
                dp, treatment, target, mi_thresholds,
                method, bootstrap, join_key, device
            )

            if num_node not in res:
                res[num_node] = {}

            for mi_threshold, cur_res in result_suna.items():
                if mi_threshold not in res[num_node]:
                    res[num_node][mi_threshold] = {
                        'est': [],
                        'size': 0,
                        'preprocess': 0,
                        'end_to_end': 0,
                        'search': 0,
                        'update_res': 0,
                    }

                res[num_node][mi_threshold]['est'].append(
                    (cur_res.get('est', 0), cur_res.get('gt', 0)))
                res[num_node][mi_threshold]['size'] += cur_res.get(
                    'size', 0)
                res[num_node][mi_threshold]['preprocess'] += cur_res.get(
                    'preprocess', 0)
                res[num_node][mi_threshold]['end_to_end'] += cur_res.get(
                    'end_to_end', 0)
                res[num_node][mi_threshold]['search'] += cur_res.get(
                    'search', 0)
                res[num_node][mi_threshold]['update_res'] += cur_res.get(
                    'update_res', 0)
    return res


def gen_synthetic_data(runs, num_nodes, num_samples):
    os.makedirs('experiment/datasets/synthetic', exist_ok=True)
    for i in range(runs):
        for j in num_nodes:
            dp = DataProfile(seed=i)
            dp.generate_G(j)
            dp.generate_D_from_G(num_samples=num_samples)

            treatment_ind, target_ind = random_pair(j)
            treatment, target = \
                dp.ordered_nodes[treatment_ind], dp.ordered_nodes[target_ind]

            dp.generate_partitions_from_D(treatment, target, ['join_key'])
            res = (dp, treatment, target)

            with open(f'experiment/datasets/synthetic/data_{j}_{i}.pkl', 'wb') as file:
                pickle.dump(res, file)


#     from test_utils import hist_mi_gt
#     print(f'''GT integration:
# {dp.D.head()}
# ''')
#     print(f"treatment: {treatment}, outcome: {target}")
#     col1 = [col for col in dp.D.columns if col not in {treatment, target, 'join_key'}]
#     col2 = [treatment]
#     for j in range(10):
#         sample_df = dp.D.sample(n=len(dp.D), replace=True)
#         H_x_bwd, H_y_bwd, H_xy_bwd, r2 = hist_mi_gt(col1, col2, sample_df, 1, std=True)
#         H_x_fwd, H_y_fwd, H_xy_fwd, r2 = hist_mi_gt(col2, col1, sample_df, 1, std=True)
#         for i in range(len(col1) - 1):
#             print(f'''
# Bootstrap {j}:
# Attribute is {col1[i]},
# MI diff: {H_x_fwd[i] + H_y_fwd[i] - (H_x_bwd[i] + H_y_bwd[i])},
# MI bwd: {H_x_bwd[i] + H_y_bwd[i] - H_xy_bwd[i]}''')


if __name__ == "__main__":
    runs = 100
    num_nodes = [10, 50, 100, 500]
    mi_thresholds = [0.01, 0.02, 0.05, 0.1]
    # num_nodes = [500]
    # mi_thresholds = [0.02]
    # gen_synthetic_data(runs, num_nodes, 10000)
    res = accuracy_exp(
        runs, num_nodes, mi_thresholds, hist=False,
        fhm=True, bootstrap=False, gpu=False
    )
    with open('experiment/suna_synthetic_result_cpu_fhm.pkl', 'wb') as file:
        pickle.dump(res, file)
    res = accuracy_exp(
        runs, num_nodes, mi_thresholds, hist=True,
        fhm=False, bootstrap=False, gpu=False
    )
    with open('experiment/suna_synthetic_result_cpu_hist.pkl', 'wb') as file:
        pickle.dump(res, file)
        
    res = accuracy_exp(
        runs, num_nodes, mi_thresholds, hist=False,
        fhm=True, bootstrap=True, gpu=False
    )
    with open('experiment/suna_synthetic_result_cpu_bts_fhm.pkl', 'wb') as file:
        pickle.dump(res, file)
    res = accuracy_exp(
        runs, num_nodes, mi_thresholds, hist=True,
        fhm=False, bootstrap=True, gpu=False
    )
    with open('experiment/suna_synthetic_result_cpu_bts_hist.pkl', 'wb') as file:
        pickle.dump(res, file)


    # res, _ = accuracy_exp(runs=1, factorized_hist=True)
    # print(res)
    # res, _ = accuracy_exp(runs=1, approx=True)
    # print(res)






