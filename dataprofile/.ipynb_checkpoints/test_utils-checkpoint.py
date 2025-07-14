import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from scipy.special import digamma
from sklearn.neighbors import KDTree


eps = 1e-5


def generate_df(join_key_domain, max_rows_per_key,
                num_attributes, min_rows_per_key=0, random_seed=None, prefix='V'):
    if random_seed is not None:
        np.random.seed(random_seed)

    rows_per_key = np.random.randint(
        min_rows_per_key, max_rows_per_key + 1, size=len(join_key_domain))

    all_keys = []
    all_attributes = []

    for key, num_rows in zip(join_key_domain, rows_per_key):
        if num_rows > 0:
            all_keys.extend([key] * num_rows)
            attributes = np.random.randn(num_rows, num_attributes)
            all_attributes.append(attributes)

    if all_keys:
        attributes_array = np.vstack(all_attributes)
        df = pd.DataFrame(attributes_array, columns=[
                          f'{prefix}{i+1}' for i in range(num_attributes)])
        df.insert(0, 'join_key', all_keys)
        return df, [f'{prefix}{i+1}' for i in range(num_attributes)]
    else:
        columns = ['join_key'] + \
            [f'{prefix}{i+1}' for i in range(num_attributes)]
        return pd.DataFrame(columns=columns), [
            f'{prefix}{i+1}' for i in range(num_attributes)]


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


def polynomial_regression_gt(df1, df2, k, std=False):
    merged_df = pd.merge(df1, df2, on='join_key', how='inner')
    X_cols = [col for col in df1.columns if col != 'join_key']
    Y_cols = [col for col in df2.columns if col != 'join_key']
    max_length = max(len(X_cols), len(Y_cols))
    if len(X_cols) == 1:
        X_cols = max_length*X_cols
    elif len(Y_cols) == 1:
        Y_cols = max_length*Y_cols

    sklearn_coefficients = np.zeros((k + 1, len(X_cols)))

    if std:
        merged_df_std = merged_df.copy()
        for col in X_cols + Y_cols:
            merged_df_std[col] = (
                merged_df[col] - merged_df[col].mean()) / merged_df[col].std()
        merged_df = merged_df_std

    for i, (x_col, y_col) in enumerate(zip(X_cols, Y_cols)):
        x_data = merged_df[x_col].values
        y_data = merged_df[y_col].values

        x_data_reshaped = x_data.reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=k, include_bias=True)
        x_poly = poly_features.fit_transform(x_data_reshaped)

        model = LinearRegression(fit_intercept=False)
        model.fit(x_poly, y_data)

        sklearn_coefficients[:, i] = model.coef_

    print(f'''Sklearn coefficients:
{sklearn_coefficients}
''')


def mi_from_ind(inds, bins):
    histogram = np.zeros(bins)
    unique_indices, counts = np.unique(inds, return_counts=True)
    histogram[unique_indices.astype(int)] = counts
    total_count = counts.sum()
    # print(f"GT histogram: {histogram}")
    probabilities = eps + histogram / total_count
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy, histogram


# def hist_mi_gt(df1, df2, k, std=False):
#     merged_df = pd.merge(df1, df2, on='join_key', how='inner')
def hist_mi_gt(df1, df2, k, std=False):
    merged_df = pd.merge(df1, df2, on='join_key', how='inner')
    c = 3.49 * (len(merged_df)**(-1/3))
    X_cols = [col for col in df1.columns if col != 'join_key']
    Y_cols = [col for col in df2.columns if col != 'join_key']
    max_length = max(len(X_cols), len(Y_cols))
    if len(X_cols) == 1:
        X_cols = max_length*X_cols
    elif len(Y_cols) == 1:
        Y_cols = max_length*Y_cols
    H_x = np.zeros(len(X_cols))
    H_y = np.zeros(len(X_cols))
    H_xy = np.zeros(len(X_cols))
    r2 = np.zeros(len(X_cols))

    hist_gt = []

    for i, (x_col, y_col) in enumerate(zip(X_cols, Y_cols)):
        x_data = merged_df[x_col].values
        y_data = merged_df[y_col].values
        if std:
            x_data = (x_data - np.mean(x_data)) / np.std(x_data, ddof=1)
            y_data = (y_data - np.mean(y_data)) / np.std(y_data, ddof=1)

        x_data_reshaped = x_data.reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=k, include_bias=True)
        x_poly = poly_features.fit_transform(x_data_reshaped)

        model = LinearRegression(fit_intercept=False)
        model.fit(x_poly, y_data)

        y_pred = model.predict(x_poly)
        r2[i] = r2_score(y_data, y_pred)

        res = y_data - y_pred

        if std:
            res = res / np.std(res, ddof=1)

        res_width = c * np.std(res, ddof=1)
        x_width = c * np.std(x_data, ddof=1)
        res_inds = ((res - np.min(res)) // res_width).astype(int)
        x_inds = ((x_data - np.min(x_data)) // x_width).astype(int)
        num_bins_res = int(np.max(res_inds) - np.min(res_inds) + 1)
        num_bins_x = int(np.max(x_inds) - np.min(x_inds) + 1)
        comb_inds = x_inds * num_bins_res + res_inds

        H_x[i], _ = mi_from_ind(x_inds, num_bins_x)
        H_y[i], res_y_hist = mi_from_ind(res_inds, num_bins_res)
        H_xy[i], _ = mi_from_ind(comb_inds, num_bins_x * num_bins_res)
        hist_gt.append(res_y_hist)
    # hist_gt = np.array(hist_gt).T
    return H_x, H_y, H_xy, r2, hist_gt


def linear_std_mi_gt(df1, df2):
    merged_df = pd.merge(df1, df2, on='join_key', how='inner')
    c = 3.49 * (len(merged_df)**(-1/3))
    X_cols = [col for col in df1.columns if col != 'join_key']
    Y_cols = [col for col in df2.columns if col != 'join_key']
    assert len(X_cols) == 1
    max_length = max(len(X_cols), len(Y_cols))

    X_cols = max_length*X_cols
    x_vecs = []
    y_vecs = []
    coeffs = []
    sds = []
    res_mins_x, res_mins_y = [], []
    r2 = np.zeros(len(X_cols))
    H_x = np.zeros(len(X_cols))
    H_y = np.zeros(len(X_cols))
    H_res_x = np.zeros(len(X_cols))
    H_res_y = np.zeros(len(X_cols))

    for i, (x_col, y_col) in enumerate(zip(X_cols, Y_cols)):
        x_data = merged_df[x_col].values
        y_data = merged_df[y_col].values
        x_data = (x_data - np.mean(x_data)) / np.std(x_data, ddof=1)
        y_data = (y_data - np.mean(y_data)) / np.std(y_data, ddof=1)
        x_vecs.append(x_data)
        y_vecs.append(y_data)
        x_data_reshaped = x_data.reshape(-1, 1)
        y_data_reshaped = y_data.reshape(-1, 1)

        model = LinearRegression(fit_intercept=False)
        model.fit(x_data_reshaped, y_data)
        coeff = model.coef_
        coeffs.append(coeff)

        y_pred = model.predict(x_data_reshaped)

        model = LinearRegression(fit_intercept=False)
        model.fit(y_data_reshaped, x_data)
        assert abs(coeff - model.coef_) < eps

        x_pred = model.predict(y_data_reshaped)
        r2[i] = r2_score(x_data, x_pred)

        res_y = y_data - y_pred
        sd_res = np.std(res_y, ddof=1)
        sds.append(sd_res)
        res_mins_y.append(np.min(res_y) / sd_res)
        res_x = x_data - x_pred
        assert abs(sd_res - np.std(res_x, ddof=1)) < eps
        res_mins_x.append(np.min(res_x) / sd_res)

    f1 = max([abs(coef/s) for coef, s in zip(coeffs, sds)])
    f2 = max([abs(1/s) for s in sds])
    for i, (x, y) in enumerate(zip(x_vecs, y_vecs)):
        # (Y - beta * X) / sd = Y / sd - (beta / sd) / f1 * f1 * X
        y_inds = (y / sds[i] - res_mins_y[i]) // c
        beta_x = coeffs[i] / (sds[i] * f1) * (f1 * x // c)
        # (X - beta * Y) / sd =  1 / (f2 * sd) * f2 * X - beta * Y / sd
        x_inds = 1 / (sds[i] * f2) * ((f2 * x) // c)
        beta_y = (coeffs[i] * y + res_mins_x[i]) // c
        res_y_inds = y_inds - beta_x
        res_x_inds = x_inds - beta_y
        inds_res_y = np.clip(np.round(res_y_inds).astype(int), 0, None)
        inds_res_x = np.clip(np.round(res_x_inds).astype(int), 0, None)
        inds_x = ((x - np.min(x)) // c).astype(int)
        inds_y = ((y - np.min(y)) // c).astype(int)

        # print(f"Est res_y bin inds: {inds_res_y}")
        # inds_res_y_gt = ((y - coeffs[i] * x) / sds[i] - res_mins_y[i]) // c
        # print(f"GT res_y bin inds: {inds_res_y_gt}")
        # print(f"Est Error: {inds_res_y_gt - inds_res_y}")

        num_bins_res_x = np.max(inds_res_x) - np.min(inds_res_x) + 1
        num_bins_res_y = np.max(inds_res_y) - np.min(inds_res_y) + 1
        num_bins_x = np.max(inds_x) - np.min(inds_x) + 1
        num_bins_y = np.max(inds_y) - np.min(inds_y) + 1

        H_res_y[i], _ = mi_from_ind(inds_res_y, num_bins_res_y)
        H_res_x[i], _ = mi_from_ind(inds_res_x, num_bins_res_x)
        H_x[i], _ = mi_from_ind(inds_x, num_bins_x)
        H_y[i], _ = mi_from_ind(inds_y, num_bins_y)
    return H_x, H_y, H_res_y, H_res_x, r2




















