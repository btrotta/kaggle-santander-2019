import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn import model_selection, metrics, linear_model
import datetime as dt
from scipy import special

# Since the following forum topic and script show that relationships between columns don't matter, treat all columns
# individually and then build a meta model to combine the predictions.
# https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83882
# https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works?scriptVersionId=11467087

# Values which repeat in a column seem to show stronger signal, so include a feature based on that. However, as
# observed in the following kernel, some of the test data seems to be simulated:
# https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split?scriptVersionId=11948999
# Therefore exclude these from the counts.

test_mode = False

# read input
train = pd.read_csv(os.path.join('input', 'train.csv'))
test = pd.read_csv(os.path.join('input', 'test.csv'))
if test_mode:
    all_data = train
else:
    all_data = pd.concat([train, test], ignore_index=True, sort=True).reset_index()
    train_bool = all_data['target'].notnull()

# identify the fake test data as in above kernel
train_cols = [c for c in train.columns if c not in ['ID_code', 'target', 'predicted', 'size', 'index']]
unique_vals = {}
for c in train_cols:
    sizes = test.groupby(c)['ID_code'].size()
    unique_vals[c] = sizes.loc[sizes == 1].index
unique_bool = test.copy()
for c in train_cols:
    unique_bool[c] = test[c].isin(unique_vals[c])
num_unique = unique_bool[train_cols].sum(axis=1)
test_real = test.loc[num_unique > 0]

# add the counts
count_data = pd.concat([train, test_real], axis=0, sort=True)
for i in range(200):
    c = 'var_' + str(i)
    all_size = count_data.groupby(c)['ID_code'].size().to_frame(c + '_size')
    count_data['rank'] = count_data[c].rank(method='first')
    count_data['bin'] = (count_data['rank'] - 1) // 300
    count_data['size'] = count_data.groupby(c)['ID_code'].transform('size')
    count_data['avg_size_in_bin'] = count_data.groupby('bin')['size'].transform('mean')
    count_data[c + '_size_scaled'] = count_data['size'] / count_data['avg_size_in_bin']
    all_size[c + '_size_scaled'] = count_data.groupby(c)[c + '_size_scaled'].first()
    all_data = pd.merge(all_data, all_size, 'left', left_on=c, right_index=True)


def rolling_log_reg(var_arr, target_arr, window, step_size):
    est = linear_model.LogisticRegression(random_state=0)
    ans = np.zeros(len(range(0, len(var_arr), step_size)))
    ans[:] = np.nan
    for i, center in enumerate(range(0, len(var_arr), step_size)):
        min_ind = max(0, center - window // 2)
        max_ind = min(len(var_arr) - 1, center + window // 2)
        x_curr = var_arr[min_ind:max_ind][:, np.newaxis]
        y_curr = target_arr[min_ind:max_ind]
        if len(np.unique(y_curr)) > 1:
            est.fit(x_curr, y_curr)
            ans[i] = est.predict_proba(np.array([var_arr[center]])[:, np.newaxis])[:, 1]
        else:
            ans[i] = np.unique(y_curr)[0]
    return np.interp(var_arr, var_arr[np.arange(0, len(var_arr), step_size)], ans)


def rolling_prediction(input_df, train_bool, c):
    col_copy = input_df[[c, c + '_size', 'target']].copy()
    size1 = col_copy.loc[(col_copy[c + '_size'] == 1) & train_bool].copy().sort_values(c)
    size2 = col_copy.loc[(col_copy[c + '_size'] > 1) & train_bool].copy().sort_values(c)
    size1[c + '_rolling'] = rolling_log_reg(size1[c].values, size1['target'].values, 20000, 500)
    size2[c + '_rolling'] = rolling_log_reg(size2[c].values, size2['target'].values, 20000, 500)
    input_df[c + '_rolling'] = np.nan
    input_df.loc[input_df[c + '_size'] <= 1, c + '_rolling'] \
        = np.interp(input_df.loc[input_df[c + '_size'] <= 1, c], size1[c].values, size1[c + '_rolling'].values)
    input_df.loc[input_df[c + '_size'] > 1, c + '_rolling'] \
        = np.interp(input_df.loc[input_df[c + '_size'] > 1, c].values, size2[c].values, size2[c + '_rolling'].values)
    return input_df


# define lgb model
params = {'objective': 'binary', 'learning_rate': 0.01, 'lambda_l1': 0, 'lambda_l2': 0, 'num_leaves': 4,
          'bagging_fraction': 0.8, 'bagging_freq': 1, 'max_depth': 2, 'seed': 0, 'verbosity': -1}
num_rounds = 2000

# controls the contribution of rolling average model to final prediction
rolling_factor = 0.3

# log file
log_file = 'Log_{}.txt'.format(dt.datetime.now().strftime('%y%m%d_%H%M'))

# cross validate and find the optimal number of iterations for each column
best_iter = {c: [] for c in [c + '_size' for c in train_cols] + [c + '_size_scaled' for c in train_cols]}
auc_train = []
auc_test = []
cv = model_selection.KFold(5, shuffle=True, random_state=0)
train = all_data.loc[all_data['target'].notnull()].copy()
for train_ind, test_ind in list(cv.split(train)):
    lgb_results = train[train_cols].copy()
    rolling_results = train[train_cols].copy()
    train_bool = train.index.isin(train_ind)
    test_bool = train.index.isin(test_ind)
    # individual column models
    for c in train_cols:
        # lgb: average 2 models with different features
        lgb_results[c] = 0
        for size_col in [c + '_size', c + '_size_scaled']:
            lgb_train = lgb.Dataset(train.loc[train_bool, [c, size_col]],
                                    label=train.loc[train_bool, 'target'].values)
            lgb_valid = lgb.Dataset(train.loc[test_bool, [c, size_col]],
                                    label=train.loc[test_bool, 'target'].values)
            est = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train, lgb_valid],
                            valid_names=['train', 'valid'], num_boost_round=num_rounds, early_stopping_rounds=100)
            lgb_results[c] += 0.5 * special.logit(est.predict(train[[c, size_col]],
                                         num_iteration=max(est.best_iteration + 100, num_rounds)))

            best_iter[size_col].append(est.best_iteration)
        # rolling logistic model
        train = rolling_prediction(train, train_bool, c)
        rolling_results[c] = special.logit(train[c + '_rolling'])

    # combine predictions and train meta model
    comb_pred = (1 - rolling_factor) * lgb_results + rolling_factor * rolling_results
    train['predicted'] = special.expit(comb_pred.mean(axis=1))

    # score
    auc_train.append(metrics.roc_auc_score(train.loc[train_bool, 'target'].values,
                                          train.loc[train_bool, 'predicted'].values))
    auc_test.append(metrics.roc_auc_score(train.loc[test_bool, 'target'].values,
                                          train.loc[test_bool, 'predicted'].values))
    print('train auc: ', auc_train)
    print('test auc: ', auc_test)
    with open(log_file, 'a') as f:
        f.write('\ntrain auc: {}'.format(auc_train))
        f.write('\ntest auc: {}'.format(auc_test))
print('mean train auc: ', np.mean(auc_train))
print('mean test auc: ', np.mean(auc_test))
with open(log_file, 'a') as f:
    f.write('\nmean train auc: {}'.format(np.mean(auc_train)))
    f.write('\nmean test auc: {}'.format(np.mean(auc_test)))

# train on all data and write output
if not(test_mode):
    train_bool = all_data['target'].notnull()
    lgb_results = all_data[train_cols].copy()
    rolling_results = all_data[train_cols].copy()
    for c in train_cols:
        # lgb: average 2 models with different features
        lgb_results[c] = 0
        for size_col in [c + '_size', c + '_size_scaled']:
            lgb_train = lgb.Dataset(all_data.loc[train_bool, [c, size_col]],
                                    label=all_data.loc[train_bool, 'target'].values)
            est = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_train],
                            valid_names=['train'],
                            num_boost_round=min(num_rounds, max(1, int(np.mean(best_iter[size_col])))))
            lgb_results[c] += 0.5 * special.logit(est.predict(all_data[[c, size_col]]))
        # rolling mean
        all_data = rolling_prediction(all_data, train_bool, c)
        rolling_results[c] = special.logit(all_data[c + '_rolling'])

    # combine predictions and train meta model
    comb_pred = (1 - rolling_factor) * lgb_results + rolling_factor * rolling_results
    all_data['predicted'] = special.expit(comb_pred.mean(axis=1))
    all_data['target'] = all_data['predicted']
    all_data.loc[~train_bool, ['ID_code', 'target']].to_csv(
        'Submission_{}.csv'.format(dt.datetime.now().strftime('%y%m%d_%H%M')), index=False)
