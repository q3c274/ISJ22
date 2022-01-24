import numpy as np
import pandas as pd
from load_datasets import *
from cv_errors import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

"""
Code to run the experiment of figure 2 in the paper. We compute income as a
function of experience for several morphological variants of 'senior
administrative assistant' and 'project manager'.

We compare matching & averaging with embedding-based machine learning and
store the results in results/variants_heterogeneity.
"""

# ------------------ Load data ------------------ #

datasets = load_all(dir_path='../datasets/raw/', clean_jt=True)
names = list(datasets.keys())

# Raw FastText embeddings (punctuation removed + lowercase)
Xg, gender, dirty_jt = make_dataset_gender(datasets, keep=names, exceptions=[])
Xs, salary, groups = make_dataset_salary(datasets, keep=names, exceptions=[])

# Random shuffling of the data
p = np.random.RandomState(seed=42).permutation(len(gender))
Xg, Xs, gender, salary, dirty_jt, groups = Xg[p], Xs[p], gender[p], salary[p], dirty_jt[p], groups[p]

# ------------------ Variants heterogeneity  ------------------ #

variants = ['9020s administrative assistant salary ',
            'administrative assistant',
            'administrative asst',
            'asst administrative']

### Plug-in estimates
results = {}
for v in variants:
    m1 = (dirty_jt == v)
    seniority_cat, y_cat = Xs[m1,300], salary[m1]
    unq_sen, counts = np.unique(seniority_cat, return_counts=True)
    mean_salary = []
    for sen in unq_sen:
        m2 = (seniority_cat == sen)
        mean_salary.append(np.mean(y_cat[m2]))
    results[v] = unq_sen, counts, mean_salary

x_all, x_counts = np.unique(np.concatenate([item[0] for item in results.values()]), return_counts=True)
y_all = []
for sen in x_all:
    y_mean, n_tot = 0, 0
    for item in results.values():
        S, C, Y = item
        if sen in S:
            idx = list(S).index(sen)
            y_mean += Y[idx]*C[idx]
            n_tot += C[idx]
    y_all.append(y_mean/n_tot)

X, y = Xs[:,:301], salary
model = HistGradientBoostingRegressor(learning_rate=0.1, random_state=42)
model.fit(X, y)

### Machine learning
x2_all = np.arange(34)
y2_all = np.zeros_like(x2_all)

results2 = {}
for v in variants:
    idx = np.where(dirty_jt == v)[0]
    n_emp = len(idx)
    xc = X[idx[0]]
    unq_sen = np.arange(34)
    mean_salary = []
    for k, sen in enumerate(unq_sen):
        xc[300] = sen
        mean_salary.append(model.predict(xc[np.newaxis,:])[0])
    results2[v] = unq_sen, np.array(mean_salary)
    y2_all = y2_all + np.array(mean_salary)

y2_all /= len(variants)

variants2 = ['0361 project manager',
             '2128 project manager',
             '9109 project manager',
             'manager project',
             'mgr project',
             'project manager']

### Plug-in estimates
results3 = {}
for v in variants2:
    m1 = (dirty_jt == v)
    seniority_cat, y_cat = Xs[m1,300], salary[m1]
    unq_sen, counts = np.unique(seniority_cat, return_counts=True)
    mean_salary = []
    for sen in unq_sen:
        m2 = (seniority_cat == sen)
        mean_salary.append(np.mean(y_cat[m2]))
    results3[v] = unq_sen, counts, mean_salary

x3_all, x_counts = np.unique(np.concatenate([item[0] for item in results3.values()]), return_counts=True)
y3_all = []
for sen in x3_all:
    y_mean, n_tot = 0, 0
    for item in results3.values():
        S, C, Y = item
        if sen in S:
            idx = list(S).index(sen)
            y_mean += Y[idx]*C[idx]
            n_tot += C[idx]
    y3_all.append(y_mean/n_tot)
    

### Machine learning
x4_all = np.arange(34)
y4_all = np.zeros_like(x4_all)

results4 = {}
for v in variants2:
    idx = np.where(dirty_jt == v)[0]
    n_emp = len(idx)
    xc = X[idx[0]]
    unq_sen = np.arange(34)
    mean_salary = []
    for k, sen in enumerate(unq_sen):
        xc[300] = sen
        mean_salary.append(model.predict(xc[np.newaxis,:])[0])
    results4[v] = unq_sen, np.array(mean_salary)
    y4_all = y4_all + np.array(mean_salary)

y4_all /= len(variants2)

### Save results
import pickle
def save_obj(obj, name):
    with open(f'../results/variants_heterogeneity/{name}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return

save_obj(results, 'results')
save_obj(results2, 'results2')
save_obj(results3, 'results3')
save_obj(results4, 'results4')

save_obj(x_all, 'x_all')
save_obj(x2_all, 'x2_all')
save_obj(x3_all, 'x3_all')
save_obj(x4_all, 'x4_all')

save_obj(y_all, 'y_all')
save_obj(y2_all, 'y2_all')
save_obj(y3_all, 'y3_all')
save_obj(y4_all, 'y4_all')