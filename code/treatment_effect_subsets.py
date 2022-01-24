import numpy as np
import pandas as pd
from load_datasets import *
from models import *
from time import time
import re
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, KFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier,\
HistGradientBoostingRegressor, GradientBoostingRegressor
from joblib import Parallel, delayed
from ast import literal_eval

"""
Compute employees individual treatment effects on subsets of the 14 databases.
These numbers can then be averaged on all employees to obtain the average
treatment effect, or on groups of employee to obtain heterogeneous treatment
effects.

Individual treatment effects (ITE) are computed through an augmented-inverse-
propensity-weigthing estimator which models the response to the treatment
(salary) and the probability of being treated given the covariates
(propensity-score).
We also use a cross-fitting procedure to avoid overfitting. The data is
split into several folds, and we estimate the ITEs on a fold with models
trained on the remaining folds.
Finally, we compute ITEs on databases subsets: 1 DB, 3DBs, 7DBs and 14 DBs.

"""

# ------------------ Load data ------------------ #

datasets = load_all(dir_path='../datasets/raw/', clean_jt=True)
names = list(datasets.keys())

# Raw FastText embeddings (punctuation removed + lowercase)
Xg, gender, dirty_jt = make_dataset_gender(datasets, keep=names, exceptions=[])
Xs, salary, groups = make_dataset_salary(datasets, keep=names, exceptions=[])

# Cleaned FastText embeddings (manual cleaning on OpenRefine)
Xg_cft, Xs_cft, clean_jt = load_cleaned_embeddings(
    Xg, Xs, '../datasets/merged_cleaned_simple_matched_data.csv')

# Random shuffling of the data
p = np.random.permutation(len(gender))
Xg, Xs, yg, ys, groups = Xg[p], Xs[p], gender[p], salary[p], groups[p]
Xg_cft, Xs_cft, dirty_jt, clean_jt = Xg_cft[p], Xs_cft[p], dirty_jt[p], clean_jt[p]

df = pd.read_csv('../datasets/merged_raw_data.csv')
ethnicity = df['Ethnicity']

D_matching = {'White': '1',
             'Hispanic or Latino': '3',
             'Black or African American': '2',
             'Asian': '5',
             'Choose Not To Disclose': '-1',
             'Two or more races': '6',
             'American Indian/Alaska Native': '4',
             'Native Hawaiian/Pacific Isl': '5',
             '(Invalid) Other': '-1',
             '(Invalid) Asian/Pacific Isl': '5',
             'Hispanic/Latino': '3',
             'Two or More Races': '6',
             'American Indian or Alaskan Native': '4',
             'Native Hawaiian or Pacific Islander': '5',
             'OTHR': '-1',
             'WHT': '1',
             'BLK': '2',
             'ASIN': '5',
             'MEXA': '3',
             'OASI': '-1',
             'SPAN': '3',
             'KORN': '5',
             'PUER': '3',
             'TWO': '6',
             'AMIN': '4',
             'VIET': '5',
             'CUBA': '3',
             'GUAM': '3',
             'CHIN': '5',
             'FILI': '5',
             'HAWA': '5',
             'JAPN': '5',
             'PACF': '5',
             'SAMO': '5',
             'WHITE': '1',
             'ASIAN': '5',
             'BLACK': '2',
             'HISPA': '3',
             'AMIND': '4',
             '2ORMORE': '6',
             'PACIF': '5',
             'NSPEC': '-1',
             'Native Hawaiian or Other Pacific Islander': '5',
             'WHITE (NON HISPANIC OR LATINO)': '1',
              'HISPANIC OR LATINO': '3',
             'BLACK OR AFRICAN AMERICAN (NON HISPANIC OR LATINO)': '2',
             'ASIAN (NON HISPANIC OR LATINO)': '5',
             'TWO OR MORE RACES (NON HISPANIC OR LATINO)': '6',
             'AMERICAN INDIAN OR ALASKA NATIVE (NONHISPANIC/LAT)': '4',
             'NATIVE HAWAIIAN/OTHER PACIFIC ISLANDER (NON HIS)': '5',
             'OTHER': '-1',
             'NATIVE AMERICAN/ALASKAN': '4',
             'UNKNOWN': '-1',
             'Black': '2',
             'American Indian/Alaskan Native': '4',
             '0': '-1',
              'Declined to Specify': '-1',
             'American Indian or Alaska Native': '4',
             'Non Resident Alien': '-1',
             'Unknown': '-1',
             'Black/African American': '2',
             'Two-or-more': '6',
             'Native Hawaiian/Oth Pac Island': '5',
             'Not Applicable': '-1',
             'NHISP': '3',
             'not available': '-1',
             '2+RACE': '6',
             ' ': '-1'}

enc_ethnicity = ethnicity.copy()
for key, val in D_matching.items():
    enc_ethnicity[ethnicity == key] = val
enc_ethnicity = enc_ethnicity.to_numpy(dtype=int)[p]

employer = Xg[:,301:305]
enc_employer = (employer * np.arange(1,5)).sum(axis=1)

# ------------- ATE estimation with machine-learning models ------------- #

def AIPW_estimator(ms, mg, Xs, Xg, ys, yg):
    
    # Output response model (salary)
    Xs[:,301] = 1 # Male
    y1 = ms.predict(Xs)
    Xs[:,301] = 0 # Female
    y0 = ms.predict(Xs)
    # Output propensity model (gender)
    e = mg.predict_proba(Xg)[:,1]
    e1 = np.zeros_like(y1)
    e0 = np.zeros_like(y1)
    mask = (yg == 1)
    e1[mask] = (ys[mask] - y1[mask]) / e[mask]
    e0[~mask] = (ys[~mask] - y0[~mask]) / (1 - e[~mask])   
    ite = y1 - y0 + e1 - e0
    return ite

def ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets, subset_size, n_splits,
                n_jobs, matched_data, rotated_data=False):
    
    def foo(Xs, Xg, ys, yg):
        
        # For each subset, make splits
        kf = KFold(n_splits=n_splits, random_state=0)
        # Store ITEs
        ite = []
        # For each split
        for idx1, idx2 in kf.split(Xs):
            # Init models
            ms = HistGradientBoostingRegressor()
            mg = HistGradientBoostingClassifier()
            # Train models
            ms.fit(Xs[idx1], ys[idx1])
            mg.fit(Xg[idx1], yg[idx1])
            # Make estimates on test set
            ite.append(AIPW_estimator(
                ms, mg, Xs[idx2], Xg[idx2], ys[idx2], yg[idx2]))
        # Return mean ite and n_employees
        return np.concatenate(ite).mean(), len(Xs)
    
    # Make subsets
    if subset_size == 1:
        gss = LeaveOneGroupOut()
        subsets = gss.split(Xs, ys, groups)
    elif subset_size < 14:
        gss = GroupShuffleSplit(
            n_subsets, test_size=subset_size, random_state=0)
        subsets = gss.split(Xs, ys, groups)
    else: # subset_size = 14
        subsets = [(None, np.arange(len(Xs)))]
    
    output = Parallel(n_jobs=n_jobs)(delayed(foo)(
        Xs[idx], Xg[idx], ys[idx], yg[idx]) for _, idx in subsets)
    # Save a 2D array with ATEs and n_employees
    if rotated_data:
        np.savetxt(fname=f'../results/ATE_subsets/model/{subset_size}DB_matched={matched_data}_rotated.csv',
                   X=np.array(output))
    else:
        np.savetxt(fname=f'../results/ATE_subsets/model/{subset_size}DB_matched={matched_data}.csv',
                   X=np.array(output))
    return output
    
### ATE with matched/raw data

# # ATE on 1 DB
# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=14, subset_size=1,
#             n_splits=5, n_jobs=14, matched_data=False)
# ATE_subsets(Xs_cft, Xg_cft, ys, yg, groups, n_subsets=14, subset_size=1,
#             n_splits=5, n_jobs=14, matched_data=True)
# print('1 DB: Done')

# # ATE on 3 DB
# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=14, subset_size=3,
#             n_splits=5, n_jobs=14, matched_data=False)
# ATE_subsets(Xs_cft, Xg_cft, ys, yg, groups, n_subsets=14, subset_size=3,
#             n_splits=5, n_jobs=14, matched_data=True)
# print('3 DB: Done')

# # ATE on 7 DB
# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=14, subset_size=7,
#             n_splits=5, n_jobs=14, matched_data=False)
# ATE_subsets(Xs_cft, Xg_cft, ys, yg, groups, n_subsets=14, subset_size=7,
#             n_splits=5, n_jobs=14, matched_data=True)
# print('7 DB: Done')

# # ATE on 14 DB 
# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=1, subset_size=14,
#             n_splits=5, n_jobs=1, matched_data=False)
# ATE_subsets(Xs_cft, Xg_cft, ys, yg, groups, n_subsets=1, subset_size=14,
#             n_splits=5, n_jobs=1, matched_data=True)
# print('14 DB: Done')

# ## ATE with matching errors, simulated by a rotation on half the databases
# rotated_groups = np.random.choice(14, 7, replace=False)
# mask = np.zeros(len(groups), dtype=bool)
# for k, gk in enumerate(groups):
#     if gk in rotated_groups:
#         mask[k] = True
# Xs[mask,:300] = Xs[mask,:300][:,::-1]
# Xg[mask,:300] = Xg[mask,:300][:,::-1]

# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=14, subset_size=1,
#             n_splits=5, n_jobs=14, matched_data=False, rotated_data=True)

# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=14, subset_size=3,
#             n_splits=5, n_jobs=14, matched_data=False, rotated_data=True)

# ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=14, subset_size=7,
#             n_splits=5, n_jobs=14, matched_data=False, rotated_data=True)

# output = ATE_subsets(Xs, Xg, ys, yg, groups, n_subsets=1, subset_size=14,
#             n_splits=5, n_jobs=1, matched_data=False, rotated_data=True)

# ------------- ATE with empirical IPW ------------- #

import numpy_indexed as npi

def ATE_emp_IPW(Xg, yg, ys, dismiss):
    idx_cat = npi.group_by(Xg).split(np.arange(len(yg)))
    ATE = 0
    n_emp = 0
    cpt, cpt2 = 0, 0
    for ic in idx_cat:
        m1 = yg[ic] == 1
        m0 = yg[ic] == 0
        if dismiss:
            # Dismiss categories with only men/women
            if m1.any() and m0.any():
                ate_cat = (ys[ic][m1].mean() - ys[ic][m0].mean()) * len(ic)
                ATE += ate_cat
                n_emp += len(ic)
                cpt += 1
            else:
                cpt2 += len(ic)
        else:
            # Keep categories with only men/women
            if not m1.any():
                y1 = 0
            else:
                y1 = ys[ic][m1].mean()
            if not m0.any():
                y0 = 0
            else:
                y0 = ys[ic][m0].mean()
            ate_cat = (y1 - y0) * len(ic)
            ATE += ate_cat
            n_emp += len(ic)
    ATE = ATE / n_emp
    print(cpt2)
    return ATE

def ATE_emp_IPW_subset(Xg, yg, ys, groups, n_subsets, subset_size, n_jobs, dismiss):
    
    # Make subsets
    if subset_size == 1:
        gss = LeaveOneGroupOut()
        subsets = gss.split(Xs, ys, groups)
    elif subset_size < 14:
        gss = GroupShuffleSplit(
            n_subsets, test_size=subset_size, random_state=0)
        subsets = gss.split(Xs, ys, groups)
    else: # subset_size = 14
        subsets = [(None, np.arange(len(Xs)))]
        
    output = Parallel(n_jobs=n_jobs)(delayed(ATE_emp_IPW)(
            Xg[idx], yg[idx], ys[idx], dismiss) for _, idx in subsets)
    # Save a 2D array with ATEs and n_employees
    np.savetxt(fname=f'../results/ATE_subsets/{subset_size}DB_empirical_IPW_dismiss={dismiss}.csv',
               X=np.array(output))
    return output

from sklearn.preprocessing import OrdinalEncoder
clean_jt_enc = OrdinalEncoder().fit_transform(dirty_jt[:,None])[:,0]
Xg_emp = np.zeros((len(Xg), 4))
Xg_emp[:,0] = clean_jt_enc
Xg_emp[:,1] = Xg[:,300]
Xg_emp[:,2] = enc_employer
Xg_emp[:,3] = enc_ethnicity

# # ATE on 1 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=14, subset_size=1, n_jobs=8, dismiss=False)
# print('Done')
# # ATE on 3 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=14, subset_size=3, n_jobs=8, dismiss=False)
# print('Done')
# # ATE on 7 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=14, subset_size=7, n_jobs=8, dismiss=False)
# print('Done')
# # ATE on 14 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=1, subset_size=14, n_jobs=8, dismiss=False)
# print('Done')


# # ATE on 1 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=14, subset_size=1, n_jobs=8, dismiss=True)
# print('Done')
# # ATE on 3 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=14, subset_size=3, n_jobs=8, dismiss=True)
# print('Done')
# # ATE on 7 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=14, subset_size=7, n_jobs=8, dismiss=True)
# print('Done')
# # ATE on 14 DB
# ATE_emp_IPW_subset(Xg_emp, yg, ys, groups, n_subsets=1, subset_size=14, n_jobs=8, dismiss=True)
ATE_emp_IPW(Xg_emp, yg, ys, dismiss=True)
print('Done')

# ------------- ATE with fuzzy matching model ------------- #

import numpy_indexed as npi
import numexpr as ne
from sklearn.metrics.pairwise import cosine_similarity
from time import time

Xg_fuzzy = Xg[:,:303].copy()
Xg_fuzzy[:,301] = enc_employer
Xg_fuzzy[:,302] = enc_ethnicity

def ATE_Fuzzy_Matching(Xg, yg, ys, t):
    
    ### Compute propensity score
    # Group by covariates and compute propensity-score for each group
    idx_cat = npi.group_by(Xg).split(np.arange(len(yg)))
    n_emp_cat = np.array([len(icat) for icat in idx_cat])
    cov_cat, propensity_cat = npi.group_by(Xg).mean(yg)
    # Compute similarity score between all groups
    print('Start 2')
    t0 = time()
    a, b = 1/(1-t), -t/(1-t)
    jt_sim = a * cosine_similarity(cov_cat[:,:300], cov_cat[:,:300]) + b
    print('cos_sim', time() - t0); t0 = time()
    jt_sim[jt_sim < 0] = 0
    print('clip', time() - t0); t0 = time()
    exp_sim = np.array([np.abs(exp - cov_cat[:,300]) <= 2 for exp in cov_cat[:,300]])
    print('exp_sim', time() - t0); t0 = time()
    employer_sim = np.array([emp == cov_cat[:,301] for emp in cov_cat[:,301]])
    print('emp_sim', time() - t0); t0 = time()
    ethnicity_sim = np.array([eth == cov_cat[:,302] for eth in cov_cat[:,302]])
    print('eth_sim', time() - t0); t0 = time()
    sim = ne.evaluate('jt_sim * exp_sim * employer_sim * ethnicity_sim')
    print('sim_product', time() - t0); t0 = time()
    sum_sim = sim.sum(axis=1)
    print('sum_sim', time() - t0); t0 = time()
    # Recompute propensity-score as a weighted sum across groups
    e1_cat_fuzzy = (sim @ propensity_cat[:,None])[:,0] / sum_sim
    print('e1_cat_fuzzy', time() - t0); t0 = time()
    
    ### Compute outcome y1 and y0
    y1_cat, y0_cat = np.zeros(len(cov_cat)), np.zeros(len(cov_cat))
    for k, icat in enumerate(idx_cat):
        ys_cat = ys[icat]
        m1 = yg[icat] == 1
        y1_cat[k] = np.mean(ys_cat[m1])
        y0_cat[k] = np.mean(ys_cat[~m1])
    print('compute y1_cat, y0_cat', time() - t0); t0 = time()
    mask1 = (~np.isnan(y1_cat)).astype(int)
    sim1 = ne.evaluate('sim * mask1')
    print('sim1', time() - t0); t0 = time()
    mask0 = (~np.isnan(y0_cat)).astype(int)
    sim0 = ne.evaluate('sim * mask0')
    print('sim0', time() - t0); t0 = time()
    sum_sim1 = sim1.sum(axis=1)
    print('sum_sim1', time() - t0); t0 = time()
    sum_sim0 = sim0.sum(axis=1)
    print('sum_sim0', time() - t0); t0 = time()
    #print(len(np.where(sum_sim1 == 0)[0]), len(np.where(sum_sim0 == 0)[0]), len(sum_sim0))
    y1_cat[np.isnan(y1_cat)] = 0
    y0_cat[np.isnan(y0_cat)] = 0
    # Recompute outcomes as a weighted sum across groups
    y1_cat_fuzzy = (sim1 @ y1_cat[:,None])[:,0] / sum_sim1
    y0_cat_fuzzy = (sim0 @ y0_cat[:,None])[:,0] / sum_sim0
    print('y1/0_cat_fuzzy', time() - t0); t0 = time()
    ### Compute ATE
    aipw_ate_cat = np.zeros_like(n_emp_cat, dtype=float)
    cpt_ipw = 0
    for k, icat in enumerate(idx_cat):
        aipw_ate = 0
        ipw_ate = 0
        for i in range(n_emp_cat[k]):
            Wi, Yi = yg[icat][i], ys[icat][i]
            mu1, mu0 = y1_cat_fuzzy[k], y0_cat_fuzzy[k]
            ei = e1_cat_fuzzy[k]
            if ei == 0:
                ipw_ate += - (1 - Wi) * Yi / (1 - ei)
            elif ei == 1:
                ipw_ate += Wi * Yi / ei
            else:
                ipw_ate += Wi * Yi / ei - (1 - Wi) * Yi / (1 - ei)
            aipw_ate += Wi * (Yi - mu1) / ei - (1 - Wi) * (Yi - mu0) / (1 - ei)
        aipw_ate /= n_emp_cat[k]
        aipw_ate += mu1 - mu0
        ipw_ate /= n_emp_cat[k]
        if np.isnan(aipw_ate):
            aipw_ate_cat[k] = ipw_ate
            cpt_ipw += n_emp_cat[k]
        else:
            aipw_ate_cat[k] = aipw_ate
    print(cpt_ipw, n_emp_cat.sum())
    ATE = (aipw_ate_cat * n_emp_cat).sum() / n_emp_cat.sum()
    return ATE

def ATE_fuzzy_subsets(Xg, yg, ys, groups, n_subsets, subset_size,
                      n_jobs, t, rotated_data=False):
    
    # Make subsets
    if subset_size == 1:
        gss = LeaveOneGroupOut()
        subsets = gss.split(Xg, yg, groups)
    elif subset_size < 14:
        gss = GroupShuffleSplit(
            n_subsets, test_size=subset_size, random_state=0)
        subsets = gss.split(Xg, yg, groups)
    else: # subset_size = 14
        subsets = [(None, np.arange(len(Xg)))]
    
    output = Parallel(n_jobs=n_jobs)(delayed(ATE_Fuzzy_Matching)(
        Xg[idx], yg[idx], ys[idx], t) for _, idx in subsets)
    # Save a 2D array with ATEs and n_employees
    if rotated_data:
        np.savetxt(fname=f'../results/ATE_subsets/fuzzy/{subset_size}DB_rotated.csv',
                   X=np.array(output))
    else:
        np.savetxt(fname=f'../results/ATE_subsets/fuzzy/{subset_size}DB.csv',
                   X=np.array(output))
    return

print('Start')
# ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=14, subset_size=1,
#                   n_jobs=14, t=0.8, rotated_data=False)
# print('1 DB: Done')

# ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=14, subset_size=3,
#                   n_jobs=14, t=0.8, rotated_data=False)
# print('3 DB: Done')

ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=14, subset_size=7,
                  n_jobs=5, t=0.8, rotated_data=False)
print('7 DB: Done')

ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=1, subset_size=14,
                  n_jobs=1, t=0.8, rotated_data=False)
print('14 DB: Done')

## Rotated data
rotated_groups = np.random.choice(14, 7, replace=False)
mask = np.zeros(len(groups), dtype=bool)
for k, gk in enumerate(groups):
    if gk in rotated_groups:
        mask[k] = True
Xg_fuzzy[mask,:300] = Xg_fuzzy[mask,:300][:,::-1]

print('Start')
# ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=14, subset_size=1,
#                   n_jobs=14, t=0.8, rotated_data=True)
# print('1 DB: Done')

# ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=14, subset_size=3,
#                   n_jobs=14, t=0.8, rotated_data=True)
# print('3 DB: Done')

ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=14, subset_size=7,
                  n_jobs=5, t=0.8, rotated_data=True)
print('7 DB: Done')

ATE_fuzzy_subsets(Xg_fuzzy, yg, ys, groups, n_subsets=1, subset_size=14,
                  n_jobs=1, t=0.8, rotated_data=True)
print('14 DB: Done')


# ------------- Comparison of ATE estimates on project manager ------------- #

# variants = ['0361 project manager', '2128 project manager',
#             '9109 project manager', 'mgr project', 'project manager'] #+ ['manager project']

# # from sklearn.preprocessing import OrdinalEncoder
# # dirty_jt_enc = OrdinalEncoder().fit_transform(dirty_jt[:,None])[:,0]

# ### Empirical IPW
# # Raw data
# ATE_var = {}
# for variant in variants:
#     mask = dirty_jt == variant
#     Xg_var, yg_var, ys_var = Xg[mask,300:], yg[mask], ys[mask]
#     ate_var = ATE_emp_IPW(Xg_var, yg_var, ys_var, dismiss=False)
#     n_emp_var = len(yg_var)
#     freq_var = yg_var.mean()
#     ATE_var[variant] = (ate_var, n_emp_var, freq_var)
# emp_IPW_pm_raw = np.array([ATE_var[v][0] for v in variants])
# # Matched data
# mask = [jt in variants for jt in dirty_jt]
# Xg_var, yg_var, ys_var = Xg[mask,300:], yg[mask], ys[mask]
# emp_IPW_pm_matched = ATE_emp_IPW(Xg_var, yg_var, ys_var, dismiss=False)
