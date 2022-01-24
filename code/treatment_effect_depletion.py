import numpy as np
import numpy_indexed as npi
from numpy import dot
from numpy.linalg import norm
import numexpr as ne
import pandas as pd
from load_datasets import *
from models import *
from time import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, KFold
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

"""
This code compares average salary gap estimates across different
methods (matching & averaging, fuzzy matching with embeddings, machine
learning with embbedings) on 14 employee tables that are progressively
depleted from either male or female employees.

In practice, for a given depletion rate dr between 0 and 1 (sex imbalance in
the paper), a fraction dr of the male/female employees are removed from each
table. The choice of sex is random for each table, so that in total 7 tables
are depleted from men, and 7 from women. When dr = 0, the original data is
used. When dr = 1, tables only contains either men or women.

Increasing the sex imbalance forces the estimation model to compare similar
employees across tables with different job titles variants, which typically
requires entity matching across tables. We show that a simple approach based
on machine-learning with embeddings perform better, since it can leverage job
similarities, even when there is no clear correspondence between entries.

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

# Manual matching of the ethnicities
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

clean_jt_enc = OrdinalEncoder().fit_transform(clean_jt[:,None])[:,0]
Xg_matching = np.zeros((len(Xg), 4))
Xg_matching[:,0] = clean_jt_enc
Xg_matching[:,1] = Xg[:,300]
Xg_matching[:,2] = enc_employer
Xg_matching[:,3] = enc_ethnicity

Xg_fuzzy = Xg[:,:303].copy()
Xg_fuzzy[:,301] = enc_employer
Xg_fuzzy[:,302] = enc_ethnicity

# ------------- ATE estimation with machine-learning models ------------- #

def AIPW_estimator(ms, mg, Xs, Xg, ys, yg):
    
    """
    Augmented Inverse Propensity Weighting Estimator with maching machine-learning
    models. We use gradient boosted trees to model the outcome and the
    propensity-score.
    """
    
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

def ATE_learning(Xs, Xg, ys, yg, tuning=True):
    """
    Tune the hyperparameters of the models, and compute the average treatment
    effect with an AIPW estimator.
    """
    
    # Init models
    if tuning:
        est_s = HistGradientBoostingRegressor()
        est_g = HistGradientBoostingClassifier()
        parameters = {'learning_rate':[0.1, 0.3, 0.5], 'max_depth': [8, 12, None]}
        ms = GridSearchCV(est_s, parameters, cv=5)
        mg = GridSearchCV(est_g, parameters, cv=5)
    else:
        ms = HistGradientBoostingRegressor()
        mg = HistGradientBoostingClassifier()
    # Train models
    ms.fit(Xs, ys)
    mg.fit(Xg, yg)
    if tuning:
        print(ms.best_params_, mg.best_params_)
    # Make estimates on test set
    ite = AIPW_estimator(ms, mg, Xs.copy(), Xg, ys, yg)
    # Return mean ite and n_employees
    return ite.mean(), len(ys)

# ------------- ATE with covariates matching ------------- #

def ATE_matching(Xs, Xg, ys, yg, dismiss=True):
    """
    Compute the average treatment effect with matching & averaging.
    Only matched employees with same covariates but different sex are used
    to compute the average salary gap, others are dismissed.
    """
    
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
    return ATE, len(ys), 1 - (cpt2 / len(ys))

# ------------- ATE with fuzzy matching model ------------- #

def ATE_fuzzy(Xs, Xg, ys, yg, threshold=0.8, n_chunks=20):
    
    """
    Compute the ATE with fuzzy matching based on job titles embeddings.
    For a group of employees with given covariates and sex, we use employees
    with 'similar' covariates but different sex to measure the salary gap
    for that group. ATEs on each group are then averaged to form the global
    ATE.
    
    'Similar' employees are selected based on the cosine similarity of their
    job titles, and a matching threshold.
    """
    
    idx_cat = npi.group_by(Xg).split(np.arange(len(yg)))
    n_emp_cat = np.array([len(icat) for icat in idx_cat])
    cov_cat, propensity_cat = npi.group_by(Xg).mean(yg)
    _, sal_cat = npi.group_by(Xg).mean(ys)
    idx_cat_both_sex = np.where((propensity_cat != 1) & (propensity_cat != 0))[0]
    only_men = propensity_cat == 1
    only_women = propensity_cat == 0
    cov_cat_men, cov_cat_women = cov_cat[only_men], cov_cat[only_women]
    sal_cat_men, sal_cat_women = sal_cat[only_men], sal_cat[only_women]
    n_emp_cat_men, n_emp_cat_women = n_emp_cat[only_men], n_emp_cat[only_women]
    
    ### Normalize embeddings
    cov_cat_men[:,:300] /= norm(cov_cat_men[:,:300], axis=1)[:,None]
    cov_cat_women[:,:300] /= norm(cov_cat_women[:,:300], axis=1)[:,None]
    
    ### ATE on categories with both sex
    ate_cat = []
    weight_ate_cat = []
    for k in idx_cat_both_sex:
        icat = idx_cat[k]
        ys_cat, yg_cat = ys[icat], yg[icat]
        m1 = yg_cat == 1
        ate_cat.append(ys_cat[m1].mean() - ys_cat[~m1].mean())
        weight_ate_cat.append(len(icat))
    
    ### Make chunks
    chunks = np.array_split(np.arange(len(cov_cat_men)), n_chunks)
    for chunk in chunks:
        ### Fuzzy matching from cat_men to cat_women
        jt_sim = (cov_cat_men[chunk,:300] @ cov_cat_women[:,:300].T) > threshold
        for jt_sim_cat_m, n_emp_cat_m, sal_cat_m, cat_m in zip(
            jt_sim, n_emp_cat_men[chunk], sal_cat_men[chunk], cov_cat_men[chunk]):
            other_sim = np.all(cat_m[300:] == cov_cat_women[:,300:], axis=1)
            mask_sim = jt_sim_cat_m & other_sim
            if mask_sim.any():
                ate_cat.append(sal_cat_m - sal_cat_women[mask_sim].mean())
                weight_ate_cat.append(n_emp_cat_m)
    
    ### Make chunks
    chunks = np.array_split(np.arange(len(cov_cat_women)), n_chunks)
    for chunk in chunks:
        ### Fuzzy matching from cat_men to cat_women
        jt_sim = (cov_cat_women[chunk,:300] @ cov_cat_men[:,:300].T) > threshold
        for jt_sim_cat_w, n_emp_cat_w, sal_cat_w, cat_w in zip(
            jt_sim, n_emp_cat_women[chunk], sal_cat_women[chunk], cov_cat_women[chunk]):
            other_sim = np.all(cat_w[300:] == cov_cat_men[:,300:], axis=1)
            mask_sim = jt_sim_cat_w & other_sim
            if mask_sim.any():
                ate_cat.append(sal_cat_men[mask_sim].mean() - sal_cat_w)
                weight_ate_cat.append(n_emp_cat_w)
    
    ate_cat, weight_ate_cat = np.array(ate_cat), np.array(weight_ate_cat)
    ATE = (ate_cat * weight_ate_cat).sum() / weight_ate_cat.sum()
    return ATE, weight_ate_cat.sum() / len(ys)

# ------------- ATE estimation on depleted data ------------- #

def make_setups(Xs, Xg, ys, yg, groups, splits, n_depletions,
                depletion_rate, n_jobs, corrupted_dbs=None):
    
    def rdm_split(Xs, Xg, ys, yg, groups, split, n_depletions, depletion_rate, corrupted_dbs):
        
        setups = []
        Xs_copy, Xg_copy = Xs.copy(), Xg.copy()
        # Corrupt embeddings of half the DBs
        mask = np.sum([groups == db for db in corrupted_dbs], axis=0).astype(bool)
        Xs_copy[mask,:300] = Xs_copy[mask,:300][:,::-1]
        Xg_copy[mask,:300] = Xg_copy[mask,:300][:,::-1]
        # Make random split
        db_men = split
        mask = np.sum([groups == db for db in db_men], axis=0).astype(bool)
        Xs1, Xs0 = Xs_copy[mask], Xs_copy[~mask]
        Xg1, Xg0 = Xg_copy[mask], Xg_copy[~mask]
        ys1, ys0 = ys[mask], ys[~mask]
        yg1, yg0 = yg[mask], yg[~mask]
        # Make random depletion
        idx1, idx0 = np.where(yg1 == 0)[0], np.where(yg0 == 1)[0]
        for _ in range(n_depletions):
            idx1_del = np.random.choice(idx1, int(len(idx1) * depletion_rate), replace=False)
            idx0_del = np.random.choice(idx0, int(len(idx0) * depletion_rate), replace=False)
            Xs1_, Xs0_ = np.delete(Xs1, idx1_del, 0), np.delete(Xs0, idx0_del, 0)
            Xg1_, Xg0_ = np.delete(Xg1, idx1_del, 0), np.delete(Xg0, idx0_del, 0)
            ys1_, ys0_ = np.delete(ys1, idx1_del, 0), np.delete(ys0, idx0_del, 0)
            yg1_, yg0_ = np.delete(yg1, idx1_del, 0), np.delete(yg0, idx0_del, 0)
            Xs_, Xg_ = np.concatenate([Xs1_, Xs0_]), np.concatenate([Xg1_, Xg0_])
            ys_, yg_ = np.concatenate([ys1_, ys0_]), np.concatenate([yg1_, yg0_])
            p = np.random.permutation(len(ys_))
            setups.append((Xs_[p], Xg_[p], ys_[p], yg_[p]))
        return setups
    
    # Make list of depleted setups
    setups = Parallel(n_jobs=n_jobs)(delayed(rdm_split)(
        Xs, Xg, ys, yg, groups, split, n_depletions,
        depletion_rate, corr_dbs) for split, corr_dbs in zip(splits, corrupted_dbs))
    setups = [item for sublist in setups for item in sublist]
    return setups

def ATE_depleted(estimator, Xs, Xg, ys, yg, groups, splits, n_depletions,
                depletion_rate, n_jobs, corrupted_dbs, fname=None):
    
    setups = make_setups(Xs, Xg, ys, yg, groups, splits, n_depletions,
                         depletion_rate, n_jobs, corrupted_dbs)
    ATEs = Parallel(n_jobs=n_jobs)(delayed(estimator)(
        Xs_, Xg_, ys_, yg_) for Xs_, Xg_, ys_, yg_ in setups)
    #ATEs = [estimator(Xs_, Xg_, ys_, yg_) for Xs_, Xg_, ys_, yg_ in setups]    
    if fname != None:
        np.savetxt(fname=f'../results/ATE_depleted/{fname}.csv', X=np.array(ATEs))
    else:
        return ATEs
    return

# ------------- Experiments ------------- #

n_splits = 500 # The number of random male/female splits on which the ATE is estimated
np.random.seed(0)
splits = np.array([np.random.choice(14, 7, replace=False) for k in range(n_splits)])

corrupted_dbs = np.array([np.random.choice(14, 7, replace=False) for k in range(n_splits)])
corrupted_dbs[:] = -1 # comment to artificially corrupt the data (embeddings from 7 tables are randomly permuted)

n_depletions = 1 # Number of random depletions to perform. Doesn't have a strong influence so it is set to 1.
n_jobs = 50
dr_list = [30, 55, 75, 88, 97, 100] 

### Matching
for dr in dr_list:
    print(f'{dr}%')
    ATE_depleted(ATE_matching, Xg_matching, Xg_matching, ys, yg, groups,
                 splits, n_depletions, depletion_rate=dr/100, n_jobs=n_jobs,
                 fname=f'matching_{dr}%')

splits_list = np.array_split(splits, 10) # Cut the splits into chunks to avoid memory errors.
corrupted_dbs_list = np.array_split(corrupted_dbs, 10)

# ### Learning
# for dr in dr_list:
#     print(f'{dr}%')
#     for k, (splits_k, corrupted_dbs_k) in enumerate(zip(splits_list, corrupted_dbs_list)):
#         print(f'Split {k}')
#         ATE_depleted(ATE_learning, Xs, Xg, ys, yg, groups,
#                      splits_k, n_depletions, depletion_rate=dr/100, n_jobs=n_jobs,
#                      corrupted_dbs=corrupted_dbs_k, fname=f'learning_{dr}%_{k}')

### Tuned learning
for dr in dr_list:
    print(f'{dr}%')
    for k, (splits_k, corrupted_dbs_k) in enumerate(zip(splits_list, corrupted_dbs_list)):
            print(f'Split {k}')
            ATE_depleted(ATE_learning, Xs, Xg, ys, yg, groups,
                         splits_k, n_depletions, depletion_rate=dr/100, n_jobs=n_jobs,
                         corrupted_dbs=corrupted_dbs_k, fname=f'learning_tuned_{dr}%_{k}')

### Fuzzy
for dr in dr_list:
    print(f'{dr}%')
    for k, (splits_k, corrupted_dbs_k) in enumerate(zip(splits_list, corrupted_dbs_list)):
        print(f'Split {k}')
        ATE_depleted(ATE_fuzzy, Xg_fuzzy, Xg_fuzzy, ys, yg, groups,
                     splits_k, n_depletions, depletion_rate=dr/100, n_jobs=n_jobs,
                     corrupted_dbs=corrupted_dbs_k, fname=f'fuzzy_corrupted_true_{dr}%_{k}')
        
### ATE estimates for depletion rate = 0

print('Matching & averaging', ATE_matching(Xs, Xg, ys, yg, dismiss=True))
print('Fuzzy matching', ATE_fuzzy(Xg_fuzzy, Xg_fuzzy, ys, yg, threshold=0.8, n_chunks=40)
print('Embedding & learning', ATE_learning(Xs, Xg, ys, yg, tuning=True))