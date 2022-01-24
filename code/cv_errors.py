from load_datasets import *
from models import *
import numpy as np
import numpy_indexed as npi
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor,\
HistGradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.base import clone
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

"""
Code to evaluate the estimation errors of matching and learning based
approaches on 3 tasks:

    (1) estimation of the salary conditional to the job title and experience
        level
    (2) estimation of the 0.75-quantile of salary for a given job
    (3) estimation of the propensity-score conditional to the job, i.e. the
        probability of being a man given the job title.
        
To do so, we implement a cross-validation procedure: we compute estimates on
7 tables, and leave out 7 tables to measure the error. This step is repeated
on 40 different random splits.

We compare matching & averaging (called plug-in estimation here), fuzzy
matching with embeddings, and machine learning with embeddings, on raw and
manually curated data. Results are saved in 'results/cv_errors/xxx.csv', and
contain the estimation error and the number of variants in the training data
for each employee in the test data.
Having the number of variants in the training data allows to compare
performances on categories of employees that are or aren't present in the
training data.

"""

def make_splits(X, y, groups, n_splits, test_size, nested_cross_val=False):
    
    if nested_cross_val:
        gss = LeaveOneGroupOut()
    else:
        gss = GroupShuffleSplit(
            n_splits, test_size=test_size, random_state=42)
    splits = []
    for idx1, idx2 in gss.split(X, y, groups):
        splits.append((idx1, idx2))
    return splits

def n_variants(X_nem, X_mem, y, groups, n_splits, test_size, D_var, n_jobs,
               nested_cross_val=False):
    
    def foo(idx1, idx2, X_mem, X_nem, D_var):
        X1_nem, X2_mem = X_nem[idx1], X_mem[idx2]
        X1_nem_cat = np.unique(X1_nem, axis=0)
        X2_mem_cat, inverse_X2 = np.unique(
            X2_mem, axis=0, return_inverse=True)
        n_var_cat = np.zeros(len(X2_mem_cat))
        for k, xc2 in enumerate(X2_mem_cat):
            n_var = 0
            if X_mem.shape[1] == 2: # salary
                key_xc2 = f'{int(xc2[0])}, {int(xc2[1])}'
            else:
                key_xc2 = f'{int(xc2[0])}'
            for xc1 in D_var[key_xc2]:
                n_var += (X1_nem_cat == xc1).all(axis=1).any()
            n_var_cat[k] = n_var
        n_var_split = n_var_cat[inverse_X2]
        return n_var_split
    splits = make_splits(
        X_mem, y, groups, n_splits,test_size, nested_cross_val)
    n_var_splits = Parallel(n_jobs=n_jobs)(delayed(foo)(
        idx1, idx2,  X_mem, X_nem, D_var) for idx1, idx2 in splits)
    return np.concatenate(n_var_splits)

def make_D_var(X_nem, X_mem, n_jobs):
    
    def foo(k, xc, inverse, X_nem):
        mask = inverse == k
        if X_mem.shape[1] == 2: # salary
            key = f'{int(xc[0])}, {int(xc[1])}'
        else:
            key = f'{int(xc[0])}'
        return key, np.unique(X_nem[mask], axis=0).astype(int)
        
    unq_X_mem, inverse = np.unique(X_mem, axis=0, return_inverse=True)
    output = Parallel(n_jobs)(
        delayed(foo)(k, xc, inverse, X_nem) for k, xc in enumerate(unq_X_mem))
    return dict(output)

def cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size,
              n_jobs, nested_cross_val=False):
    
    def foo(task, model, X, X_mem, y, groups, idx1, idx2):
        model_ = clone(model)
        model_.groups1 = groups[idx1]
        model_.X1_nem = X_nem[idx1]
        model_.X1_mem = X_mem[idx1]
        # Train model
        X1, y1 = X[idx1], y[idx1]
        model_.fit(X1, y1)
        # Make estimates (on the cleaned jobs since it is more realistic)
        X2, X2_mem, y2 = X[idx2], X_mem[idx2], y[idx2]
        if task == 'quantile':
            X2_mem_cat, inverse_X2 = np.unique(X2_mem, axis=0, return_inverse=True)
            y2_cat = np.array([np.quantile(
                y2[(X2_mem == xc2)[:,0]], q=0.75) for xc2 in X2_mem_cat])
            y2 = y2_cat[inverse_X2]
        if task == 'sex':
            y2_pred = model_.predict_proba(X2)[:,1]
        else:
            y2_pred = model_.predict(X2)
        # Return cross-val errors
        cv_err = np.abs(y2 - y2_pred)
        return cv_err
    
    splits = make_splits(
        X_mem, y, groups, n_splits, test_size, nested_cross_val)
    output = Parallel(n_jobs=n_jobs)(delayed(foo)(
        task, model, X, X_mem, y, groups, idx1, idx2) for idx1, idx2 in splits)
    return np.concatenate(output)

if __name__ == '__main__':
    
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
    p = np.random.RandomState(seed=42).permutation(len(gender))
    Xg, Xs, gender, salary, dirty_jt, groups = Xg[p], Xs[p], gender[p], salary[p], dirty_jt[p], groups[p]
    Xg_cft, Xs_cft, clean_jt = Xg_cft[p], Xs_cft[p], clean_jt[p]

    # No entity matching
    unq_dirty_jt, inverse_dirty_jt = np.unique(dirty_jt, return_inverse=True)
    Xg_nem, Xs_nem = no_entity_matching(Xg, Xs, inverse_dirty_jt)

    # Manual entity matching
    unq_clean_jt, inverse_clean_jt = np.unique(clean_jt, return_inverse=True)
    Xg_mem, Xs_mem = manual_entity_matching(Xg_cft, Xs_cft, inverse_clean_jt)

    print('Data loaded')

    # Parameters
    n_splits = 40
    test_size = 7
    n_jobs = 40


#     #################################################
#     #                    SALARY
#     #################################################

    task = 'salary'
    X_nem, X_mem = Xs_nem[:,:2], Xs_mem[:,:2]
    y = salary
    D_var = make_D_var(X_nem, X_mem, n_jobs)
    n_var = n_variants(X_nem, X_mem, y, groups, n_splits, test_size, D_var, n_jobs)

    # Matching & averaging + manual matching
    X = X_mem
    model = PlugInEstimator(task=task)
    model_name = 'plug_in'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2)**0.5:.0f}, {np.mean(cv_err[n_var==0]**2)**0.5:.0f},\
    {np.mean(cv_err[n_var==1]**2)**0.5:.0f},{np.mean(cv_err[n_var>1]**2)**0.5:.0f}')
    
    # Matching & averaging + partial matching
    X = np.hstack([X_nem, X_mem])
    model = PlugInEstimator2(task, D_var)
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    print(f'matching & averaging + partial matching, {np.mean(cv_err[n_var>1]**2)**0.5:.0f}')
    
    # Fuzzy matching + raw data
    X = np.hstack([Xs[:,:300], X_mem])
    model = FuzzyEstimator(task=task, n_chunks=20, matching=False)
    model_name = 'fuzzy_raw'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2)**0.5:.0f}, {np.mean(cv_err[n_var==0]**2)**0.5:.0f},\
    {np.mean(cv_err[n_var==1]**2)**0.5:.0f},{np.mean(cv_err[n_var>1]**2)**0.5:.0f}')

    # Fuzzy matching + manual matching
    X = np.hstack([Xs_cft[:,:300], X_mem])
    model = FuzzyEstimator(task=task, n_chunks=20, matching=False)
    model_name = 'fuzzy_matched'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2)**0.5:.0f}, {np.mean(cv_err[n_var==0]**2)**0.5:.0f},\
    {np.mean(cv_err[n_var==1]**2)**0.5:.0f},{np.mean(cv_err[n_var>1]**2)**0.5:.0f}')

    # Embeddings + raw data
    X = Xs[:,:301]
    model = tuned_HGB(task=task, learning_rate=None)
    model_name = 'embeddings_raw'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2)**0.5:.0f}, {np.mean(cv_err[n_var==0]**2)**0.5:.0f},\
    {np.mean(cv_err[n_var==1]**2)**0.5:.0f},{np.mean(cv_err[n_var>1]**2)**0.5:.0f}')

    # Embeddings + manual matching
    X = Xs_cft[:,:301]
    model = tuned_HGB(task=task, learning_rate=None)
    model_name = 'embeddings_matched'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2)**0.5:.0f}, {np.mean(cv_err[n_var==0]**2)**0.5:.0f},\
    {np.mean(cv_err[n_var==1]**2)**0.5:.0f},{np.mean(cv_err[n_var>1]**2)**0.5:.0f}')


    # #################################################
    # #                   QUANTILE
    # #################################################
    
    task = 'quantile'
    X_nem, X_mem = Xs_nem[:,:1], Xs_mem[:,:1]
    y = salary
    D_var = make_D_var(X_nem, X_mem, n_jobs)
    n_var = n_variants(X_nem, X_mem, y, groups, n_splits, test_size, D_var, n_jobs)

    # Matching & averaging + manual matching
    X = X_mem
    model = PlugInEstimator(task=task)
    model_name = 'plug_in'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err):.0f}, {np.mean(cv_err[n_var==0]):.0f},\
    {np.mean(cv_err[n_var==1]):.0f},{np.mean(cv_err[n_var>1]):.0f}')
    
    # Matching & averaging + partial matching
    X = np.hstack([X_nem, X_mem])
    model = PlugInEstimator2(task, D_var)
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    print(f'matching & averaging + partial matching, {np.mean(cv_err[n_var>1]):.0f}')
    
    # Fuzzy matching + raw data
    X = np.hstack([Xs[:,:300], X_mem])
    model = FuzzyEstimator(task=task, n_chunks=20, matching=False)
    model_name = 'fuzzy_raw'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err):.0f}, {np.mean(cv_err[n_var==0]):.0f},\
    {np.mean(cv_err[n_var==1]):.0f},{np.mean(cv_err[n_var>1]):.0f}')

    # Fuzzy matching + manual matching
    X = np.hstack([Xs_cft[:,:300], X_mem])
    model = FuzzyEstimator(task=task, n_chunks=20, matching=False)
    model_name = 'fuzzy_matched'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err):.0f}, {np.mean(cv_err[n_var==0]):.0f},\
    {np.mean(cv_err[n_var==1]):.0f},{np.mean(cv_err[n_var>1]):.0f}')

    # Embeddings + raw data
    X = Xs[:,:300]
    model = GradientBoostingRegressor(loss='quantile', alpha=0.75)
    model_name = 'embeddings_raw'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err):.0f}, {np.mean(cv_err[n_var==0]):.0f},\
    {np.mean(cv_err[n_var==1]):.0f},{np.mean(cv_err[n_var>1]):.0f}')

    # Embeddings + manual matching
    X = Xs_cft[:,:300]
    model = GradientBoostingRegressor(loss='quantile', alpha=0.75)
    model_name = 'embeddings_matched'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err):.0f}, {np.mean(cv_err[n_var==0]):.0f},\
    {np.mean(cv_err[n_var==1]):.0f},{np.mean(cv_err[n_var>1]):.0f}')


    # #################################################
    # #                      SEX
    # #################################################

    task = 'sex'
    X_nem, X_mem = Xs_nem[:,:1], Xs_mem[:,:1]
    y = gender
    D_var = make_D_var(X_nem, X_mem, n_jobs)
    n_var = n_variants(X_nem, X_mem, y, groups, n_splits, test_size, D_var, n_jobs)

    # Matching & averaging + manual matching
    X = X_mem
    model = PlugInEstimator(task=task)
    model_name = 'plug_in'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2):.4f}, {np.mean(cv_err[n_var==0]**2):.4f},\
    {np.mean(cv_err[n_var==1]**2):.4f},{np.mean(cv_err[n_var>1]**2):.4f}')
    
    # Matching & averaging + partial matching
    X = np.hstack([X_nem, X_mem])
    model = PlugInEstimator2(task, D_var)
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    print(f'matching & averaging + partial matching, {np.mean(cv_err[n_var>1]**2):.4f}')
    
    # Fuzzy matching + raw data
    X = np.hstack([Xs[:,:300], X_mem])
    model = FuzzyEstimator(task=task, n_chunks=20, matching=False)
    model_name = 'fuzzy_raw'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2):.4f}, {np.mean(cv_err[n_var==0]**2):.4f},\
    {np.mean(cv_err[n_var==1]**2):.4f},{np.mean(cv_err[n_var>1]**2):.4f}')

    # Fuzzy matching + manual matching
    X = np.hstack([Xs_cft[:,:300], X_mem])
    model = FuzzyEstimator(task=task, n_chunks=20, matching=False)
    model_name = 'fuzzy_matched'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2):.4f}, {np.mean(cv_err[n_var==0]**2):.4f},\
    {np.mean(cv_err[n_var==1]**2):.4f},{np.mean(cv_err[n_var>1]**2):.4f}')

    # Embeddings + raw data
    X = Xs[:,:300]
    model = tuned_HGB(task=task, learning_rate=None)
    model_name = 'embeddings_raw'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2):.4f}, {np.mean(cv_err[n_var==0]**2):.4f},\
    {np.mean(cv_err[n_var==1]**2):.4f},{np.mean(cv_err[n_var>1]**2):.4f}')

    # Embeddings + manual matching
    X = Xs_cft[:,:300]
    model = tuned_HGB(task=task, learning_rate=None)
    model_name = 'embeddings_matched'
    cv_err = cv_errors(task, model, X, X_nem, X_mem, y, groups, n_splits, test_size, n_jobs)
    df = pd.DataFrame(np.array([n_var, cv_err]).T, columns=['n_variants', 'cv_error'])
    df.to_csv(f'../results/cv_errors/{task}_{model_name}.csv', index=False)
    print(f'ERROR {model_name}: {np.mean(cv_err**2):.4f}, {np.mean(cv_err[n_var==0]**2):.4f},\
    {np.mean(cv_err[n_var==1]**2):.4f},{np.mean(cv_err[n_var>1]**2):.4f}')