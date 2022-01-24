import numpy as np
import numpy_indexed as npi
from sklearn.base import BaseEstimator, clone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from joblib import Parallel, delayed
from time import time
from cv_errors import *

"""
Implementations of the estimation methods studied in our experiments:
    
    - tuned_HGB for machine-learning with embeddings
    - FuzzyEstimator for fuzzy matching with embeddings
    - PlugInEstimator for matching & averaging with manual matching
    - PlugInEstimator2 for matching & averaging with partial matching
    (pick one random variant from the train data instead of merging them all)

"""

class tuned_HGB(BaseEstimator):
    
    """
    Scikit-learn histogram gradient-boosted tree models, tuned with nested
    cross-validation to minimize the error on a unseen table.
    
    Parameters
    ----------
    
    task : str
        The estimation task to perform, either 'salary', 'quantile', or 'sex'.
    learning_rate : None or float
        The learning rate of the model. If None, a nested cross-validation
        procedure is used to determine the best one.
    fit_on : str
        If fit_on = 'all', all the validation data is used to compute the
        validation error. Set fit_on = 'seen' or 'unseen' to optimize the
        learning rate for unseen or seen categories only.
        
    """
    
    def __init__(self, task, learning_rate=None, fit_on='all'):
        
        self.task = task
        self.learning_rate = learning_rate
        self.fit_on = fit_on
        return
    
    def param_tuning(self, X1, y1):
        
        D_var = make_D_var(self.X1_nem, self.X1_mem, n_jobs=1)
        n_var = n_variants(self.X1_nem, self.X1_mem, y1, self.groups1, n_splits=None,
                           test_size=None, D_var=D_var, n_jobs=1, nested_cross_val=True)
        lr_list = np.logspace(-2, -0.5, 4)
        res = np.zeros(len(lr_list))
        for k in range(len(lr_list)):
            if self.task == "salary":
                self2 = HistGradientBoostingRegressor(learning_rate=lr_list[k])
            else:
                self2 = HistGradientBoostingClassifier(learning_rate=lr_list[k])
            cv_err = cv_errors(self.task, self2, X1, self.X1_nem, self.X1_mem, y1, self.groups1,
                               n_splits=None, test_size=None, n_jobs=1, nested_cross_val=True)
            if self.task != 'quantile':
                cv_err = cv_err**2
            if self.fit_on == 'unseen':
                res[k] = cv_err[n_var == 0].mean()
            elif self.fit_on == 'seen':
                res[k] = cv_err[n_var >= 1].mean()
            else:
                res[k] = cv_err.mean()
        self.learning_rate = lr_list[np.argmin(res)]
        print(int(sum(n_var == 0)/len(n_var)*100)/100)
        return
    
    def fit(self, X1, y1):
        
        # Parameter tuning
        if self.learning_rate == None:
            self.param_tuning(X1, y1)
            print(self.learning_rate)
        # Fit on all train data with tuned params
        if self.task == "salary":
            self.model = HistGradientBoostingRegressor(learning_rate=self.learning_rate)
        else:
            self.model = HistGradientBoostingClassifier(learning_rate=self.learning_rate)
        self.model.fit(X1, y1)
        return
    
    def predict(self, X2):
        return self.model.predict(X2)
    
    def predict_proba(self, X2):
        return self.model.predict_proba(X2)
        
class FuzzyEstimator(BaseEstimator):
    
    """
    Fuzzy matching with embeddings of the job titles. Matching & averaging
    estimates of similar groups of employees are averaged to form an estimate
    for the group of interest. Groups estimates are weighted with a triangular
    kernel based on the cosine similarity of their job title embeddings with
    the job title of interest.
    
    Parameters
    ----------
    
    task : str
        The estimation task to perform, either 'salary', 'quantile', or 'sex'.
    n_chunks : int
        The number of chunks used to split the data. Increase the value to
        avoid memory errors.
    matching : bool
        This parameter is not used in our experiments and must be set to False.
    threshold : float
        The cosine similarity threshold below which other groups estimates are
        not taken into account in the average.
    alpha : float
        This parameter is not used in our experiments.
    fit_on : str
        If fit_on = 'all', all the validation data is used to compute the
        validation error. Set fit_on = 'seen' or 'unseen' to optimize the
        learning rate for unseen or seen categories only.
    """
    
    def __init__(self, task, n_chunks, matching, threshold=0, alpha=1, fit_on='all'):
        
        self.task = task
        self.n_chunks = n_chunks
        self.matching = matching
        self.pie = PlugInEstimator(self.task)
        self.threshold = threshold
        self.alpha = alpha
        self.fit_on = fit_on
        return
    
    def param_tuning(self, X1, y1):
        
        D_var = make_D_var(self.X1_nem, self.X1_mem, n_jobs=1)
        n_var = n_variants(self.X1_nem, self.X1_mem, y1, self.groups1, n_splits=None,
                           test_size=None, D_var=D_var, n_jobs=1, nested_cross_val=True)
        t_list = [0.9, 0.8, 0.7, 0.6, 0.5]
        res = np.zeros(len(t_list))
        for k in range(len(t_list)):
            self2 = FuzzyEstimator(self.task, self.n_chunks, self.matching, t_list[k], 1)
            cv_err = cv_errors(self.task, self2, X1, self.X1_nem, self.X1_mem, y1, self.groups1,
                                   n_splits=None, test_size=None, n_jobs=1, nested_cross_val=True)
            if self.task != 'quantile':
                cv_err = cv_err**2
            if self.fit_on == 'unseen':
                res[k] = cv_err[n_var == 0].mean()
            elif self.fit_on == 'seen':
                res[k] = cv_err[n_var >= 1].mean()
            else:
                res[k] = cv_err.mean()
        self.threshold = t_list[np.argmin(res)]
        
        if self.matching:
            res2 = np.zeros(len(a_list))
            a_list = [0.9, 0.8, 0.7, 0.55, 0.4]
            for k in range(len(a_list)):
                self2 = FuzzyEstimator(self.task, self.n_chunks, self.matching,
                                       self.threshold, a_list[k])
                cv_err = cv_errors(self.task, self2, X1, self.X1_nem, self.X1_mem, y1, self.groups1,
                                   n_splits=None, test_size=None, n_jobs=1, nested_cross_val=True)
                if self.task != 'quantile':
                    cv_err = cv_err**2
                if self.fit_on == 'unseen':
                    res2[k] = cv_err[n_var == 0].mean()
                elif self.fit_on == 'seen':
                    res2[k] = cv_err[n_var >= 1].mean()
                else:
                    res2[k] = cv_err.mean()
            self.alpha = a_list[np.argmin(res2)]
        return
    
    def fit(self, X1, y1):
        """ X1 contains both fastText embeddings and job title IDs."""
        # Parameter tuning
        if self.threshold == 0:
            self.param_tuning(X1, y1)
        # Fit on all train data with tuned params
        X1_ft = X1[:,:300]
        X1 = X1[:,300:]
        X1_cat, indices, counts = np.unique(X1, axis=0, return_counts=True, return_index=True)
        self.pie.fit(X1, y1)
        self.X1_cat = X1_cat
        self.X1_cat_ft = X1_ft[indices]
        self.counts_X1_cat = counts
        return
    
    def predict(self, X2):
        """ X2 contains both fastText embeddings and job title IDs."""
        
        np.seterr(divide='ignore', invalid='ignore')
        X2_ft = X2[:,:300]
        X2 = X2[:,300:]
        X2_cat, indices, inverse = np.unique(X2, axis=0, return_index=True, return_inverse=True)
        X2_cat_ft = X2_ft[indices]
        pie_est_X1 = self.pie.predict(self.X1_cat)
        counts_X1_cat = self.counts_X1_cat
        idx_chunks = np.array_split(np.arange(len(X2_cat)), self.n_chunks)
        # Plug-in estimates
        plugin_cat_est, var_cat_bool = self.pie.predict(X2_cat, return_var_bool=True)
        # Global estimates
        global_cat_est = np.zeros(len(X2_cat)) 
        for ic in idx_chunks:
            t = self.threshold
            a, b = 1/(1-t), -t/(1-t)
            S = a * cosine_similarity(X2_cat_ft[ic], self.X1_cat_ft) + b 
            S = S.clip(0)
            S = S * counts_X1_cat
            if X2_cat.shape[1] == 2:
                S *= np.array([self.X1_cat[:,1] == exp for exp in X2_cat[ic,1]])
            global_cat_est_chunk = np.sum(S * pie_est_X1, axis=1)/S.sum(axis=1)
            mask = ~S.any(axis=1)
            global_cat_est_chunk[mask] = plugin_cat_est[ic][mask]
            global_cat_est[ic] = global_cat_est_chunk
        # Shrinkage estimates
        if self.matching:
            model_cat_est = plugin_cat_est*self.alpha + (1-self.alpha)*global_cat_est
            mask = var_cat_bool == 0
            model_cat_est[mask] = global_cat_est[mask]
        else:
            model_cat_est = global_cat_est            
        return model_cat_est[inverse]
    
    def predict_proba(self, X2):
        
        model_est = self.predict(X2)
        model_est = np.array([1-model_est, model_est]).T
        return model_est
    
class PlugInEstimator(BaseEstimator):
    
    """ Matching & averaging estimator. """
    
    def __init__(self, task):
        
        self.task = task
        return 
    
    def fit(self, X, y):
        
        task = self.task
        X_cat, counts = np.unique(X, axis=0, return_counts=True)
        self.counts_X1_cat = counts
        if task == 'quantile':
            m_list = [X == xc for xc in X_cat]
            y_cat = np.array([np.quantile(y[m[:,0]], q=0.75) for m in m_list])
        else:
            m_list = [(X == xc).all(axis=1) for xc in X_cat]
            y_cat = np.array([y[m].mean() for m in m_list]) 
        if task == 'salary': # salary
            self.D = {f'{int(x[0])}, {int(x[1])}': y for x,y in zip(X_cat, y_cat)}
            l1, l2 = npi.group_by(X[:,1]).mean(y) # global estimate for a given experience
            D_all = {f'all_{int(x)}': y for x,y in zip(l1, l2)}
            self.D.update(D_all)
        else: # quantile, sex
            self.D = {f'{int(x)}': y for x,y in zip(X_cat, y_cat)}
        self.D['all'] = np.sum(y_cat*counts)/np.sum(counts)
        return
    
    def predict(self, X, return_var_bool=False):
        
        X_cat, inverse = np.unique(X, axis=0, return_inverse=True)
        model_cat_est = np.zeros(len(X_cat))
        var_cat_bool = np.ones(len(X_cat))
        for k, x in enumerate(X_cat):
            try: # get plug-in estimate
                if X_cat.shape[1] == 2: # salary
                    model_cat_est[k] = self.D[f'{int(x[0])}, {int(x[1])}']
                else: # quantile, sex
                    model_cat_est[k] = self.D[f'{int(x)}']
            except KeyError: # get global estimate
                var_cat_bool[k] = 0
                try:
                    model_cat_est[k] = self.D[f'all_{int(x[1])}']
                except (IndexError, KeyError) as error:
                    model_cat_est[k] = self.D['all']
        if return_var_bool:
            return model_cat_est[inverse], var_cat_bool[inverse]
        return model_cat_est[inverse]
    
    def predict_proba(self, X, return_var_bool=False):
        
        if return_var_bool:
            model_est, var_bool = self.predict(X, return_var_bool=True)
            model_est = np.array([1-model_est, model_est]).T
            return model_est, var
        else:
            model_est = self.predict(X, return_var_bool=False)
            model_est = np.array([1-model_est, model_est]).T
        return model_est
    
class PlugInEstimator2(BaseEstimator):
    
    """ Matching & averaging with partial matching: when several variants
    are available, we use only one to compute the estimate. """

    def __init__(self, task, D_var):
        
        self.task = task
        self.D_var = D_var
        return 
    
    def fit(self, X, y):
        
        task = self.task
        if task == 'salary':
            X = X[:,:2]
        else:
            X = X[:,:1]
        X_cat, counts = np.unique(X, axis=0, return_counts=True)
        self.counts_X1_cat = counts
        if task == 'quantile':
            m_list = [X == xc for xc in X_cat]
            y_cat = np.array([np.quantile(y[m[:,0]], q=0.75) for m in m_list])
        else:
            m_list = [(X == xc).all(axis=1) for xc in X_cat]
            y_cat = np.array([y[m].mean() for m in m_list]) 
            
        if task == 'salary':
            self.D = {f'{int(x[0])}, {int(x[1])}': y for x,y in zip(X_cat, y_cat)}
        else: # quantile, sex
            self.D = {f'{int(x)}': y for x,y in zip(X_cat, y_cat)}
        self.D['all'] = -2
        return

    def predict(self, X):
        
        task = self.task
        if task == 'salary':
            X = X[:,2:]
        else:
            X = X[:,1:]
        X_cat, inverse = np.unique(X, axis=0, return_inverse=True)
        model_cat_est = np.zeros(len(X_cat))
        for k, x in enumerate(X_cat):
            if task == 'salary':
                x_vars = self.D_var[f'{int(x[0])}, {int(x[1])}']
                yk = []
                for xv in x_vars:
                    try:
                        yk.append(self.D[f'{int(xv[0])}, {int(xv[1])}'])
                    except:
                        pass
            else:
                x_vars = self.D_var[f'{int(x[0])}']
                yk = []
                for xv in x_vars:
                    try:
                        yk.append(self.D[f'{int(xv[0])}'])
                    except:
                        pass
            if yk != []:
                model_cat_est[k] = np.random.choice(yk, 1)[0]
            else:
                model_cat_est[k] = -1
        return model_cat_est[inverse]
    
    def predict_proba(self, X):
        
        model_est = self.predict(X)
        model_est = np.array([1-model_est, model_est]).T
        return model_est
    
def make_Xy_cat(X, y, task):
    
    X_cat, y_cat = npi.group_by(X).mean(y)
    if task == 'quantile':
        y_cat = npi.group_by(X).split(y)
        y_cat = np.array([np.quantile(yc, q=0.75) for yc in y_cat])
    return X_cat, y_cat