import numpy as np
import pandas as pd

"""
Code to reproduce the results of table 3 in the paper, i.e. cross-validated
errors for salary, quantile and propensity-score estimation.
We compare matching & averaging, fuzzy matching (on raw and manually matched
data) and machine-learning with embeddings (on raw and manually matched data).

"""

dir_name = '../results/cv_errors/'

### Salary estimation

fnames = ['salary_plug_in.csv', 'salary_fuzzy_raw.csv',
          'salary_fuzzy_matched.csv', 'salary_embeddings_raw.csv',
          'salary_embeddings_matched.csv']
methods = ['matching & averaging', 'fuzzy matching, raw data',
           'fuzzy matching, manually matched data',
           'embedding & learning, raw data',
           'embedding & learning, manually matched data']

print('SALARY ESTIMATION')
for method, fname in zip(methods, fnames):
    cv_err = pd.read_csv(dir_name + fname)['cv_error'].to_numpy()
    rsme = np.mean(cv_err**2)**0.5
    print(method, int(round(rsme)))
print('-'*50)

### Quantile estimation

fnames = ['quantile_plug_in.csv', 'quantile_fuzzy_raw.csv',
          'quantile_fuzzy_matched.csv', 'quantile_embeddings_raw.csv',
          'quantile_embeddings_matched.csv']
methods = ['matching & averaging', 'fuzzy matching, raw data',
           'fuzzy matching, manually matched data',
           'embedding & learning, raw data',
           'embedding & learning, manually matched data']

print('0.75-QUANTILE ESTIMATION')
for method, fname in zip(methods, fnames):
    cv_err = pd.read_csv(dir_name + fname)['cv_error'].to_numpy()
    mae = np.mean(cv_err)
    print(method, int(round(mae)))
print('-'*50)
    
### Propensity estimation

fnames = ['sex_plug_in.csv', 'sex_fuzzy_raw.csv',
          'sex_fuzzy_matched.csv', 'sex_embeddings_raw.csv',
          'sex_embeddings_matched.csv']
methods = ['matching & averaging', 'fuzzy matching, raw data',
           'fuzzy matching, manually matched data',
           'embedding & learning, raw data',
           'embedding & learning, manually matched data']

print('PROPENSITY-SCORE ESTIMATION')
for method, fname in zip(methods, fnames):
    cv_err = pd.read_csv(dir_name + fname)['cv_error'].to_numpy()
    brier_score = np.mean(cv_err**2)
    print(method, round(brier_score*1000)/1000)