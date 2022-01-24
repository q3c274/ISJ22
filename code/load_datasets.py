import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import os
import re
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fclusterdata, ward, fcluster
from scipy.spatial.distance import pdist

"""
Code to load and preprocess data from the 14 databases considered in our
experiments.
We rely on a pretrained fastText model to embed job titles, and on a
Gamma-Poisson factorization from the dirty-cat package to embed employee
ethnicities.

"""

# --- Import functions from dirtycat packages

import sys
sys.path.append('dirtycat_dev/dirty_cat/')
from pretrained_fastText import PretrainedFastText
from gamma_poisson_factorization import OnlineGammaPoissonFactorization

def load_ft_model():
    
    """ Returns an english pretrained fastText model. """
    
    # --- Load fastText model
    assert os.path.isfile('fastText_bins/wiki.en.bin'),\
    "ERROR: fastText pretrained model 'wiki.en.bin' is not in folder 'code/fastText_bins'"
    bin_folder = 'fastText_bins/'
    ft_model = PretrainedFastText(n_components=1,
                                  bin_folder=bin_folder,
                                  language='english', load=True)
    return ft_model

def remove_NaN(df, columns):
    
    """
    Remove NaN values in a pandas DataFrame.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to clean.
    
    columns: dict
        A dictionary linking standardized column names like 'Job Title' to
        the column name in the dataframe ('JOB_TITLE', 'Job Position', ...).
        
    Returns
    -------
    df: pandas.DataFrame
        The cleaned dataframe.
        
    """
    
    for k, v in columns.items():
        mask = pd.notna(df[v])
        df = df[mask]
    return df

def remove_PT(df, k, v):
    
    """
    Remove part-time employees from the data.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to clean.
    
    k: str
        The name of the column indicating employee status like part-time or
        full-time.
    
    v : str
        The value used to mask full-time employees.
    
    Returns
    -------
    df: pandas.DataFrame
        The cleaned dataframe.
        
    """
    
    mask = df[k] == v
    df = df[mask]
    return df

def clean_job_titles(df, clean_jt, rmv_punct=True):
    
    """
    Clean and return job titles in a database. Cleaning consists in removing
    punctuation and setting job titles to lowercase.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to clean.
    
    clean_jt: bool
        Set clean_jt = True to perform cleaning (ponctuation + lowercase).
    
    rmv_punct: bool, default = True
        Set rmv_punct = True to remove punctuation while cleaning.
    
    Returns
    -------
    job_titles: np.ndarray, dtype = str
        The cleaned job titles.
        
    """
    
    job_titles = df['Job Title'].to_numpy(dtype=str)
    if clean_jt:
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        for k, jt in enumerate(job_titles):
            jt = jt.lower()
            if rmv_punct:
                job_titles[k] = re.sub("\s\s+" , " ", jt.translate(translator))
            else:
                job_titles[k] = re.sub("\s\s+" , " ", jt)
    return job_titles
        
def clean_df(df, columns, **kwargs):
    
    """
    Clean and return a database. NaN values and part-time workers are removed,
    column names are standardized.
    
    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to clean.
    
    columns: dict
        A dictionary linking standardized column names like 'Job Title' to
        the column name in the dataframe ('JOB_TITLE', 'Job Position', ...).
    
    remove_PT: bool, optional
        Set to True to remove part-time workers.
    
    kmask: str, optional
        If remove_PT = True, kmask is the column name used to mask full-time
        workers.
    
    kmask: str, optional
        If remove_PT = True, vmask is the value used to mask full-time workers.
        
    Returns
    -------
    df: pandas.DataFrame
        The cleaned dataframe.
        
    """
    
    df = remove_NaN(df, columns)
    if kwargs['remove_PT']:
        k, v = kwargs['kmask'], kwargs['vmask']
        df = remove_PT(df, k, v)
    L = [df[column] for column in columns.values()]
    df = pd.DataFrame(L, columns.keys()).T
    return df.reset_index()

###############################################################################

def load_austin_city(dir_path, clean_jt, return_df=False):
    
    """
    Load, clean and return database austin_city.csv.
    
    Parameters
    ----------
    dir_path: str
        Path to the folder containing the database.
    
    clean_jt: bool
        Set to True to clean the database.
    
    return_df: bool, default=False
        Set to True to return a dataframe instead of a dictionary.
    
    Returns
    -------
    D: dictionary
        If return_df=False, returns a dictionary with the following keys:
        'Annual Salary', 'Job Title', 'Hiring Date', 'Gender' and 'Ethnicity'.
        
    """
    
    # Initialize parameters
    name, ext = 'austin_city', '.csv'
    C = {'Annual Salary': 'Annual Salary', 'Job Title': 'Title',
         'Hiring Date': 'Date of Employment', 'Gender': 'Gender',
         'Ethnicity': 'Ethnicity'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Part Time/Full Time', vmask='Full Time')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'M').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_bexar_county(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database bexar_county.csv. """
    
    # Initialize parameters
    name, ext = 'bexar_county', '.csv'
    C = {'Annual Salary': 'Annual Salary', 'Job Title': 'Position Title',
         'Hiring Date': 'Hire Date', 'Gender': 'Gender',
         'Ethnicity': 'Race'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Fulltime/Part-Time', vmask='Fulltime')
    # Annual Salary
    D['Annual Salary'] = df['Annual Salary'].to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_dallas_city(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database dallas_city.csv. """
    
    # Initialize parameters
    name, ext = 'dallas_city', '.csv'
    C = {'Annual Salary': 'Rate of Pay',
         'Job Title': 'Job Code Description',
         'Hiring Date': 'Hire Date',
         'Gender': 'Gender', 'Ethnicity': 'Ethnicity',
         'Annual Hours': 'Annual Hours'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Status Description',
                  vmask='Active Full-time (Permanent)')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',', '.')
    y = y.to_numpy(dtype=float)
    mask = y < 300
    y[mask] = y[mask] * df['Annual Hours'][mask]
    D['Annual Salary'] = y
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'M').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_fort_worth_city(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database fort_worth_city.csv. """
    
    # Initialize parameters
    name, ext = 'fort_worth_city', '.csv'
    C = {'Annual Salary': 'Annual Rt', 'Job Title': 'Job',
         'Hiring Date': 'Last Start', 'Gender': 'Sex',
         'Ethnicity': 'Ethnic Grp'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Full/Part Time', vmask='Full-Time')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'M').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_houston_city(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database houston_city.csv. """
    
    # Initialize parameters
    name, ext = 'houston_city', '.csv'
    C = {'Annual Salary': 'Annual Salary',
         'Job Title': 'Job Title', 'Hiring Date': 'Hire Date',
         'Gender': 'Gender', 'Ethnicity': 'Racial Category'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Employee Grp', vmask='Full Time')
    mask = df['Annual Salary'] != '0'; df = df[mask]
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_san_antonio_city(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database san_antonio_city.csv. """
    
    # Initialize parameters
    name, ext = 'san_antonio_city', '.csv'
    C = {'Annual Salary': 'FY18 ANNUAL SALARY2',
         'Job Title': 'JOB TITLE', 'Hiring Date': 'HIRE DATE1',
         'Gender': 'GENDER', 'Ethnicity': 'ETHNIC ORIGIN10'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=False)
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'MALE').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_tarrant_county(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database tarrant_county.csv. """
    
    # Initialize parameters
    name, ext = 'tarrant_county', '.csv'
    C = {'Annual Salary': 'Budgeted Annual Salary',
         'Job Title': 'Name of Position',
         'Hiring Date': 'Hire Date',
         'Gender': 'Gender', 'Ethnicity': 'Race'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=False)
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_texas_state(dir_path, clean_jt, return_df=False):
    
    """ 
    Load, clean and return database texas_state.csv.
    This database is not used in the experiments.
    
    """
    
    # Initialize parameters
    name, ext = 'texas_state', '.csv'
    C = {'Annual Salary': 'ANNUAL',
         'Job Title': 'CLASS TITLE',
         'Hiring Date': 'EMPLOY DATE',
         'Gender': 'GENDER', 'Ethnicity': 'ETHNICITY'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='HRS PER WK', vmask=40)
    # Annual Salary
    D['Annual Salary'] = df['Annual Salary'].to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([(19-int(s[-2:]))%100 for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_am(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_am.csv. """
    
    # Initialize parameters
    name, ext = 'ut_am', '.csv'
    C = {'Annual Salary': 'Annual Budgeted Salary',
         'Job Title': 'Title Description',
         'Hiring Date': 'Curr Empl Date',
         'Gender': 'Gender', 'Ethnicity': 'Race-Ethnicity'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Part/Full Time',
                  vmask='F')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[-4:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_austin(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_austin.csv. """
    
    # Initialize parameters
    name, ext = 'ut_austin', '.csv'
    C = {'Annual Salary': 'Annual Salary Rate',
         'Job Title': 'Title',
         'Hiring Date': 'Hire Date',
         'Gender': 'Gender', 'Ethnicity': 'RaceEthinicity'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext, encoding='latin1')
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Fulltime or Parttime', vmask='Full-time')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_galveston(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_galveston.csv. """
    
    # Initialize parameters
    name, ext = 'ut_galveston', '.csv'
    C = {'Annual Salary': 'ANNUAL_PAY',
         'Job Title': 'JOBTITLE',
         'Hiring Date': 'LAST_HIRE_DT',
         'Gender': 'GENDER', 'Ethnicity': 'ETHNIC_GROUP_DESCR'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='FULL_PART_TIME', vmask='Full-time')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:10]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_houston(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_houston.csv. """
    
    # Initialize parameters
    name, ext = 'ut_houston', '.csv'
    C = {'Annual Salary': 'Annual Rt',
         'Job Title': 'Job Title',
         'Hiring Date': 'Orig Hire Date',
         'Gender': 'Sex', 'Ethnicity': 'Ethnic Grp'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Full/Part', vmask='F')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:10]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'M').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_houston_health(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_houston_health.csv. """
    
    # Initialize parameters
    name, ext = 'ut_houston_health', '.csv'
    C = {'Annual Salary': 'Total Base Comp',
         'Job Title': 'Job Title',
         'Hiring Date': 'Original Hire Date',
         'Gender': 'Gender', 'Ethnicity': 'Ethnicity - Consolidated'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Employment Status', vmask='F')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:10]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'M').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_md_anderson(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_md_anderson.csv. """
    
    # Initialize parameters
    name, ext = 'ut_md_anderson', '.csv'
    C = {'Annual Salary': 'Annual Rate',
         'Job Title': 'Job Title',
         'Hiring Date': 'First Start Date',
         'Gender': 'Gender', 'Ethnicity': 'Ethnic Grp'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext, encoding='latin1')
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Full/Part Time', vmask='Full-Time')
    # Annual Salary
    D['Annual Salary'] = df['Annual Salary'].to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:10]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_sa_health(dir_path, clean_jt, return_df=False):
    
    """ Load, clean and return database ut_sa_health.csv. """
    
    # Initialize parameters
    name, ext = 'ut_sa_health', '.csv'
    C = {'Annual Salary': 'Comp Rate',
         'Job Title': 'Job Title',
         'Hiring Date': 'Rehire Dt',
         'Gender': 'Sex', 'Ethnicity': 'Ethnic Grp'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext, encoding='latin1')
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Full/Part', vmask='Full-Time')
    # Annual Salary
    y = df['Annual Salary'].str.replace(',','.')
    D['Annual Salary'] = y.to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:10]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Ethnicity
    D['Ethnicity'] = df['Ethnicity'].to_numpy(dtype=str)
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def load_ut_sw_medical(dir_path, clean_jt, return_df=False):
    
    """
    Load, clean and return database ut_sw_medical.csv.
    This database is not used in the experiments, since the Ethnicity feature
    is encoded with different columns and booleans.
    
    """
    
    # Initialize parameters
    name, ext = 'ut_sw_medical', '.csv'
    C = {'Annual Salary': 'Annual Pay',
         'Job Title': 'Job Code Description',
         'Hiring Date': 'Date First Hire',
         'Gender': 'Gender'}
    D = {}
    # Load dataframe + basic cleaning
    df = pd.read_csv(dir_path+name+ext)
    df = clean_df(df, columns=C, remove_PT=True,
                  kmask='Full Time Type Description',
                  vmask='Full-Time')
    # Annual Salary
    D['Annual Salary'] = df['Annual Salary'].to_numpy(dtype=float)
    # Job Title
    D['Job Title'] = clean_job_titles(df, clean_jt)
    # Seniority
    D['Seniority'] = np.array([2019-int(s[6:10]) for s in df['Hiring Date']])
    # Gender
    D['Gender'] = np.array((df['Gender'] == 'Male').astype(int))
    # Return D or df
    if return_df:
        df["Annual Salary"] = D['Annual Salary']
        return df
    return D

###############################################################################

def entity_encoding(datasets):
    
    """
    Add 'Entity_Encoding' feature to the datasets. This feature is one-encoded
    from the following categories: city, county, university,
    university_hospital.
    
    Parameters
    ----------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees features.
    
    """

    cities = ['austin_city', 'fort_worth_city', 'houston_city',
              'san_antonio_city', 'dallas_city']
    counties = ['bexar_county', 'tarrant_county']
    universities = ['ut_am', 'ut_houston', 'ut_austin']
    university_hospitals = ['ut_galveston', 'ut_md_anderson',
                            'ut_sa_health', 'ut_houston_health',
                            'ut_sw_medical']
    for name in datasets.keys():
        n_samples = len(datasets[name]['Annual Salary'])
        x = np.zeros((n_samples, 4))
        idx = [name in L for L in [cities, counties, universities,
                                  university_hospitals]].index(True)
        x[:,idx] = 1
        datasets[name]['Entity_Encoding'] = x
    return

def ethnicity_encoding(datasets):
    
    """
    Add 'Ethnicity_Encoding' feature to the datasets. This feature is encoded
    with dirtycat Gamma-Poison factorization, trained on all ethnicities string
    values (dirty categorical variable: 'Black','BLK','Black of Afro-American).
    This encoding factorizes n-grams counts into a weighted sum of n=10 topics.
    Topics activations are used for encoding.
    
    Parameters
    ----------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees features.
    
    """
    
    ethnicities = np.hstack([df['Ethnicity'] for df in datasets.values()])
    np.random.shuffle(ethnicities)
    gamma_poisson = OnlineGammaPoissonFactorization(n_topics=10, r=None)
    gamma_poisson.fit(ethnicities)
    for df in datasets.values():
        df['Ethnicity_Encoding'] = gamma_poisson.transform(df['Ethnicity'])
    return

def job_title_encoding(datasets):
    
    """
    Add 'Job_Title_Encoding' feature to the datasets. Job titles are embedded
    with a pretrained fastText model.
    
    Parameters
    ----------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees features.
    
    """
    
    ft_model= load_ft_model()
    for df in datasets.values():
        df['Job_Title_Encoding'] = ft_model.fit_transform(df['Job Title'])
    del ft_model
    return

###############################################################################

def load_all(dir_path, clean_jt, exceptions=[]):
    
    """
    Load and clean all databases, and encode features.
    
    Parameters
    ----------
    dir_path:
        Path to the folder containing the databases.
    
    clean_jt:
        Set to True to clean job titles.
    
    exceptions: list of strings
        Names of the databases to exclude.
    
    Returns
    -------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees encoded features.
    
    """
    
    names = ['austin_city', 'bexar_county', 'dallas_city',
             'fort_worth_city', 'houston_city',
             'san_antonio_city', 'tarrant_county',
             'ut_am', 'ut_austin', 'ut_galveston',
             'ut_houston', 'ut_houston_health',
             'ut_md_anderson', 'ut_sa_health']
    
    loaders = [load_austin_city, load_bexar_county, load_dallas_city,
               load_fort_worth_city, load_houston_city,
               load_san_antonio_city, load_tarrant_county,
               load_ut_am, load_ut_austin, load_ut_galveston,
               load_ut_houston, load_ut_houston_health, 
               load_ut_md_anderson, load_ut_sa_health]
    
    for exc in exceptions:
        idx = names.index(exc)
        names.remove(exc)
        loaders.pop(idx)
    
    # Load data    
    datasets = {}
    for name, loader in zip(names,loaders):
        datasets[name] = loader(dir_path, clean_jt)
        print(name)
    # Job title encoding
    job_title_encoding(datasets)
    # Ethnicity encoding
    ethnicity_encoding(datasets)
    # Entity encoding
    entity_encoding(datasets)
    return datasets

def load_all_raw(dir_path, clean_jt, exceptions=[]):
    
    """
    Same as load_all, but doesn't encode entity, ethnicity and job titles.
    
    """
    
    names = ['austin_city', 'bexar_county', 'dallas_city',
             'fort_worth_city', 'houston_city',
             'san_antonio_city', 'tarrant_county',
             'ut_am', 'ut_austin', 'ut_galveston',
             'ut_houston', 'ut_houston_health',
             'ut_md_anderson', 'ut_sa_health']
    
    loaders = [load_austin_city, load_bexar_county, load_dallas_city,
               load_fort_worth_city, load_houston_city,
               load_san_antonio_city, load_tarrant_county,
               load_ut_am, load_ut_austin, load_ut_galveston,
               load_ut_houston, load_ut_houston_health, 
               load_ut_md_anderson, load_ut_sa_health]
    
    for exc in exceptions:
        idx = names.index(exc)
        names.remove(exc)
        loaders.pop(idx)
    
    # Load data    
    datasets = {}
    for name, loader in zip(names,loaders):
        datasets[name] = loader(dir_path, clean_jt)
        print(name)
    # Job title encoding
    #job_title_encoding(datasets)
    # Ethnicity encoding
    #ethnicity_encoding(datasets)
    # Entity encoding
    #entity_encoding(datasets)
    return datasets

###############################################################################

def make_merged_data(datasets, fname):
    
    """
    Merge and save all datasets into a dataframe, and add a 'Database_ID'
    column to distingush groups.
    
    Parameters
    ----------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees features.
    
    fname: str
        The name of the file in which the merged datasets are stored.
        Default directory is '../datasets/'.
        
    """
    
    D = {}
    for key in ['Job Title', 'Seniority', 'Gender',
                'Ethnicity', 'Annual Salary']:
        D[key] = np.hstack([DB[key] for DB in datasets.values()])
    D["Database_ID"] = np.hstack([[idx]*len(DB['Gender']) for 
                               idx, DB in enumerate(datasets.values())])
    df = pd.DataFrame(D)
    df.to_csv(f'../datasets/{fname}.csv', index=False)
    return

def load_cleaned_embeddings(Xg, Xs, fname):
    
    """
    Return features arrays where fastText embeddings are replaced with cleaned
    embeddings from the merged database with manual cleaning and entity
    matching.
    
    Parameters
    ----------
    Xg, Xs: np.ndarrays (N_samples, N_features1), (N_samples, N_features2)
        Features arrays for sex classification and salary/quantile regression
        tasks.
    
    fname: str
        Name of the file with merged, cleaned and matched data.
    
    Returns
    -------
    Xg_cft, Xs_cft: np.ndarrays (N_samples, N_features1), (N_samples,
                    N_features2)
        Features arrays with manual cleaning and entity matching for sex
        classification and salary/quantile regression tasks.
    
    clean_jt: np.ndarray (N_samples,)
        Cleaned and matched job titles. Punctuation is also removed, string are
        lowercased, multiple whitespaces and whitespaces at the beginning/end
        of string are stripped.
        
    """
    
    ft_model = load_ft_model()
    df = pd.read_csv(fname)
    cjt = df['Cleaned Job Title'].to_numpy(str)
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    cjt = np.array([jt.translate(translator).lower() for jt in cjt])
    cjt = [re.sub("\s\s+" , " ", jt) for jt in cjt]
    cjt = [re.sub("^\s" , "", jt) for jt in cjt]
    cjt = np.array([re.sub("\s$" , "", jt) for jt in cjt])
    ft_cjt = ft_model.fit_transform(cjt)
    Xg_cft, Xs_cft = Xg.copy(), Xs.copy()
    Xg_cft[:,:300], Xs_cft[:,:300] = ft_cjt, ft_cjt
    
    # Merge similar word permutations
    ft_sum = ft_cjt.sum(axis=1)
    n1, n2 = len(np.unique(ft_sum)), len(np.unique(ft_cjt, axis=0))
    #print(n1, n2)
    assert n1 == n2, "SUM IS NOT INJECTIVE"
    D1 = {}
    for jt, ft in zip(cjt, ft_sum):
        D1[ft] = jt
    clean_jt = cjt.copy()
    for k, ft in enumerate(ft_sum):
        clean_jt[k] = D1[ft]
        
    return Xg_cft, Xs_cft, clean_jt
    
def make_dataset_salary(datasets, keep, exceptions, return_list=False):
    
    """
    Return features and target arrays for salary/quantile regression.
    Also returns the groups to whom employees belong.
    
    Parameters
    ----------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees encoded features.
        
    keep: list of strings
        Name of the datasets to keep.
    
    exceptions: list of strings
        Name of the datasets to exclude.
    
    return_list: bool, default=False
        If True, return lists of features and target arrays instead of merged
        arrays.
    
    Returns
    -------
    Xs, salary, groups or Xs_list, ys_list.
    
    """
    
    X, y = [], []
    groups = []
    for k, (name, df) in enumerate(datasets.items()):
        if (name not in exceptions) and (name in keep):
            y.append(df['Annual Salary'])
            x = np.hstack([df['Job_Title_Encoding'],
                           df['Seniority'][:, np.newaxis],
                           df['Gender'][:, np.newaxis],
                           df['Entity_Encoding'],
                           df['Ethnicity_Encoding']])
            X.append(x)
            groups.append([k]*len(df['Annual Salary']))
    if return_list:
        return np.array(X), np.array(y)        
    return np.vstack(X), np.hstack(y), np.hstack(groups) 

def make_dataset_gender(datasets, keep, exceptions, return_list=False):
    
    """
    Return features and target arrays for sex classification.
    Also returns the "dirty" job titles.
    
    Parameters
    ----------
    datasets: dict
        Dictionary whose keys are the names of the databases, and values are
        dictionaries with employees encoded features.
        
    keep: list of strings
        Name of the datasets to keep.
    
    exceptions: list of strings
        Name of the datasets to exclude.
    
    return_list: bool, default=False
        If True, return lists of features and target arrays instead of merged
        arrays.
    
    Returns
    -------
    Xg, gender, dirty_jt or Xs_list, ys_list.
    
    """
    
    X, y, jt = [], [], []
    for name, df in datasets.items():
        if (name not in exceptions) and (name in keep):
            y.append(df['Gender'])
            jt.append(df['Job Title'])
            x = np.hstack([df['Job_Title_Encoding'],
                           df['Seniority'][:, np.newaxis],
                           df['Entity_Encoding'],
                           df['Ethnicity_Encoding']])
            X.append(x)
    if return_list:
        return np.array(X), np.array(y), np.array(jt)
    return np.vstack(X), np.hstack(y), np.hstack(jt)

def quick_entity_matching(Xg, Xs, n_clusters, inverse, method='ward', **kwargs):
    
    """
    Return features arrays where job titles are clustered based on the cosine
    similarity of their fastText embeddings. Clusters are then labeled with
    integers. Labels are shuffled to break the hierarchy created in the
    clustering procedure (similar labels are similar job titles). This allows
    to emulate a naive/quick entity matching procedure and compare it to no or
    manual entity matching.
    
    Parameters
    ----------
    Xg, Xs: np.ndarrays
        Features arrays with fastText embeddings.
    
    n_clusters: int
        Number of clusters to form.
    
    inverse:
        The indices to reconstruct the original FT array from the unique
        FT array. Obtained with np.unique(FT, return_inverse=True), where FT
        is Xs[:, :300].
    
    method: string
        Set to 'ward' for ward clustering, and 'kmeans' for kmeans clustering.
        Ward clustering is faster and works better than kmeans for large
        n_clusters.
    
    unq_FT: np.ndarray, optional
        If method = 'kmeans', unq_FT, inverse = np.unique(FT,
        return_inverse=True).
        
    Z_ward: np.ndarray, optional
        The Z parameter of scipy fcluster. Obtained with 
        Z_ward = ward(pdist(unq_FT, metric="cosine")).
    
    Returns
    -------
    Xg_qem, Xs_qem: np.ndarrays
        The features arrays where job titles are clustered and labeled.
    
    """
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=10)
        kmeans.fit(normalize(kwargs['unq_FT'], axis=1))
        labels = kmeans.labels_
    elif method == 'ward':
        labels = fcluster(kwargs['Z_ward'], t=n_clusters, criterion='maxclust')
    unq_l, inv_l = np.unique(labels, return_inverse=True)
    np.random.shuffle(unq_l)
    #print(labels)
    labels = unq_l[inv_l] # Random shuffling of labels to break hierarchy.
    #print(labels)
    N_samples, Ng_features = Xg.shape
    _, Ns_features = Xs.shape
    Xg_qem = np.zeros((N_samples, Ng_features-300+1))
    Xs_qem = np.zeros((N_samples, Ns_features-300+1))
    Xg_qem[:,0] = labels[inverse]
    Xg_qem[:,1:] = Xg[:,300:]
    Xs_qem[:,0] = labels[inverse]
    Xs_qem[:,1:] = Xs[:,300:]
    return Xg_qem, Xs_qem

def no_entity_matching(Xg, Xs, inverse):
    
    """
    Return features arrays where job titles are labeled with a unique integer.
    
    Parameters
    ----------
    Xg, Xs: np.ndarrays
        Features arrays with fastText embeddings.
    
    inverse: np.ndarray
        The indices to reconstruct the original dirty_jt array from the unique
        dirty_jt array. Obtained with np.unique(dirty_jt, return_inverse=True).
    
    Returns
    -------
    Xg_nem, Xs_nem: np.ndarrays
        The features arrays where dirty job titles are encoded with a 1D array
        of integers.
    
    """
    
    N_samples, Ng_features = Xg.shape
    _, Ns_features = Xs.shape
    Xg_nem = np.zeros((N_samples, Ng_features-300+1))
    Xs_nem = np.zeros((N_samples, Ns_features-300+1))
    unq_l, inv_l = np.unique(inverse, return_inverse=True)
    np.random.shuffle(unq_l)
    #print(inverse)
    inverse = unq_l[inv_l]
    #print(inverse)
    Xg_nem[:,0] = inverse+1
    Xg_nem[:,1:] = Xg[:,300:]
    Xs_nem[:,0] = inverse+1
    Xs_nem[:,1:] = Xs[:,300:]
    return Xg_nem, Xs_nem

def manual_entity_matching(Xg_cft, Xs_cft, inverse_clean_jt):
    
    """
    Return features arrays where job titles are labeled with a unique integer.
    Job titles are cleaned and matched beforehand.
    
    Parameters
    ----------
    Xg_cft, Xs_cft: np.ndarrays
        Features arrays with clean fastText embeddings.
    
    inverse_clean_jt: np.ndarray
        The indices to reconstruct the original clean_jt array from the unique
        clean_jt array. Obtained with np.unique(clean_jt, return_inverse=True).
    
    Returns
    -------
    Xg_mem, Xs_mem: np.ndarrays
        The features arrays where clean job titles are encoded with a 1D array
        of integers.
    
    """
    
    return no_entity_matching(Xg_cft, Xs_cft, inverse_clean_jt)

def list_variants(unq_clean_jt, clean_jt, dirty_jt):
    
    """
    Return morphological variants of the same entity, as well as the number of
    variants per entity.
    
    Parameters
    ----------
    Xg_cft, Xs_cft: np.ndarrays
        Features arrays with clean fastText embeddings.
    
    clean_jt: np.ndarray (N_samples, )
        Manually cleaned and matched job titles.
    
    dirty_jt: np.ndarray (N_samples, )
        Raw job titles.
        
    Returns
    -------
    variants: np.ndarray (N_entity, )
        An array where each row is a list with the morphological variants of
        a certain entity (job position).
    
    n_variants: np.ndarray (N_entity, )
        The number of morphological variants per entity.
        
    """
    
    n1 = len(unq_clean_jt)
    variants, n_variants = np.empty(n1, dtype=object), np.empty(n1, dtype=int)
    for k, ujt in enumerate(unq_clean_jt):
        if k%1000 == 0:
            print(f'# --- {k}')
        mask = (clean_jt == ujt)
        variants[k] = vjt = np.unique(dirty_jt[mask])
        n_variants[k] = len(vjt)
    return np.array(variants), np.array(n_variants)