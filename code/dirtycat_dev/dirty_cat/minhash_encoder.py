import os
import sys
import numpy as np
import pandas as pd
import warnings

from scipy.special import logsumexp
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
    LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF, \
    TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.utils import murmurhash3_32, check_random_state

#from fasttext import load_model
#import category_encoders as cat_enc

from .fast_hash import ngram_min_hash
from .utils import LRUDict

class MinHashEncoder(BaseEstimator, TransformerMixin):
    """
    minhash method applied to ngram decomposition of strings

    Parameters
    ----------
    n_components : integer
        The number of dimension for each sample
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    hashing : {'fast', 'murmur'}, default=fast
        Hashing function. fast is faster but
        might have some concern with its entropy
    minmax_hash : boolean, default=False
        if True, return min hash and max hash concatenated
    """

    def __init__(self, n_components, ngram_range=(2, 4),
                 hashing='fast', minmax_hash=False):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.hashing = hashing
        self.minmax_hash = minmax_hash
        self.count = 0

    def get_unique_ngrams(self, string, ngram_range):
        """
        Return a list of different n-grams in a string
        """
        spaces = ' '  # * (n // 2 + n % 2)
        string = spaces + " ".join(string.lower().split()) + spaces
        ngram_set = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            string_list = [string[i:] for i in range(n)]
            ngram_set |= set(zip(*string_list))
        return ngram_set

    def minhash(self, string, n_components, ngram_range):
        min_hashes = np.ones(n_components) * np.infty
        grams = self.get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = self.get_unique_ngrams(' Na ', self.ngram_range)
        for gram in grams:
            hash_array = np.array([
                murmurhash3_32(''.join(gram), seed=d, positive=True)
                for d in range(n_components)])
            min_hashes = np.minimum(min_hashes, hash_array)
        return min_hashes/(2**32-1)

    def get_hash(self, string):
        if self.hashing == 'fast':
            if self.minmax_hash:
                assert self.n_components % 2 == 0,\
                       "n_components should be even when minmax_hash=True"
                return np.concatenate([ngram_min_hash(string, self.ngram_range,
                                                      seed, return_minmax=True)
                                      for seed in range(self.n_components//2)])
            else:
                return np.array([ngram_min_hash(string, self.ngram_range, seed)
                                for seed in range(self.n_components)])

        elif self.hashing == 'murmur':
            assert not(self.minmax_hash),\
                   "minmax_hash not implemented with murmur"
            return self.minhash(
                    string, n_components=self.n_components,
                    ngram_range=self.ngram_range)
        else:
            raise ValueError("hashing function must be 'fast' or"
                             "'murmur', got '{}'"
                             "".format(self.hashing))

    def reformat_input(self, X):
        
        """
        Check the input type of X and convert it to a 1D array, in order to
        be processed by the fit/transform methods.
        
        Parameters
        ----------
        
        X : (N, 1) or (N, ) shaped list or array. Or a pandas DataFrame
        or DataSeries.
        
        
        Returns
        -------
        
        A 1D array with the same values as X, with shape (N, ).
        
        """
        
        assert type(X) in [list, np.ndarray, pd.core.frame.DataFrame,
                           pd.core.series.Series], 'ERROR: input type {}\
                           not supported'.format(type(X))
        if isinstance(X, np.ndarray):
            return X.ravel() 
        elif isinstance(X, list):
            return np.array(X).ravel()
        else: # DataFrames or DataSeries
            return X.to_numpy().ravel()
         
    def fit(self, X, y=None):

        self.hash_dict = LRUDict(capacity=2**10)
        return self

    def transform(self, X):
        
        X = self.reformat_input(X)
        X_out = np.zeros((len(X), self.n_components))

        # TODO Parallel run here
        for i, x in enumerate(X):
            if x not in self.hash_dict:
                self.hash_dict[x] = self.get_hash(x)

        for i, x in enumerate(X):
            X_out[i, :] = self.hash_dict[x]

        return X_out