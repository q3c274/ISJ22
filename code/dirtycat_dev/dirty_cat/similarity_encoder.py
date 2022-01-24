import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from . import string_distances
from .string_distances import get_ngram_count, preprocess


def _ngram_similarity_one_sample_inplace(
        x_count_vector, vocabulary_count_matrix, str_x,
        vocabulary_ngram_counts, se_dict, unq_X, i, ngram_range):
    """Update inplace a dict of similarities between a string and a vocabulary


    Parameters
    ----------
    x_count_vector: np.array
        count vector of the sample based on the ngrams of the vocabulary
    vocabulary_count_matrix: np.array
        count vector of the vocabulary based on its ngrams
    str_x: str
        the actual sample string
    vocabulary_ngram_counts: np.array
        number of ngrams for each unique element of the vocabulary
    se_dict: dict
        dictionary containing the similarities for each x in unq_X
    unq_X: np.array
        the arrayes of all unique samples
    i: str
        the index of x_count_vector in the csr count matrix
    ngram_range: tuple

    """
    nonzero_idx = x_count_vector.indices
    nonzero_vals = x_count_vector.data

    samegrams = np.asarray(
        (vocabulary_count_matrix[:, nonzero_idx].minimum(nonzero_vals)).sum(
            axis=1))

    allgrams = get_ngram_count(
        str_x, ngram_range) + vocabulary_ngram_counts - samegrams
    similarity = np.divide(samegrams, allgrams)
    se_dict[unq_X[i]] = similarity.reshape(-1)


def ngram_similarity(X, cats, ngram_range, hashing_dim, dtype=np.float64):
    """
    Similarity encoding for dirty categorical variables:
        Given to arrays of strings, returns the
        similarity encoding matrix of size
        len(X) x len(cats)

    ngram_sim(s_i, s_j) =
        ||min(ci, cj)||_1 / (||ci||_1 + ||cj||_1 - ||min(ci, cj)||_1)
    """
    min_n, max_n = ngram_range
    unq_X = np.unique(X)
    cats = np.array([' %s ' % cat for cat in cats])
    unq_X_ = np.array([' %s ' % x for x in unq_X])
    if not hashing_dim:
        vectorizer = CountVectorizer(analyzer='char',
                                     ngram_range=(min_n, max_n),
                                     dtype=dtype)
        vectorizer.fit(np.concatenate((cats, unq_X_)))
    else:
        vectorizer = HashingVectorizer(analyzer='char',
                                       ngram_range=(min_n, max_n),
                                       n_features=hashing_dim, norm=None,
                                       alternate_sign=False,
                                       dtype=dtype)
        # The hashing vectorizer is stateless. We don't need to fit it on the data
        vectorizer.fit(X)
    count_cats = vectorizer.transform(cats)
    count_X = vectorizer.transform(unq_X_)
    # We don't need the vectorizer anymore, delete it to save memory
    del vectorizer
    sum_cats = np.asarray(count_cats.sum(axis=1))
    SE_dict = {}

    for i, x in enumerate(count_X):
        _, nonzero_idx, nonzero_vals = sparse.find(x)
        samegrams = np.asarray((count_cats[:, nonzero_idx].minimum(nonzero_vals)
                                ).sum(axis=1))
        allgrams = x.sum() + sum_cats - samegrams
        similarity = np.divide(samegrams, allgrams)
        SE_dict[unq_X[i]] = similarity.reshape(-1)
    # We don't need the counts anymore, delete them to save memory
    del count_cats, count_X

    out = np.empty((len(X), similarity.size), dtype=dtype)
    for x, out_row in zip(X, out):
        out_row[:] = SE_dict[x]

    return np.nan_to_num(out, copy=False)


def get_prototype_frequencies(prototypes):
    """
    Computes the frequencies of the values contained in prototypes
    Reverse sorts the array by the frequency
    Returns a numpy array of the values without their frequencies
    """
    uniques, counts = np.unique(prototypes, return_counts=True)
    sorted_indexes = np.argsort(counts)[::-1]
    return uniques[sorted_indexes], counts[sorted_indexes]


def get_kmeans_prototypes(X, n_prototypes, hashing_dim=128,
                          ngram_range=(3, 3), sparse=False, sample_weight=None, random_state=None):
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = HashingVectorizer(analyzer='char', norm=None,
                                   alternate_sign=False,
                                   ngram_range=ngram_range,
                                   n_features=hashing_dim)
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    if indexes_prototypes.shape[0] < n_prototypes:
        warnings.warn('Final number of unique prototypes is lower than ' +
                      'n_prototypes (expected)')
    return np.sort(X[indexes_prototypes])


_VECTORIZED_EDIT_DISTANCES = {
    'levenshtein-ratio': np.vectorize(string_distances.levenshtein_ratio),
    'jaro': np.vectorize(string_distances.jaro),
    'jaro-winkler': np.vectorize(string_distances.jaro_winkler),
}


class SimilarityEncoder(OneHotEncoder):
    """Encode string categorical features as a numeric array.

    The input to this transformer should be an array-like of
    strings.
    The method is based on calculating the morphological similarities
    between the categories.
    The categories can be encoded using one of the implemented string
    similarities: ``similarity='ngram'`` (default), 'levenshtein-ratio',
    'jaro', or 'jaro-winkler'.
    This encoding is an alternative to OneHotEncoder in the case of
    dirty categorical variables.

    Parameters
    ----------
    similarity : str {'ngram', 'levenshtein-ratio', 'jaro', or\
'jaro-winkler'}
        The type of pairwise string similarity to use.

    ngram_range : tuple (min_n, max_n), default=(2, 4)
        Only significant for ``similarity='ngram'``. The range of
        values for the n_gram similarity.

    categories : 'auto', 'k-means', 'most_frequent' or a list of lists/arrays
    of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the i-th
          column. The passed categories must be sorted and should not mix
          strings and numeric values.
        - 'most_frequent' : Computes the most frequent values for every
           categorical variable
        - 'k-means' : Computes the K nearest neighbors of K-mean centroids
           in order to choose the prototype categories

        The categories used can be found in the ``categories_`` attribute.
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' or 'ignore' (default)
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to ignore). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.
    hashing_dim : int type or None.
        If None, the base vectorizer is CountVectorizer, else it's set to
        HashingVectorizer with a number of features equal to `hashing_dim`.
    n_prototypes: number of prototype we want to use.
        Useful when `most_frequent` or `k-means` is used.
        Must be a positive non null integer.
    random_state: either an int used as a seed, a RandomState instance or None.
        Useful when `k-means` strategy is used.
    n_jobs: int, optional
        maximum number of processes used to compute similarity matrices. Used
        only if ``fast=True`` in ``SimilarityEncoder.transform``

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).

    References
    ----------

    For a detailed description of the method, see
    `Similarity encoding for learning with dirty categorical variables
    <https://hal.inria.fr/hal-01806175>`_ by Cerda, Varoquaux, Kegl. 2018
    (accepted for publication at: Machine Learning journal, Springer).


    """

    def __init__(self, similarity='ngram', ngram_range=(2, 4),
                 categories='auto', dtype=np.float64,
                 handle_unknown='ignore', hashing_dim=None, n_prototypes=None,
                 random_state=None, n_jobs=None):
        super().__init__()
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.similarity = similarity
        self.ngram_range = ngram_range
        self.hashing_dim = hashing_dim
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self.n_jobs = n_jobs

        if not isinstance(categories, list):
            assert categories in [None, 'auto', 'k-means', 'most_frequent']
        if categories in ['k-means', 'most_frequent'] and (n_prototypes is None or n_prototypes == 0):
            raise ValueError('n_prototypes expected None or a positive non null integer')
        if categories == 'auto' and n_prototypes is not None:
            warnings.warn('n_prototypes parameter ignored with category type \'auto\'')

    def get_most_frequent(self, prototypes):
        """ Get the most frequent category prototypes
        Parameters
        ----------
        prototypes : the list of values for a category variable
        Returns
        -------
        The n_prototypes most frequent values for a category variable
        """
        values, _ = get_prototype_frequencies(prototypes)
        return values[:self.n_prototypes]
    
    def reformat_input(self, X):
        
        """
        Check the input type of X and convert it to a (N, 1) array, in order to
        be processed by the fit/transform methods.
        
        Parameters
        ----------
        
        X : (N, 1) or (N, ) shaped list or array. Or a pandas DataFrame
        or DataSeries.
        
        
        Returns
        -------
        
        A 1D array with the same values as X, with shape (N, 1).
        
        """
        
        assert type(X) in [list, np.ndarray, pd.core.frame.DataFrame,
                           pd.core.series.Series], 'ERROR: input type {}\
                           not supported'.format(type(X))
        if isinstance(X, np.ndarray):
            return X.reshape(-1, 1)
        elif isinstance(X, list):
            return np.array(X).reshape(-1, 1)
        else: # DataFrames or DataSeries
            return X.to_numpy().reshape(-1, 1)
        
    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        
        X = self.reformat_input(X)
        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if ((self.hashing_dim is not None) and
                (not isinstance(self.hashing_dim, int))):
            raise ValueError("value '%r' was specified for hashing_dim, "
                             "which has invalid type, expected None or "
                             "int." % self.hashing_dim)

        if self.categories not in ['auto', 'most_frequent', 'k-means']:
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        n_samples, n_features = X.shape
        self.categories_ = list()
        self.random_state_ = check_random_state(self.random_state)
  
        for i in range(n_features):
            Xi = X[:, i]
            if self.categories == 'auto':
                self.categories_.append(np.unique(Xi))
            elif self.categories == 'most_frequent':
                self.categories_.append(self.get_most_frequent(Xi))
            elif self.categories == 'k-means':
                uniques, count = np.unique(Xi, return_counts=True)
                self.categories_.append(
                    get_kmeans_prototypes(uniques, self.n_prototypes, sample_weight=count,
                                          random_state=self.random_state_))
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                self.categories_.append(np.array(self.categories[i], dtype=object))

        if self.similarity == 'ngram':
            self.vectorizers_ = []
            self.vocabulary_count_matrices_ = []
            self.vocabulary_ngram_counts_ = []

            for i in range(n_features):
                vectorizer = CountVectorizer(
                    analyzer='char', ngram_range=self.ngram_range,
                    dtype=self.dtype, strip_accents=None)

                # Store the raw-categories (and not the preprocessed
                # categories) but use the preprocessed categories to compute
                # the stored count_matrices. This done to preserve the
                # equivalency between the user input and the categories_
                # attribute of the SimilarityEncoder, while being compliant
                # with the CountVectorizer preprocessing steps.
                preprocessed_categories = np.array(list(map(
                    preprocess, self.categories_[i])), dtype=object)

                vocabulary_count_matrix = vectorizer.fit_transform(
                    preprocessed_categories)

                vocabulary_ngram_count = list(map(lambda x: get_ngram_count(
                    preprocess(x), self.ngram_range), self.categories_[i]))

            self.vectorizers_.append(vectorizer)
            self.vocabulary_count_matrices_.append(vocabulary_count_matrix)
            self.vocabulary_ngram_counts_.append(vocabulary_ngram_count)

        return self

    def transform(self, X, fast=True):
        """Transform X using specified encoding scheme.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_new : 2-d array, shape [n_samples, n_features_new]
            Transformed input.

        """
        
        X = self.reformat_input(X)
        n_samples, n_features = X.shape

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)

        if self.similarity in ('levenshtein-ratio',
                               'jaro',
                               'jaro-winkler'):
            out = []
            vect = _VECTORIZED_EDIT_DISTANCES[self.similarity]
            for j, cats in enumerate(self.categories_):
                unqX = np.unique(X[:, j])
                encoder_dict = {x: vect(x, cats.reshape(1, -1))
                                for x in unqX}
                encoder = [encoder_dict[x] for x in X[:, j]]
                encoder = np.vstack(encoder)
                out.append(encoder)
            return np.hstack(out)

        elif self.similarity == 'ngram':
            min_n, max_n = self.ngram_range

            total_length = sum(len(x) for x in self.categories_)
            out = np.empty((len(X), total_length), dtype=self.dtype)
            last = 0
            for j, cats in enumerate(self.categories_):
                if fast:
                    encoded_Xj = self._ngram_similarity_fast(X[:, j], j)
                else:
                    encoded_Xj = ngram_similarity(
                        X[:, j], cats, ngram_range=(min_n, max_n),
                        hashing_dim=self.hashing_dim, dtype=np.float32)

                out[:, last:last + len(cats)] = encoded_Xj
                last += len(cats)
            return out
        else:
            raise ValueError("Unknown similarity: '%s'" % self.similarity)

    def _ngram_similarity_fast(self, X, col_idx):
        """
        Fast computation of ngram similarity, for SimilarityEncoder.


        SimilarityEncoder.transform uses the count vectors of the vocabulary in
        its computations. In ngram_similarity, these count vectors have to be
        re-computed each time, which can slow down the execution. In this
        method, the count vectors are recovered from the
        ``vocabulary_count_matrices`` attribute of the SimilarityEncoder,
        speeding up the execution.
        Parameters
        ----------
        X: np.array, list
            observations being transformed.
        col_idx: int
            the column index of X in the original feature matrix.
        """
        min_n, max_n = self.ngram_range
        vectorizer = self.vectorizers_[col_idx]

        unq_X = np.unique(X)
        unq_X_ = np.array(list(map(preprocess, unq_X)))

        X_count_matrix = vectorizer.transform(unq_X_)
        vocabulary_count_matrix = self.vocabulary_count_matrices_[col_idx]
        vocabulary_ngram_count = np.array(
            self.vocabulary_ngram_counts_[col_idx]).reshape(-1, 1)

        se_dict = {}

        Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(
            _ngram_similarity_one_sample_inplace)(
            X_count_vector, vocabulary_count_matrix, x_str,
            vocabulary_ngram_count, se_dict, unq_X, i, self.ngram_range) for
            X_count_vector, x_str, i in zip(
                X_count_matrix, unq_X_, range(len(unq_X))))

        out = np.empty(
            (len(X), vocabulary_count_matrix.shape[0]), dtype=self.dtype)

        for x, out_row in zip(X, out):
            out_row[:] = se_dict[x]

        return np.nan_to_num(out, copy=False)
