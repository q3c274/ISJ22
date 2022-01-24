import os
import numpy as np
from fasttext import load_model
from sklearn.base import BaseEstimator, TransformerMixin
import time


class PretrainedFastText(BaseEstimator, TransformerMixin):
    
    """
    Category embedding using a fastText pretrained model (downloadable here:
    https://fasttext.cc/docs/en/crawl-vectors.html)
    
    """

    def __init__(self, n_components, language='english', bin_folder='', load=False):
        
        self.n_components = n_components
        self.language = language
        self.bin_folder = bin_folder
        self.path_dict = dict(
            english='wiki.en.bin',
            french='cc.fr.300.bin',
            hungarian='cc.hu.300.bin')
        
        if self.language not in self.path_dict.keys():
            raise AttributeError(
                'language %s has not been downloaded yet' % self.language)
        if load:
            self.ft_model = load_model(
                os.path.join(self.bin_folder, self.path_dict[self.language]))
    
    def fit(self, X, y=None):

        if not hasattr(self, 'ft_model'):
            self.ft_model = load_model(
                os.path.join(self.bin_folder, self.path_dict[self.language]))
            
        return self

    def transform(self, X):

        start = time.process_time()
        if not isinstance(X, np.ndarray):
            X = X.to_numpy().ravel()
        unq_X, lookup = np.unique(X, return_inverse=True)
        X_dict = dict()
        for i, x in enumerate(unq_X):
            if x.find('\n') != -1:
                unq_X[i] = ' '.join(x.split('\n'))

        for x in unq_X:
            X_dict[x] = self.ft_model.get_sentence_vector(x)

        X_out = np.empty((len(lookup), 300))
        for x, x_out in zip(unq_X[lookup], X_out):
            x_out[:] = X_dict[x]
        return X_out