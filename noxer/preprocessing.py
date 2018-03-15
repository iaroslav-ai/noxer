"""
Feature preprocessing of data, such as expanding
categorical features to numerical ones.
"""

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column with index `key` from some matrix X"""
    def __init__(self, key, row_space=True, as_matrix=True):
        self.key = key
        self.row_space = row_space
        self.as_matrix = as_matrix

    def fit(self, X, y=None):
        return self  # do nothing during fitting procedure

    def transform(self, data_matrix):
        # return a matrix with single column
        if self.row_space:
            R = data_matrix[:, [self.key]] # eg numpy array
        else:
            R = data_matrix[[self.key]] # eg pandas dataframe

        R = np.array(R)

        if not self.as_matrix:
            R = R[:, 0]

        return R

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Wrapper around LabelBinarizer. Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def _digest_shape(self, X):
        # create label encoder
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                M = X
            elif X.ndim == 2:
                M = X[:, 0]
            else:
                raise ValueError('One hot encoder does not work with nd, n>2 data')
        elif isinstance(X, list):
            if isinstance(X[0], list):
                M = [x[0] for x in X]
            else:
                M = X

        return M

    def fit(self, X, y=None):
        self.classes_ = list(sorted(set(self._digest_shape(X))))
        return self

    def transform(self, X, y=None):
        M = self._digest_shape(X)
        M = np.array(M)
        R = [M == c for c in self.classes_]
        R = np.column_stack(R)
        return R


class IntegerEncoder(BaseEstimator, TransformerMixin):
    """Wrapper around LabelBinarizer. Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def fit(self, X, y=None):
        # create label encoder
        M = X[:, 0]
        self.encoder = LabelEncoder()
        self.encoder.fit(M)
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X[:,0])[:, np.newaxis]