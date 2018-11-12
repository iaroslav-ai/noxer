"""
Feature preprocessing of data, such as expanding
categorical features to numerical ones.
"""

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin, MetaEstimatorMixin, clone
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.feature_selection.from_model import SelectorMixin, _get_feature_importances

import numpy as np

class SelectFromModelPercentile(BaseEstimator, SelectorMixin, MetaEstimatorMixin):
    """
    Extracts a certain percentile of weights based on importance
    of the features, as measured by some fitted model.
    """
    def __init__(self, estimator, percentile=10.0, prefit=False, norm_order=1):
        self.estimator = estimator
        self.percentile = percentile
        self.prefit = prefit
        self.norm_order = norm_order

    def _get_support_mask(self):
        # SelectFromModelPercentile can directly call on transform.
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError(
                'Either fit SelectFromModelPercentile before transform or set "prefit='
                'True" and pass a fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator)
        sz = int(len(scores) * self.percentile)
        I = scores.argsort()[-sz:]

        threshold = scores*0.0
        threshold[I] = 1.0
        threshold = threshold > 0.5

        return threshold

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self


    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """
        if self.prefit:
            raise NotFittedError(
                "Since 'prefit=True', call transform directly")
        if not hasattr(self, "estimator_"):
            self.estimator_ = clone(self.estimator)
        self.estimator_.partial_fit(X, y, **fit_params)
        return self



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


class FlattenFeatures(BaseEstimator, TransformerMixin):
    """Flattens the shape of input to the vector"""
    def fit(self, X, y=None):
        # create label encoder
        self.shape = X[0].shape
        return self

    def transform(self, X, y=None):
        X = np.reshape(X, (len(X), -1))
        return X

    def inverse_transform(self, X, y=None):
        X = np.reshape(X, (-1,) + self.shape)
        return X


class ReshapeFeatures(BaseEstimator, TransformerMixin):
    """Converts every element in input array to a particular shape."""
    def __init__(self, shape=None):
        self.shape = shape

    def fit(self, X, y=None):
        # create label encoder
        self.shape_original = X[0].shape
        return self

    def transform(self, X, y=None):
        X = np.reshape(X, (len(X),) + self.shape)
        return X

    def inverse_transform(self, X, y=None):
        X = np.reshape(X, (len(X),) + self.shape_original)
        return X