import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.special import erf, erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted

class GaussRankScaler(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-4, copy=True, n_jobs=None, interp_kind='linear', interp_copy=False):
        '''
        Load parameters for Gauss Rank Scaler.

        Args:
          epsilon (float): smoothing value
          copy (bool): make internal copy of input data `x`
          n_jobs (int): number of jobs to run in parallel
          interp_kind (str): type of interpolation
          interp_copy (bool): use references to input `x` and output data `y`

        Attributes:
          epsilon (float): smoothing value
          copy (bool): make internal copies of input data `x`
          interp_kind (str): type of interpolation
          interp_copy (bool): use references to input `x` and output data `y`
          fill_value (str):  method of filling values
          n_jobs (int): number of jobs to run in parallel
        '''
        self.epsilon = epsilon
        self.copy = copy
        self.interp_kind = interp_kind
        self.interp_copy = interp_copy
        self.fill_value = 'extrapolate'
        self.n_jobs = n_jobs

    def fit(self, X):
        '''
        Fit interpolation function with input data for scaling.

        Args:
          X (arr): input data with dimensions
            [n_samples, n_features]

        Returns:
          interp_func_ (arr): list of interpolation functions for each feature in the training set
        '''
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(x) for x in X.T)
        return self

    def _fit(self, x):
        x = self.drop_duplicates(x)
        rank = np.argsort(np.argsort(x))
        bound = 1.0 - self.epsilon
        factor = np.max(rank) / 2.0 * bound
        scaled_rank = np.clip(rank / factor - bound, -bound, bound)
        return interp1d(
            x, scaled_rank, kind=self.interp_kind, copy=self.interp_copy, fill_value=self.fill_value)

    def transform(self, X, copy=None):
        '''
        Scale the input data with the Gauss Rank algorithm.

        Args:
          X (arr): input data with dimensions
            [n_samples, n_features]
          copy (bool): make internal copy of input data `x`

        Returns:
          X (arr): transformed input data
        '''
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _transform(self, i, x):
        return erfinv(self.interp_func_[i](x))

    def inverse_transform(self, X, copy=None):
        '''
        Scale the data back to the original representation.

        Args:
          X (arr): input data with dimensions
            [n_samples, n_features]
          copy (bool): make internal copy of input data `x`

        Returns:
          X (arr): transformed input data
        '''
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._inverse_transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(self.interp_func_[i].y, self.interp_func_[i].x, kind=self.interp_kind,
                                   copy=self.interp_copy, fill_value=self.fill_value)
        return inv_interp_func(erf(x))

    @staticmethod
    def drop_duplicates(x):
        is_unique = np.zeros_like(x, dtype=bool)
        is_unique[np.unique(x, return_index=True)[1]] = True
        return x[is_unique]
