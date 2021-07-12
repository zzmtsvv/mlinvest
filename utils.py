import json
import os
import sys
import hashlib
import pandas as pd
from copy import deepcopy
import numpy as np



def check_or_create_folder(file_path):
  if '/' in file_path:
    folder_path = '/'.join(file_path.split('/')[:-1])
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

def save_json(data, file_path):
  check_or_create_folder(file_path)
  with open(file_path, 'w') as f:
    json.dump(data, f, ensure_ascii=False)

def load_json(file_path):
  with open(file_path) as f:
    data = json.load(f)
  return data

def load_config():
    base_dir = os.path.expanduser('~')
    mlinvest_dir = os.path.join(base_dir, '.mlinvest')
    config_path = os.path.join(mlinvest_dir, 'config.json')
    return load_json(config_path)

def load_secrets():
    base_dir = os.path.expanduser('~')
    mlinvest_dir = os.path.join(base_dir, '.mlinvest')
    secrets_path = os.path.join(mlinvest_dir, 'secrets.json')
    return load_json(secrets_path)

def load_tickers():
    base_dir = os.path.expanduser('~')
    mlinvest_dir = os.path.join(base_dir, '.mlinvest')
    tickers_path = os.path.join(mlinvest_dir, 'tickers.json')
    return load_json(tickers_path)

def repeat_copy(data, counter):
  return [deepcopy(data) for _ in range(counter)]

def int_hash_of_str(text:str):
  return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

def step_function(df, x, y):
  temp_df = pd.DataFrame()
  temp1 = df[[x, y]]
  temp1['idx'] = 2
  temp2 = temp1.copy()
  temp2[y] = temp2[y].shift(-1)
  temp2['idx'] = 1

  res = pd.concat([temp1, temp2])
  res = res.sort_values([x, 'idx'])
  del res['idx']

  temp_df[y] = [res[y].values[-1]]
  temp_df[x] = np.datetime64('now')

  return pd.concat([res, temp_df])


class CM:
    """
    Convergence Monitor and report convergence to :data:`sys.stderr`.
    """

    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        """
        tol : double - EM Convergence threshold

        n_iter : int - Maximum number of iterations to perform
            Maximum number of iterations to perform.
        verbose : bool - Whether per-iteration convergence reports are printed.
            Whether per-iteration convergence reports are printed.
        """
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque()
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)
        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))
      
      
def split_x_lengths(x, lengths):
  if lengths is None:
    return [x]
  cumsum = np.cumsum(lengths)
  n_samples = len(x)
  if cumsum[-1] > n_samples:
    raise ValueError(f'more than {n_samples} samples in lengths array {lengths}')
  elif cumsum[-1] != n_samples:
    warnings.warn('deec')
  return np.split(x, cumsum)[:-1]


def check_is_fitted(estimator, attribute):
    if not hasattr(estimator, attribute):
        raise NotFittedError(
            "This %s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this method."
            % type(estimator).__name__)

        
def log_normalize(x, axis=None):
  if axis is not None and x.shape[axis] == 1:
    x[:] = 0
  else:
    with np.errstate(under='ignore'):
      x_lse = special.logsumexp(x, axis, keepdims=True)
    x -= x_lse

def normalize(x, axis=None):
  summa = x.sum()
  if axis and x.ndim > 1:
    summa[summa == 0] = 1
    shape = list(x.shape)
    shape[axis] = 1
    summa.shape = shape
  
  x / summa

  
def log_mask_zero(x):
  x = np.asarray(x)
  with np.errstate(under='ignore'):
    return np.log(x)

  
def fill_covars(covars, covariance_type='full', n_components=1, n_features=1):
  if covariance_type == 'full':
    return covars
  elif covariance_type == 'diag':
    return np.array(list(map(np.diag, covars)))
  elif covariance_type == 'tied':
    return np.tile(covars, (n_components, 1, 1))
  elif covariance_type == 'spherical':
    eye = np.eye(n_features)[np.newaxis, :, :]
    covars = covars[:, np.newaxis, np.newaxis]
    return eye * covars
def _validate_covars(covars, covariance_type, n_components):
    """Do basic checks on matrix covariance sizes and values."""
    from scipy import linalg
    if covariance_type == 'spherical':
        if len(covars) != n_components:
            raise ValueError("'spherical' covars have length n_components")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be positive")
    elif covariance_type == 'tied':
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif covariance_type == 'diag':
        if len(covars.shape) != 2:
            raise ValueError("'diag' covars must have shape "
                             "(n_components, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be positive")
    elif covariance_type == 'full':
        if len(covars.shape) != 3:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        elif covars.shape[1] != covars.shape[2]:
            raise ValueError("'full' covars must have shape "
                             "(n_components, n_dim, n_dim)")
        for n, cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                    or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)
    else:
        raise ValueError("covariance_type must be one of " +
                         "'spherical', 'tied', 'diag', 'full'")
