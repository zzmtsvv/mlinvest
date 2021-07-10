import numpy as np
from sklearn.metrics import log_loss

def std(series):
  if not len(series):
    return np.nan
  return ((series - series.mean()) ** 2).mean() ** 0.5 / series[0]
 
def mean_absolute_percentage_error(y_true, y_pred): 
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def up_std(series):
  if not len(series):
    return np.nan
  up_intervals = series[series >= series.mean()]
  return ((up_intervals - series.mean()) ** 2).mean() ** 0.5 / series[0]

def down_std(series):
  if not len(series):
    return np.nan
  down_intervals = series[series < series.mean()]
  return ((down_intervals - series.mean()) ** 2).mean() ** 0.5 / series[0]

def max_rise(series):
  if not len(series):
    return np.nan
  return (series.max() - series[0]) / series[0]

def max_fall(series):
  if not len(series):
    return np.nan
  return (series[0] - series.min()) / series[0]

def median_abs_relative_error(target, pred):
  x = target != 0
  values = np.abs((target[x] - pred[x]) / target[x])
  values = values[~np.isnan(values)]
  return np.median(values)

def mean_abs_relative_error(target, pred):
  x = target != 0
  values = np.abs((target[x] - pred[x]) / target[x])
  values = values[~np.isnan(values)]
  return np.mean(values)

def log_loss_nan(target, pred):
  x = np.isnan(np.array(target)) | np.isnan(np.array(pred))
  return log_loss(target[~x], pred[~x])
