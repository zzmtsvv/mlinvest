import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import List
from sklearn.model_selection import GroupKFold
import torch
from torch import nn



def train_on_batch(model, optimizer, x_batch, y_batch, loss_function):
  model.train()
  model.zero_grad()

  output = model(x_batch.to(device).float())
  loss = loss_function(output, y_batch.to(device))
  loss.backward()
  optimizer.step()

  return loss.cpu().item()

def train_epoch(train_generator, model, loss_function, optimizer):
  epoch_loss = 0
  total = 0
  for it, (x_batch, y_batch) in enumerate(train_generator):
    batch_loss = train_on_batch(model, optimizer, x_batch, y_batch,
                                loss_function)
    epoch_loss += batch_loss * len(x_batch)
    total += len(x_batch)
  return epoch_loss / total

def trainer(count_of_epoch, batch_size, dataset,
            model, loss_function, optimizer, lr=0.001):
  
  optima = optimizer(model.parameters(), lr=lr)

  iterations = tqdm(range(count_of_epoch), desc='epoch')
  iterations.set_postfix({'train epoch loss': np.nan})

  for it in iterations:
    batch_generator = tqdm(
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                        shuffle=False), 
            leave=False,
            total=len(dataset)//batch_size+(len(dataset)%batch_size> 0))
    epoch_loss = train_epoch(batch_generator, model,
                             loss_function, optima)
    iterations.set_postfix({'train epoch loss': epoch_loss})


class LnEModel:
  '''
  just a model wrapper to fit on logarithm of target and predict the exponential
  '''
  def __init__(self, base_model):
    self.model = base_model
  
  def fit(self, x: pd.DataFrame, y):
    t = (y > 0).values

    if isinstance(self.model, nn.Module):
      # adaptation for pytorch
      train_dataset = torch.utils.data.TensorDataset(torch.tensor(x[t].values),
                                                     np.log(y[t]))
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      loss_fn = nn.L1Loss()
      optimizer = torch.optim.Adam(self.model.parameters())

      trainer(count_of_epoch=100,
              batch_size=32,
              dataset=train_dataset,
              model=self.model,
              loss_function=loss_fn,
              optimizer=optimizer)
    else:
      self.model.fit(x[t], np.log(y[t]))
  
  def predict(self, x):
    if isinstance(self.model, nn.Module):
      return np.exp(self.model(x).numpy())
    
    return np.exp(self.model.predict(x))


class TimeSeriesOOfModel:
  '''
  just a model wrapper for shorthanding out-of-fold time-series partitioning
  not optimized for pytorch
  '''
  def __init__(self, base_model, time_column, fold_counter=5):
    self.fold_counter = fold_counter
    self.time_column = time_column
    self.models = []

    for _ in range(fold_counter):
      self.models.append(deepcopy(base_model))
    
    self.time_bounds = None
    self.fitted_folds = np.zeros(fold_counter)
  
  def fill_time_bounds(self, times: List[np.datetime64]):
    max_time, min_time = max(times), min(times)
    delta = (max_time - min_time) // self.fold_counter
    self.time_bounds = list()
    for fold_number in range(1, self.fold_counter):
      self.time_bounds.append(min_time + delta * fold_number)
    self.time_bounds.append(max_time)
    # fictitiousness for mapping the fit method
    self.time_bounds.append(max_time + np.timedelta64(10000, 'D'))
  
  def fit(self, x: pd.DataFrame, y, min_samples=5):
    times = x.reset_index()[self.time_column].astype(np.datetime64).values
    self.fill_time_bounds(times)

    for fold_number in range(self.fold_conter):
      curr_mask = times <= self.time_bounds[fold_number]
      if curr_mask.sum() >= min_samples:
        self.models[fold_number].fit(x[curr_mask], y[curr_mask])
        self.fitted_folds[fold_number] = 1
  
  def predict(self, x: pd.DataFrame) -> np.array:
    times = x.reset_index()[self.time_column].astype(np.datetime64).values
    pred_df = []

    x_curr = x[times <= self.time_bounds[0]]
    curr_pred_df = pd.DataFrame()
    curr_pred_df['pred'] = [np.nan] * len(x_curr)
    curr_pred_df.index = x_curr.index
    pred_df.append(curr_pred_df)

    for fold_number in range(self.fold_counter):
      curr_mask = (times > self.time_bounds[fold_number]) * \
                  (times <= self.time_bounds[fold_number + 1])
      x_curr = x[curr_mask]
      if not len(x_curr):
        continue

      if not self.fitted_folds[fold_number]:
        curr_pred_df = pd.DataFrame()
        curr_pred_df['pred'] = [np.nan] * len(x_curr)
        curr_pred_df.index = x_curr.index
        pred_df.append(curr_pred_df)
        continue
      
      try:
        pred = self.models[fold_number].predict_proba(x_curr)[:, 1]
      except:
        pred = self.models[fold_number].predict(x_curr)
      
      curr_pred_df = pd.DataFrame()
      curr_pred_df['pred'] = pred
      curr_pred_df.index = x_curr.index
      pred_df.append(curr_pred_df)

    pred_df = pd.concat(pred_df)
    pred_df = pred_df.loc[x.index]

    return pred_df['pred'].values
  
 
