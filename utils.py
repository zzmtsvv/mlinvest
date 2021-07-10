import json
import os
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
