import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union
from mlinvest.utils import load_json



class YahooQuarterly:
  def __init__(self, data_path, count=None):
    self.data_path = data_path
    self.count = count # maximum number of last quarters to return.
  
  def load(self, index: List[str]) -> pd.DataFrame:
    res = []
    for ticker in index:
      path = f'{self.data_path}/quarterly/{ticker}.csv'
      if not os.path.exists(path):
        continue
      df = pd.read_csv(path)
      df['ticker'] = ticker
      if self.count is not None:
        df = df[:self.count]
      res.append(df)
    
    if not len(res):
      return None
    
    res = pd.concat(res).reset_index(drop=True)
    res['date'] = res['date'].astype(np.datetime64)
    res = res.drop_duplicates(['ticker', 'date'])
    res.index = range(len(res))
    return res

  
class YahooBase:
  def __init__(self, data_path):
    self.data_path = data_path
  
  def load(self, index: Optional[List[str]]=None) -> pd.DataFrame:
    res = []

    base_path = f'{self.data_path}/base'
    if index is None:
      index = [x.split('.json')[0] for x in os.listdir(base_path)]
    for ticker in index:
      path = f'{base_path}/{ticker}.json'
      if not os.path.exists(path):
        continue
      data = load_json(path)
      data['ticker'] = ticker
      res.append(data)
    
    if not len(res):
      return None
    
    return pd.DataFrame(res)
