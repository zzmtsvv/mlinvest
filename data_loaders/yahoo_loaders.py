import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union
from utils import load_json



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
