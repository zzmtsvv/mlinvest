import argparse
import os
import time
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import mlinvest.utils
from mlinvest.downloaders import YahooDownloader
from mlinvest.utils import load_config, save_json, load_tickers

import yfinance as yf

global _data_path
_data_path = None

def download_ticker(ticker):
  try:
    data = yf.Ticker(ticker)
    df = data.quarterly_financials.T
    df['date'] = df.index

    df.to_csv(f'{data_path}/quarterly/{ticker}.csv')
    save_json(data.info, f'{_data_path}/base/{ticker}.json')

    time.sleep(np.random.uniform(0.1, 0.5))
  except:
    None

def main(data_path=None):
  if data_path is None:
    config = load_config()
    data_path = config['yahoo_data_path']
  
  global _data_path
  _data_path = data_path
  tickers = load_tickers()['base_us_stocks']
  os.makedirs(f'{data_path}/quarterly', exist_ok=True)
  os.makedirs(f'{data_path}/base', exist_ok=True)

  p = Pool(12)
  for _ in tqdm(p.imap(download_ticker, tickers)):
    None

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  arg = parser.add_argument
  arg('--data_path', type=str)
  args = parser.parse_args()
  main(args.data_path)
