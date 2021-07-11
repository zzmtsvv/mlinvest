import os
import requests
import time
import numpy as np
import pandas as pd
import copy
import json
from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool
from itertools import repeat
from mlinvest.utils import load_config, load_secrets, load_json, save_json



class TinkoffDownloader:
    def __init__(self):
        self.config = load_config()
        self.secrets = load_secrets()
        self.headers = {"Authorization": 
                        "Bearer {}".format(self.secrets['tinkoff_token'])}
        
    def get_stocks(self):
        url = 'https://api-invest.tinkoff.ru/openapi/market/stocks'
        response = requests.get(url, headers=self.headers)
        result = response.json()        
        
        return result
        
        
    def get_portfolio(self):
        url = 'https://api-invest.tinkoff.ru/openapi/portfolio' \
               '?brokerAccountId={}'
        url = url.format(self.secrets['tinkoff_broker_account_id'])
        response = requests.get(url, headers=self.headers)
        portfolio = response.json()
        
        return portfolio['payload']['positions']
        
        
    def get_figi_by_ticker(self, ticker):
        url = 'https://api-invest.tinkoff.ru/' \
              'openapi/market/search/by-ticker?ticker={}'.format(ticker)
        response = requests.get(url, headers=self.headers)
        figi = response.json()['payload']['instruments'][0]['figi']            
    
        return figi
    
    def get_lot_by_ticker(self, ticker):
        url = 'https://api-invest.tinkoff.ru/' \
              'openapi/market/search/by-ticker?ticker={}'.format(ticker)
        response = requests.get(url, headers=self.headers)
        lot = response.json()['payload']['instruments'][0]['lot']            
    
        return lot
        
    def get_last_price(self, ticker):
        figi = self.get_figi_by_ticker(ticker)      
        url = 'https://api-invest.tinkoff.ru/openapi/market/candles' \
              '?figi={}&from={}&to={}&interval=day'

        end = np.datetime64('now')
        end = str(end) + '%2B00%3A00'
        start = np.datetime64('now') - np.timedelta64(3, 'D')
        start = str(start) + '%2B00%3A00'
        
        url = url.format(figi, start, end)

        response = requests.get(url, headers=self.headers)
        close_price = response.json()['payload']['candles'][-1]['c']
        
        return close_price
    
    def get_price_history(self, ticker):
        figi = self.get_figi_by_ticker(ticker)      
        url = 'https://api-invest.tinkoff.ru/openapi/market/candles' \
              '?figi={}&from={}&to={}&interval=day'

        end = np.datetime64('now')
        end = str(end) + '%2B00%3A00'
        start = np.datetime64('now') - np.timedelta64(365*4, 'D')
        start = str(start) + '%2B00%3A00'
        
        url = url.format(figi, start, end)

        response = requests.get(url, headers=self.headers)
        return response
        close_price = response.json()['payload']['candles'][-1]['c']
        
        return close_price            
        
    def post_market_order(self, ticker, side, lots):
        figi = self.get_figi_by_ticker(ticker)
        url = 'https://api-invest.tinkoff.ru/openapi/orders/market-order?figi={}&brokerAccountId={}'
        url = url.format(figi, self.secrets['tinkoff_broker_account_id'])
        data = {
                "operation": side,
                "lots": lots,
               }
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        
        return response

    
 
class YahooDownloader:
  def __init__(self):
    self.secrets = load_secrets()
    self.config = load_config
  
  def parse_json_quarterly(self, json_data):
    data = []
    for row in json_data:
      temp = dict()
      for key in row.keys():
        if key == 'endDate':
          temp['date'] = row[key]['fmt']
          continue
        
        if type(row[key]) == dict and 'raw' in row[key]:
          temp[key] = row[key]['raw']
          continue
        
        if type(row[key]) == dict and len(row[key]) == 0:
          temp[key] = None
          continue
        
        temp[key] = row[key]
      data.append(temp)
    return pd.DataFrame(data)
  
  def parse_json_base(self, json_data):
    row = dict()
    for key in json_data.keys():
      if type(json_data[key]) == dict and 'raw' in json_data[key]:
        row[key] = json_data[key]['raw']
        continue
      
      if type(json_data[key]) in [list, dict] and len(json_data[key]) == 0:
        row[key] = None
        continue
      
      row[key] = json_data[key]
    return row
  
  def download_quarterly_ticker(self, ticker):
    try:
      url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}'
      url += '?modules=incomeStatementHistoryQuarterly'
      url += ',balanceSheetHistoryQuarterly'
      url += ',cashflowStatementHistoryQuarterly'

      r = requests.get(url.format(ticker=ticker))
      if r.status_code != 200:
        print(r.status_code, ticker)
        return
      try:
        json_data = r.json()['quoteSummary']['result'][0]
      except:
        return
      if not len(json_data['incomeStatementHistoryQuarterly']['incomeStatementHistory']):
        return
      
      q1 = self.parse_json_quarterly(json_data['incomeStatementHistoryQuarterly']['incomeStatementHistory'])
      q2 = self.parse_json_quarterly(json_data['balanceSheetHistoryQuarterly']['balanceSheetStatements'])
      q3 = self.parse_json_quarterly(json_data['cashflowStatementHistoryQuarterly']['cashflowStatements'])

      df = pd.merge(q1, q2, on='date', how='left', suffixes=('', '_y'))
      quarterly_df = pd.merge(df, q3, how='left', on='date', suffixes=('', '_z'))

      filepath = f'{self.data_path}/quarterly/{ticker}.csv'
      quarterly_df.to_csv(filepath, index=False)
      time.sleep(np.random.uniform(0, 0.5))
    except:
      time.sleep(np.random.uniform(0, 0.5))
    
  def download_base_ticker(self, ticker):
    try:
      url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}'
      url += '?modules=summaryProfile,defaultKeyStatistics'
      
      r = requests.get(url.format(ticker=ticker))
      if r.status_code != 200:
        print(r.status_code, ticker)
        return
      try:
        json_data = r.json()['quoteSummary']['result'][0]
      except:
        return
      
      data = dict()
      b1 = self.parse_json_base(json_data['summaryProfile'])
      b2 = self.parse_json_base(json_data['defaultKeyStatistics'])
      data.update(b1)
      data.update(b2)
      filepath = f'{self.data_path}/base/{ticker}.json'
      save_json(data, filepath)
      
      time.sleep(np.random.uniform(0, 0.5))
    except:
      time.sleep(np.random.uniform(0, 0.5))
  
  def download_quarterly(self, data_path, tickers, n_jobs=1):
    self.data_path = data_path
    os.makedirs('{}/quarterly'.format(self.data_path), exist_ok=True)
    with Pool(n_jobs) as p:
      for _ in tqdm(p.imap(self.download_quarterly_ticker, tickers)):
        kartoshka = 0
  
  def download_base(self, data_path, tickers, n_jobs=1):
    self.data_path = data_path
    os.makedirs('{}/base'.format(self.data_path), exist_ok=True)
    with Pool(n_jobs) as p:
      for _ in tqdm(p.imap(self._download_base_data_single, tickers)):
        None 
