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
from utils import load_config, load_secrets, load_json, save_json



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
