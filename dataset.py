import yfinance as yf
import numpy as np
import torch 
import torch.nn as nn



data = yf.download('005930.KS', start = '2019-01-01', end = '2021-01-01')

class stockDataset(torch.utils.data.Dataset):
    def __init__ (self, stock_path, start_date, end_date, seq_len):
        self.stock_path= yf.download('005930.KS', start_date, end_date)
        self.seq_len = seq_len 
        self.target = seq_len+1
        self.data = self.preprocessing(self.stock_path, self.seq_len, self.target)
        
    def __getitem__(self, idx):
        result = self.data[idx]
        x, y = result[:self.seq_len], result[self.seq_len]
        return x,y
    
    def __len__(self):
        return len(self.data)
    
    def preprocessing(self, stock_path, seq_len, target):
        result = []
        close = stock_path['Close'].values
        assert seq_len <len(data), ''
        
        for idx in range(len(close)-(seq_len+1)):
            result.append(close[idx:idx+seq_len+1])
        normalized_data = []
        
        for i in result:
            normalized_result = [p/i[0]-1 for p in i]
            normalized_data.append(normalized_result)
        return np.array(normalized_data)

train_set = stockDataset('005930.KS', '2019-01-01', '2021-3-22', 50)
valid_set = stockDataset('005930.KS', '2021-03-23', '2021-07-16', 50)

