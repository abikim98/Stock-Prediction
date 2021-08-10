#!/usr/bin/env python
# coding: utf-8

# In[1]:


import FinanceDataReader as fdr


# In[2]:


import yfinance as yf
import numpy as np
import torch 
import torch.nn as nn


# In[3]:


data = yf.download('005930.KS', start = '2019-01-01', end = '2021-01-01')
data


# ## Data Preprocessing

# In[4]:


close = data['Close'].values
close.shape


# In[5]:


seq_len = 50 
target = seq_len+1

results = []
for index in range(len(close)-target):
    results.append(close[index:index+target]) #[0:51], [1:52], [2:53]


# In[6]:


print(len(results))
print(len(results[0]))


# In[7]:


normalized_data = []
for result in results:
    normalized_result = [p/result[0]-1 for p in result]
    normalized_data.append(normalized_result)
    
results = np.array(normalized_data)


# ## Split train and test set

# In[8]:


row = int(results.shape[0]*0.9)
train = results[:row]
valid = results[row:]
print(train.shape)
print(valid.shape)


# In[9]:


train_x = train[:, :-1]
train_y = train[:, -1]


print('train_x_shape', train_x.shape)
print('train_y_shape', train_y.shape)

valid_x = valid[:, :-1]
valid_y = valid[:, -1]

print('valid_x_shape', valid_x.shape)
print('valid_y_shape', valid_y.shape)


# In[10]:


train_x = train_x.reshape(train_x.shape[0], seq_len , -1)
print(train_x.shape)
train_y = train_y.reshape(train_y.shape[0], -1)
print(train_y.shape)


# In[11]:


valid_x = valid_x.reshape(valid_x.shape[0], seq_len, -1)
print(valid_x.shape)
valid_y = valid_y.reshape(valid_y.shape[0], 1)
print(valid_y.shape)


# In[12]:


import torch

train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

valid_x_tensor = torch.tensor(valid_x, dtype=torch.float32)
valid_y_tensor = torch.tensor(valid_y, dtype=torch.float32)

print(train_x_tensor.shape)
print(train_y_tensor.shape)
print(valid_x_tensor.shape)
print(valid_y_tensor.shape)


# ## Dataset

# In[13]:


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
    


# In[14]:


train_set = stockDataset('005930.KS', '2019-01-01', '2021-3-22', 50)
valid_set = stockDataset('005930.KS', '2021-03-23', '2021-07-16', 50)


# ## DataLoader

# In[15]:


train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = 20, shuffle = False)
valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size = len(valid_set), shuffle = False)


# ## Model

# In[16]:


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.lstm =nn.LSTM(input_dim, hidden_dim, num_layers, batch_first =True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        self.init_hidden(x)
        out,(hn, cn)=self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:,-1,:])
        
        return out
    
    def init_hidden(self, x):
        self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)


# In[17]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# In[18]:


num_epochs = 100
learning_rate = 0.01

input_dim = 1
hidden_dim = 32
num_layers = 1
output_dim = 1

lstm = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate)


# In[19]:


from tqdm import tqdm
import matplotlib.pyplot as plt
        
for epoch in range(num_epochs):
    lstm.train()
    train_loader = tqdm(train_dataloader)
    for data in train_loader:
        x, y = data
        x = x.reshape(x.shape[0], x.shape[1], 1).float()
        
        outputs = lstm(x.to(device))
        optimizer.zero_grad()
        loss = loss_function(outputs, y.float().to(device))
        loss.backward() 
        optimizer.step()
   
        train_loader.set_description(f"Epoch: {epoch}, loss: {loss.item():.5f}") 

    losses = []

    for data in valid_dataloader:
        x, y = data
        x = x.reshape(x.shape[0], x.shape[1], -1).float()
        y = y.float()
        with torch.no_grad():
            lstm.eval()
            valid_predict = lstm(x.to(device))

            loss = loss_function(valid_predict.cpu(), y)
    
    print('valid_loss', loss)

    data_predict = valid_predict.data.detach().cpu().numpy() 
    plt.plot(data_predict, label="Predicted Data")
    plt.plot(y, label="Actual Data")
    plt.legend()
    plt.show()


# In[ ]:





# In[ ]:




