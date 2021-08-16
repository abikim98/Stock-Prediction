import torch
import torch.nn as nn 

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers
        self.input_dim = input_dim 
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, hid = self.rnn(x)
        out = self.fc(out[:,-1,:])
        
        return out 


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        self.init_hidden(x)
        out, (hn,cn) = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:,-1,:])
        return out 
    
    def init_hidden(self, x):
        self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

