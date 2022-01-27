import torch
import torch.nn as nn
import torch.nn.functional as F


class FcNN(nn.Module):
    def __init__(self, cfg):
        super(FcNN, self).__init__()
        self.__set__params(cfg) 
        self.fc1 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc2 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc3 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc4 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        hf = F.relu(self.fc1(x))
        hf = self.dropout(hf)

        hf = F.relu(self.fc2(hf))
        hf = self.dropout(hf)

        hf = F.relu(self.fc3(hf))
        hf = self.dropout(hf)

        self.predicted = self.fc4(hf)
    
    def __set__params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])

class LSTM_FFN(nn.Module):
    def __init__(self, cfg):
        super(LSTM_FFN, self).__init__()
        self.__set__params(cfg)
        self.lstm = nn.LSTM(input_size = self.inputs_dim, hidden_size = 1, batch_first = True)
        self.fc1 = nn.Linear(1, self.inputs_dim)
        self.fc2 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc3 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)

        output = F.relu(self.fc2(output))
        output = self.dropout(output)

        output = self.fc3(output)
        self.predicted = output

    def __set__params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])
    



    