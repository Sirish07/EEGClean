import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(25)

class FcNN(nn.Module):
    def __init__(self, cfg):
        super(FcNN, self).__init__()
        self.__set__params(cfg) 
        self.fc1 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc2 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc3 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc4 = nn.Linear(self.inputs_dim, self.inputs_dim)

    def forward(self, x):
        hf = F.relu(self.fc1(x))

        hf = F.relu(self.fc2(hf))

        hf = F.relu(self.fc3(hf))

        self.predicted = self.fc4(hf)
    
    def __set__params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])

class LSTM_FFN(nn.Module):
    def __init__(self, cfg):
        super(LSTM_FFN, self).__init__()
        self.__set__params(cfg)
        self.lstm = nn.LSTM(input_size = 1, hidden_size = 1, batch_first = True)
        self.fc1 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc2 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc3 = nn.Linear(self.inputs_dim, self.inputs_dim)

    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        batch_size = output.shape[0]
        output = torch.reshape(output, (batch_size, self.inputs_dim))
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        self.predicted = torch.reshape(output, (output.shape[0], output.shape[1], 1))

    def __set__params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])
    



    