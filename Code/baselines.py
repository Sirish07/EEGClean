import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.__set__params(cfg) 
        self.fc1 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc1 = nn.Linear(self.inputs_dim, self.inputs_dim)
        self.fc1 = nn.Linear(self.inputs_dim, self.inputs_dim)
    
    def _set_params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])
    



    