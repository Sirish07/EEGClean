import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from loss_function import *

class LFADSNET(nn.Module):
    def __init__(self, cfg):
        super(LFADSNET, self).__init__()
        self._set_params(cfg)
        
        if self.seed is None:
            self.seed = random.randint(1, 10000)
        else:
            print("Present seed : {}".format(self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed)

        """ Network Initialisation """
        # Generator Forward encoder
        self.gru_Egen_forward = nn.GRUCell(input_size = self.inputs_dim, hidden_size = self.g0_encoder_dim)

        # Generator Backward encoder
        self.gru_Egen_backward = nn.GRUCell(input_size = self.inputs_dim, hidden_size = self.g0_encoder_dim)

        # Generator
        self.gru_generator = nn.GRUCell(input_size = self.factors_dim, hidden_size = self.g_dim)

        """ Fully Connected Layers """
        self.fc_g0mean = nn.Linear(in_features = 2 * self.g0_encoder_dim, out_features = self.g_dim)
        self.fc_g0logvar = nn.Linear(in_features = 2 * self.g0_encoder_dim, out_features = self.g_dim)

        self.fc_factors = nn.Linear(in_features = self.g_dim, out_features = self.factors_dim)

        self.recon_fc1 = nn.Linear(in_features = self.factors_dim, out_features = self.inputs_dim)
        self.recon_fc2 = nn.Linear(in_features = self.inputs_dim, out_features = self.inputs_dim)
        self.recon_fc3 = nn.Linear(in_features = self.inputs_dim, out_features = self.inputs_dim) 

        """ Dropout Layer """
        self.dropout = nn.Dropout(1.0 - self.keep_prob)

        for m in self.modules():
            if isinstance(m, nn.GRUCell):
                k_ih = m.weight_ih.shape[1]
                k_hh = m.weight_hh.shape[1]
                m.weight_ih.data.normal_(std = k_ih ** -0.5)
                m.weight_hh.data.normal_(std = k_hh ** -0.5)

            elif isinstance(m, nn.Linear):
                    k = m.in_features
                    m.weight.data.normal_(std = k ** -0.5)

        self.fc_factors.weight.data = F.normalize(self.fc_factors.weight.data, dim = 1)
        self.g0_prior_mu = nn.parameter.Parameter(torch.tensor(0.0))
        self.u_prior_mu  = nn.parameter.Parameter(torch.tensor(0.0))

        from math import log
        self.g0_prior_logkappa = nn.parameter.Parameter(torch.tensor(log(self.g0_prior_logkappa)))

    def initialize(self, batch_size = None):

        batch_size = batch_size if batch_size is not None else self.batch_size

        self.g0_prior_mean = torch.ones(batch_size, self.g_dim).to(self.device) * self.g0_prior_mu
        self.g0_prior_logvar = torch.ones(batch_size, self.g_dim).to(self.device) * self.g0_prior_logkappa
        self.efgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))
        self.ebgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))

        if self.save_variables:
            self.factors = torch.zeros(batch_size, self.T, self.factors_dim)
            self.initial_state = torch.zeros(batch_size, self.T, self.g_dim)
            self.predicted = torch.zeros(batch_size, self.T, self.inputs_dim)


    def encode(self, x):

        """ Encoder Layer """

        if self.keep_prob < 1.0:
            x = self.dropout(x)

        for t in range(1, self.T + 1):
            self.efgen = torch.clamp(self.gru_Egen_forward(x[:, t - 1], self.efgen), max = self.clip_val)
            self.ebgen = torch.clamp(self.gru_Egen_backward(x[:, -t], self.ebgen), max = self.clip_val)

        egen = torch.cat((self.efgen, self.ebgen), dim = 1)

        if self.keep_prob < 1.0:
            egen = self.dropout(egen)
        
        self.g0_mean = self.fc_g0mean(egen)
        self.g0_logvar = torch.clamp(self.fc_g0logvar(egen), min = np.log(0.0001))
        self.g = Variable(torch.randn(self.batch_size, self.g_dim).to(self.device)) * torch.exp(0.5 * self.g0_logvar) + self.g0_mean
        
        self.kl_loss   = KLCostGaussian(self.g0_mean, self.g0_logvar,
                                        self.g0_prior_mean, self.g0_prior_logvar)/x.shape[0]

        self.f = self.fc_factors(self.g)

    def generate(self, x):
        """ Generator Layer """

        for t in range(self.T):

            econ_and_fac = torch.cat((self.efcon[:, t+1].clone(), self.ebcon[:,t].clone(), self.f), dim = 1)

            # Dropout the controller encoder outputs and factors
            if self.keep_prob < 1.0:
                econ_and_fac = self.dropout(econ_and_fac)
            
            # Update controller with controller encoder outputs
            self.c = torch.clamp(self.gru_controller(econ_and_fac, self.c), min=0.0, max=self.clip_val)

            # Calculate posterior distribution parameters for inferred inputs from controller state
            self.u_mean   = self.fc_umean(self.c)
            self.u_logvar = self.fc_ulogvar(self.c)

            # Sample inputs for generator from u(t) posterior distribution
            self.u = Variable(torch.randn(self.batch_size, self.u_dim).to(self.device))*torch.exp(0.5*self.u_logvar) \
                        + self.u_mean

            # KL cost for u(t)
            self.kl_loss = self.kl_loss + KLCostGaussian(self.u_mean, self.u_logvar,
                                        self.u_prior_mean, self.u_prior_logvar)/x.shape[0]
                                        
            self.g = torch.clamp(self.gru_generator(self.u, self.g), min = -self.clip_val, max = self.clip_val)

            if self.keep_prob < 1.0:
                self.g = self.dropout(self.g)
            
            self.f = self.fc_factors(self.g)

            out = F.relu(self.recon_fc1(self.f))
            if self.keep_prob < 1.0:
                out = self.dropout(out)
            
            out = F.relu(self.recon_fc2(out))
            if self.keep_prob < 1.0:
                out = self.dropout(out)

            self.output = self.recon_fc3(out)

            if self.save_variables:
                self.factors[:, t] = self.f
                self.initial_state[:, t] = self.g
                self.predicted[:, t] = self.output

    def forward(self, x):

        batch_size, steps_dim, inputs_dim = x.shape
        assert steps_dim == self.T
        assert inputs_dim == self.inputs_dim

        self.batch_size = batch_size
        self.initialize(batch_size = batch_size)
        self.encode(x)
        self.generate(x)
    
    def _set_params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])