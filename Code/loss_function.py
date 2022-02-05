import tensorflow as tf
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

################################# loss functions ##########################################################

def denoise_loss_mse(denoise, clean):    
  loss = nn.MSELoss(reduction = 'mean')
  return loss(denoise, clean)

def denoise_loss_rmse(denoise, clean):      #tmse
  loss = denoise_loss_mse(denoise, clean)
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return torch.sqrt(loss)

def denoise_loss_rrmset(denoise, clean):      #tmse
  rmse1 = denoise_loss_rmse(denoise, clean)
  rmse2 = denoise_loss_rmse(clean, torch.zeros(clean.shape))
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return rmse1/rmse2

def denoise_loss_rrmsepsd(denoise, clean):
  denoise, clean = torch.squeeze(denoise), torch.squeeze(clean)
  result = []
  for len in range(denoise.shape[0]):
    psd1,_ = plt.psd(denoise[len, :], Fs = 256)
    psd2,_ = plt.psd(clean[len, :], Fs = 256)
    psd1, psd2 = torch.Tensor(psd1), torch.Tensor(psd2)
    rmse1 = denoise_loss_rmse(psd1, psd2)
    rmse2 = denoise_loss_rmse(psd2, torch.zeros(psd2.shape))
    result.append(rmse1/ rmse2)
  return np.mean(result)

def average_correlation_coefficient(denoise, clean):
  denoise, clean = torch.squeeze(denoise), torch.squeeze(clean)
  result = []
  for len in range(denoise.shape[0]):
    temp1 = pd.Series(denoise[len, :])
    temp2 = pd.Series(clean[len, :])
    covar = temp1.cov(temp2)
    var_prod = math.sqrt(temp1.var() * temp2.var())
    result.append(covar / var_prod)  
  return np.mean(result) 
  
def KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv):
    '''
    KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv)
    KL-Divergence between a prior and posterior diagonal Gaussian distribution.
    Arguments:
        - post_mu (torch.Tensor): mean for the posterior
        - post_lv (torch.Tensor): logvariance for the posterior
        - prior_mu (torch.Tensor): mean for the prior
        - prior_lv (torch.Tensor): logvariance for the prior
    '''
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).sum()
    return klc
