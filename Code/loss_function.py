import tensorflow as tf
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd

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

  psd1,_,_ = plt.psd(denoise, Fs = 256)
  psd2,_,_ = plt.psd(clean, Fs = 256)
  rmse1 = denoise_loss_rmse(psd1, psd2)
  rmse2 = denoise_loss_rmse(psd2, torch.zeros(psd2.shape))
  return rmse1 / rmse2

def average_correlation_coefficient(denoise, clean):
  denoise = pd.series(denoise)
  clean = pd.series(clean)
  covar = denoise.cov(clean)
  var_prod = torch.sqrt(denoise.var() * clean.var())
  return covar / var_prod
  
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