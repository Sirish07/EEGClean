import numpy as np
import time
import math
from tqdm import tqdm
import torch
import datetime
import torch.nn.functional as F
from loss_function import *
from utils import *

class Trainer:
    def __init__(self, cfg):
        self._set_params(cfg)
        self.last_decay_epoch = 0
        self.current_step = 0
        self.best = np.inf
        self.train_loss_store = []
        self.valid_loss_store = []
        self.full_loss_store = {'train_loss' : {}, 'train_recon_loss' : {}, 'train_kl_loss' : {},
                                'valid_loss' : {}, 'valid_recon_loss' : {}, 'valid_kl_loss' : {},
                                'l2_loss' : {}}
        self.cost_weights = {'kl' : {'weight': 0, 'schedule_start': self.kl_weight_schedule_start,
                                     'schedule_dur': self.kl_weight_schedule_dur},
                             'l2' : {'weight': 0, 'schedule_start': self.l2_weight_schedule_start,
                                     'schedule_dur': self.l2_weight_schedule_dur}}

    def train(self, model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, optimizer):
        print("=============Training Start================")
        use_tensorboard = True
        writer = create_summary_writer(self.run_name)
        for self.epoch in range(self.maxepochs):
            print(f"Epoch: {self.epoch}")
            train_loss, l2_loss, kl_weight, l2_weight, initial_state = self.__train_on_epoch(model, noiseEEG_train, EEG_train, optimizer)
            valid_loss = self.__val_on_epoch(model, noiseEEG_val, EEG_val, l2_loss)
            if self.epoch == 6 or self.epoch == 10 or self.epoch == self.maxepochs - 1:
                plotTSNE(self.epoch, initial_state)
                writer.add_figure('Initial_State', plt.gcf(), self.epoch)

            if self.scheduler_on:
                self.__apply_decay(self.train_loss_store, train_loss, optimizer)

            self.train_loss_store.append(train_loss['epoch_train_loss'])
            self.valid_loss_store.append(valid_loss['epoch_valid_loss'])

            self.full_loss_store['train_loss'][self.epoch] = float(train_loss['epoch_train_loss'])
            self.full_loss_store['train_recon_loss'][self.epoch] = float(train_loss['epoch_recon_loss'])
            self.full_loss_store['train_kl_loss'][self.epoch] = float(train_loss['epoch_kl_loss'])
            self.full_loss_store['valid_loss'][self.epoch] = float(valid_loss['epoch_valid_loss'])
            self.full_loss_store['valid_recon_loss'][self.epoch] = float(valid_loss['epoch_recon_loss'])
            self.full_loss_store['valid_kl_loss'][self.epoch] = float(valid_loss['epoch_kl_loss'])
            self.full_loss_store['l2_loss'][self.epoch] = float(l2_loss.data)

            if use_tensorboard:
                
                writer.add_scalars('1_Loss/1_Total_Loss', {'Training' : float(train_loss['epoch_train_loss']), 
                                                         'Validation' : float(valid_loss['epoch_valid_loss'])}, self.epoch)

                writer.add_scalars('1_Loss/2_Reconstruction_Loss', {'Training' :  float(train_loss['epoch_recon_loss']), 
                                                                  'Validation' : float(valid_loss['epoch_recon_loss'])}, self.epoch)
                
                writer.add_scalars('1_Loss/3_KL_Loss' , {'Training' : float(train_loss['epoch_kl_loss']), 
                                                       'Validation' : float(valid_loss['epoch_kl_loss'])}, self.epoch)
                
                writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss.data), self.epoch)
                
                writer.add_scalar('2_Optimizer/1_Learning_Rate', model.lr, self.epoch)
                writer.add_scalar('2_Optimizer/2_KL_weight', kl_weight, self.epoch)
                writer.add_scalar('2_Optimizer/3_L2_weight', l2_weight, self.epoch)

            if self.current_step >= max(self.cost_weights['kl']['schedule_start'] + self.cost_weights['kl']['schedule_dur'],
                                        self.cost_weights['l2']['schedule_start'] + self.cost_weights['l2']['schedule_dur']):
                if self.valid_loss_store[-1] < self.best:
                    self.last_saved = self.epoch
                    self.best = self.valid_loss_store[-1]
                    # saving checkpoint
                    self.__save_checkpoint(model, optimizer) 
            writer.flush()       
        if use_tensorboard:
            writer.close()
        self.__save_checkpoint(model, optimizer, force = True)


    def test(self, model, noiseEEG, EEG):
        model.eval()
        batch_size = noiseEEG.shape[0]
        with torch.no_grad():
            noiseEEG_batch, EEG_batch = torch.reshape(torch.FloatTensor(noiseEEG), (batch_size, self.T, self.inputs_dim)).to(self.device), torch.reshape(torch.FloatTensor(EEG), (batch_size, self.T, self.inputs_dim)).to(self.device)
            model(noiseEEG_batch)
            denoiseout = model.predicted
            mse_loss = denoise_loss_mse(denoiseout, EEG_batch)
            rmset_loss = denoise_loss_rrmset(denoiseout, EEG_batch)
            rmsepsd_loss = denoise_loss_rrmsepsd(denoiseout, EEG_batch)
            acc = average_correlation_coefficient(denoiseout, EEG_batch)
            print(f"MSE loss = {mse_loss}, RRMSET Loss ={rmset_loss}, RRMSE_spec Loss = {rmsepsd_loss}, ACC = {acc}")

    def __train_on_epoch(self, model, noiseEEG, EEG, optimizer):
        model.train()
        start = time.time()
        batch_size = self.batch_size
        batch_num = math.ceil(noiseEEG.shape[0]/batch_size)
        initial_state = []
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for n_batch in range(batch_num):
                self.current_step += 1
                if n_batch == batch_num:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch :] , EEG[batch_size*n_batch :]
                else:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch : batch_size*(n_batch+1)] , EEG[batch_size*n_batch : batch_size*(n_batch+1)]
                
                noiseEEG_batch, EEG_batch = torch.reshape(torch.FloatTensor(noiseEEG_batch), (batch_size, self.T, self.inputs_dim)).to(self.device), torch.reshape(torch.FloatTensor(EEG_batch), (batch_size, self.T, self.inputs_dim)).to(self.device)
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    self.__weight_schedule(self.current_step)
                    model(noiseEEG_batch)
                    denoiseout = model.predicted
                    mse_loss = denoise_loss_mse(denoiseout, EEG_batch)
                    kl_loss = model.kl_loss
                    l2_loss = model.l2_gen_scale * model.gru_generator.weight_hh.norm(2)/model.gru_generator.weight_hh.numel() #+ model.l2_con_scale * model.gru_controller.weight_hh.norm(2)/model.gru_controller.weight_hh.numel()

                    kl_weight = self.cost_weights['kl']['weight']
                    l2_weight = self.cost_weights['l2']['weight']
                    loss = mse_loss + kl_weight * kl_loss + l2_weight * l2_loss
                    assert not torch.isnan(loss.data), "Loss is NaN"

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = model.max_norm)
                    optimizer.step()
                    model.fc_factors.weight.data = F.normalize(model.fc_factors.weight.data, dim=1)

                    train_loss += loss.data / float(batch_num)
                    train_recon_loss += mse_loss.data / float(batch_num)
                    train_kl_loss += kl_loss.data / float(batch_num)

                    pbar.update()
                
                if n_batch % 5 == 0:
                    initial_state.append(model.initial_state.detach().cpu().numpy())
            pbar.close()
        end = time.time()
        initial_state = np.stack(initial_state)
        initial_state = np.reshape(initial_state, (-1, model.g_dim))
        print(f"Train loss: {train_loss}, time = {end-start}/per epoch")
        return {"epoch_train_loss": train_loss,
                "epoch_recon_loss": train_recon_loss,
                "epoch_kl_loss": train_kl_loss}, l2_loss, kl_weight, l2_weight, initial_state

    def __weight_schedule(self, current_step):
        for cost_key in self.cost_weights.keys():
            # Get step number of scheduler
            weight_step = max(current_step - self.cost_weights[cost_key]['schedule_start'], 0)
            # Calculate schedule weight
            self.cost_weights[cost_key]['weight'] = min(weight_step/ self.cost_weights[cost_key]['schedule_dur'], 1.0)
    
    def __apply_decay(self, train_loss_store, train_epoch_loss, optimizer):
        if len(train_loss_store) >= self.scheduler_patience:
                if all((train_epoch_loss['epoch_train_loss'] > past_loss for past_loss in train_loss_store[-self.scheduler_patience:])):
                    if self.epoch >= self.last_decay_epoch + self.scheduler_cooldown:
                        self.lr  = self.lr * self.lr_decay
                        self.last_decay_epoch = self.epoch
                        for g in optimizer.param_groups:
                            g['lr'] = self.lr
                        print('Learning rate decreased to %.8f'%self.lr)
    
    def __val_on_epoch(self, model, noiseEEG, EEG, l2_loss):
        model.eval()
        batch_size = self.batch_size
        batch_num = math.ceil(noiseEEG.shape[0]/batch_size)
        valid_loss = 0
        valid_recon_loss = 0
        valid_kl_loss = 0
        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for n_batch in range(batch_num):
                if n_batch == batch_num:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch :] , EEG[batch_size*n_batch :]
                else:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch : batch_size*(n_batch+1)] , EEG[batch_size*n_batch : batch_size*(n_batch+1)]
                
                noiseEEG_batch, EEG_batch = torch.reshape(torch.FloatTensor(noiseEEG_batch), (-1, self.T, self.inputs_dim)).to(self.device), torch.reshape(torch.FloatTensor(EEG_batch), (-1, self.T, self.inputs_dim)).to(self.device)
                with torch.no_grad():
                    model(noiseEEG_batch)
                    denoiseout = model.predicted
                    mse_loss = denoise_loss_mse(denoiseout, EEG_batch)
                    kl_loss = model.kl_loss
                    loss = mse_loss + kl_loss + l2_loss
                    assert not torch.isnan(loss.data), "Loss is NaN"

                    valid_loss += loss.data / float(batch_num)
                    valid_recon_loss += mse_loss.data / float(batch_num)
                    valid_kl_loss += kl_loss.data / float(batch_num)

                    pbar.update()
            pbar.close()
        print(f"Validation loss: {valid_loss}")
        return {"epoch_valid_loss": valid_loss,
                "epoch_recon_loss": valid_recon_loss,
                "epoch_kl_loss": valid_kl_loss}

    def _set_params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])

    def __save_checkpoint(self, model, optimizer, force = False, purge_limit = 20):
        EXPERIMENT = f"{self.run_name}"
        model_path = f"../Results/{EXPERIMENT}"
        if force:
            pass
        else:
            if purge_limit:
                try:
                    _,_,filenames = list(os.walk(model_path))[0]
                    split_filenames = [os.path.splitext(fn)[0].split('_') for fn in filenames]
                    epochs = [attr[2] for attr in split_filenames]
                    epochs.sort()
                    last_saved_epoch = epochs[-1]
                    if self.epoch - 20 <= int(last_saved_epoch):
                        rm_filename = [filename for filename in filenames if last_saved_epoch in filename][0]
                        os.remove(model_path + rm_filename)
                except IndexError:
                    pass
        
        timestamp = datetime.datetime.now().strftime('%y%m%d%H%M')
        epoch = str('%i'%self.epoch)
        loss = str(self.valid_loss_store[-1].item()).replace('.','-')
        output_filename = '%s_epoch_%s_loss_%s.pth'%(timestamp, epoch, loss)
        assert os.path.splitext(output_filename)[1] == '.pth', 'Output filename must have .pth extension'

        train_dict = {'best' : self.best, 'train_loss_store': self.train_loss_store,
                    'valid_loss_store' : self.valid_loss_store,
                    'full_loss_store' : self.full_loss_store,
                    'epochs' : self.epoch, 'current_step' : self.current_step,
                    'last_decay_epoch' : self.last_decay_epoch,
                    'learning_rate' : self.lr,
                    'cost_weights' : self.cost_weights}

        torch.save({'net' : model.state_dict(), 'opt' : optimizer.state_dict(), 'train' : train_dict},
                model_path+'/'+output_filename)

    def __health_check(self, model, writer):
        '''
        Checks the gradient norms for each parameter, what the maximum weight is in each weight matrix,
        and whether any weights have reached nan
        
        Report norm of each weight matrix
        Report norm of each layer activity
        Report norm of each Jacobian
        
        To report by batch. Look at data that is inducing the blow-ups.
        
        Create a -Nan report. What went wrong? Create file that shows data that preceded blow up, 
        and norm changes over epochs
        
        Theory 1: sparse activity in real data too difficult to encode
            - maybe, but not fixed by augmentation
            
        Theory 2: Edgeworth approximation ruining everything
            - probably: when switching to order=2 loss does not blow up, but validation error is huge
        '''
        
        hc_results = {'Weights' : {}, 'Gradients' : {}, 'Activity' : {}}
        odict = model._modules
        ii=1
        for name in odict.keys():
            if 'gru' in name:
                writer.add_scalar('3_Weight_norms/%ia_%s_ih'%(ii, name), odict.get(name).weight_ih.data.norm(), self.current_step)
                writer.add_scalar('3_Weight_norms/%ib_%s_hh'%(ii, name), odict.get(name).weight_hh.data.norm(), self.current_step)
                
                if self.current_step > 1:

                    writer.add_scalar('4_Gradient_norms/%ia_%s_ih'%(ii, name), odict.get(name).weight_ih.grad.data.norm(), self.current_step)
                    writer.add_scalar('4_Gradient_norms/%ib_%s_hh'%(ii, name), odict.get(name).weight_hh.grad.data.norm(), self.current_step)
            
            elif 'fc' in name or 'conv' in name:
                writer.add_scalar('3_Weight_norms/%i_%s'%(ii, name), odict.get(name).weight.data.norm(), self.current_step)
                if self.current_step > 1:
                    writer.add_scalar('4_Gradient_norms/%i_%s'%(ii, name), odict.get(name).weight.grad.data.norm(), self.current_step)
 
            ii+=1
        
        writer.add_scalar('5_Activity_norms/1_efgen', model.efcon.data.norm(), self.current_step)
        writer.add_scalar('5_Activity_norms/2_ebgen', model.ebcon.data.norm(), self.current_step)