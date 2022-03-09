import numpy as np
import time
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from loss_function import *
from utils import *

class Trainer:
    def __init__(self, cfg):
        self._set_params(cfg)
        self.last_decay_epoch = 0
        self.best = np.inf
        self.train_loss_store = []
        self.valid_loss_store = []
        self.full_loss_store = {'train_loss' : {},
                                'valid_loss' : {}}

    def train(self, model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, optimizer):
        print("=============Training Start================")
        use_tensorboard = True
        writer = create_summary_writer(self.run_name)
        current_step = 0
        for self.epoch in range(self.maxepochs):
            print("-" * 50 + str(self.epoch) + "-" * 50)
            train_loss, current_step = self.__train_on_epoch(model, noiseEEG_train, EEG_train, optimizer, current_step)
            valid_loss = self.__val_on_epoch(model, noiseEEG_val, EEG_val)

            self.train_loss_store.append(train_loss['epoch_train_loss'])
            self.valid_loss_store.append(valid_loss['epoch_valid_loss'])

            self.full_loss_store['train_loss'][self.epoch] = float(train_loss['epoch_train_loss'])
            self.full_loss_store['valid_loss'][self.epoch] = float(valid_loss['epoch_valid_loss'])

            if use_tensorboard:
                
                writer.add_scalars('1_Loss/1_Total_Loss', {'Training' : float(train_loss['epoch_train_loss']), 
                                                         'Validation' : float(valid_loss['epoch_valid_loss'])}, self.epoch)

                writer.add_scalar('2_Optimizer/1_Learning_Rate', model.lr, self.epoch)

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
        with torch.no_grad():
            noiseEEG, EEG = torch.FloatTensor(np.expand_dims(noiseEEG, axis = 1)).to(self.device), torch.FloatTensor(np.expand_dims(EEG, axis = 1)).to(self.device)
            model(noiseEEG)
            denoiseout = model.predicted
            mse_loss = denoise_loss_mse(denoiseout, EEG)
            rmset_loss = denoise_loss_rrmset(denoiseout, EEG)
            rmsepsd_loss = denoise_loss_rrmsepsd(denoiseout, EEG)
            acc = average_correlation_coefficient(denoiseout, EEG)
            print(f"MSE loss = {mse_loss}, RRMSET Loss ={rmset_loss}, RRMSE_spec Loss = {rmsepsd_loss}, ACC = {acc}")

    def __train_on_epoch(self, model, noiseEEG, EEG, optimizer, current_step):
        model.train()
        start = time.time()
        batch_size = self.batch_size
        batch_num = math.ceil(noiseEEG.shape[0]/batch_size)
        datanum = noiseEEG.shape[1]
        print(noiseEEG.shape, batch_num)
        train_loss = 0
        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for n_batch in range(batch_num):
                current_step += 1
                if n_batch == batch_num:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch :] , EEG[batch_size*n_batch :]
                else:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch : batch_size*(n_batch+1)] , EEG[batch_size*n_batch : batch_size*(n_batch+1)]
                
                if self.model_name == "LSTM_FFN":
                    noiseEEG_batch, EEG_batch = torch.reshape(torch.FloatTensor(noiseEEG_batch), (batch_size, datanum, 1)).to(self.device), torch.reshape(torch.FloatTensor(EEG_batch), (batch_size, datanum, 1)).to(self.device)
                else:
                    noiseEEG_batch, EEG_batch = torch.FloatTensor(noiseEEG_batch).to(self.device), torch.FloatTensor(EEG_batch).to(self.device)

                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    model(noiseEEG_batch)
                    denoiseout = model.predicted                        
                    mse_loss = denoise_loss_mse(denoiseout, EEG_batch)
                    loss = mse_loss
                    assert not torch.isnan(loss.data), "Loss is NaN"
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.data / float(batch_num)

                    pbar.update()
            pbar.close()
        end = time.time()
        print(f"Train loss: {train_loss}, time = {end-start}/per epoch")
        return {"epoch_train_loss": train_loss}, current_step
    
    def __val_on_epoch(self, model, noiseEEG, EEG):
        model.eval()
        batch_size = self.batch_size
        batch_num = math.ceil(noiseEEG.shape[0]/batch_size)
        datanum = noiseEEG.shape[1]
        valid_loss = 0
        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for n_batch in range(batch_num):
                if n_batch == batch_num:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch :] , EEG[batch_size*n_batch :]
                else:
                    noiseEEG_batch,EEG_batch =  noiseEEG[batch_size*n_batch : batch_size*(n_batch+1)] , EEG[batch_size*n_batch : batch_size*(n_batch+1)]
                if self.model_name == "LSTM_FFN":
                    noiseEEG_batch, EEG_batch = torch.reshape(torch.FloatTensor(noiseEEG_batch), (batch_size, datanum, 1)).to(self.device), torch.reshape(torch.FloatTensor(EEG_batch), (batch_size, datanum, 1)).to(self.device)
                else:
                    noiseEEG_batch, EEG_batch = torch.FloatTensor(noiseEEG_batch).to(self.device), torch.FloatTensor(EEG_batch).to(self.device)
                with torch.no_grad():
                    model(noiseEEG_batch)
                    denoiseout = model.predicted
                    mse_loss = denoise_loss_mse(denoiseout, EEG_batch)
                    loss = mse_loss 
                    assert not torch.isnan(loss.data), "Loss is NaN"
                    valid_loss += loss.data / float(batch_num)
                    pbar.update()
            pbar.close()
        print(f"Validation loss: {valid_loss}")
        return {"epoch_valid_loss": valid_loss}

    def _set_params(self, params):
        params = params.__dict__
        for k in params.keys():
            self.__setattr__(k, params[k])

    def __save_checkpoint(self, model, optimizer, force = False, purge_limit = 20):
        EXPERIMENT = f"{self.run_name}"
        model_path = f"../models/{EXPERIMENT}"
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
                    'epochs' : self.epoch,
                    'last_decay_epoch' : self.last_decay_epoch,
                    'learning_rate' : self.lr}

        torch.save({'net' : model.state_dict(), 'opt' : optimizer.state_dict(), 'train' : train_dict},
                model_path+'/'+output_filename)