import os
import shutil
import torch.optim as opt
import torch
import seaborn as sns
import io
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import datetime
from tensorboardX import SummaryWriter

def make_folder(filepath):
    if not os.path.isdir(filepath):
            os.makedirs(filepath)

def create_summary_writer(run_name):
    EXPERIMENT = f"{run_name}"
    MODEL_PATH = f"../Results/{EXPERIMENT}"
    writer = SummaryWriter(MODEL_PATH)
    print("Create tensorboard logger")
    return writer

def load_checkpoint(model_path, model, trainer, optimizer, input_filename='best', output='.'):
        '''
        Load checkpoint of network parameters and optimizer state
        required arguments:
            - input_filename (string): options:
                - path to input file. Must have .pth extension
                - 'best' (default) : checkpoint with lowest saved loss
                - 'recent'         : most recent checkpoint
                - 'longest'        : checkpoint after most training
            
            - dataset_name : if input_filename is not a path, must not be None
        '''
        if not os.path.exists(input_filename):
            # Get checkpoint filenames
            try:
                _,_,filenames = list(os.walk(model_path))[0]
            except IndexError:
                return
            
            if len(filenames) > 0:
            
                # Sort in ascending order
                filenames.sort()
                split_filenames = []
                # Split filenames into attributes (dates, epochs, loss)
                for fn in filenames:
                    if fn.endswith('.pth'):
                        split_filenames.append(os.path.splitext(fn)[0].split('_'))
                dates = [att[0] for att in split_filenames]
                epoch = [att[2] for att in split_filenames]
                loss  = [att[-1] for att in split_filenames]

                if input_filename == 'best':
                    # Get filename with lowest loss. If conflict, take most recent of subset.
                    loss.sort()
                    best = loss[0]
                    input_filename = [fn for fn in filenames if best in fn][-1]

                elif input_filename == 'recent':
                    # Get filename with most recent timestamp. If conflict, take first one
                    dates.sort()
                    recent = dates[-1]
                    input_filename = [fn for fn in filenames if recent in fn][0]

                elif input_filename == 'longest':
                    # Get filename with most number of epochs run. If conflict, take most recent of subset.
                    epoch.sort()
                    longest = epoch[-1]
                    input_filename = [fn for fn in filenames if longest in fn][-1]

                else:
                    assert False, 'input_filename must be a valid path, or one of \'best\', \'recent\', or \'longest\''
                    
            else:
                return

        assert os.path.splitext(input_filename)[1] == '.pth', 'Input filename must have .pth extension'

        # Load checkpoint
        state = torch.load(model_path+'/'+input_filename)

        # Set network parameters
        model.load_state_dict(state['net'])

        # Set optimizer state
        optimizer.load_state_dict(state['opt'])

        # Set training variables
        trainer.best                  = state['train']['best']
        trainer.train_loss_store      = state['train']['train_loss_store']
        trainer.valid_loss_store      = state['train']['valid_loss_store']
        trainer.full_loss_store       = state['train']['full_loss_store']
        trainer.epoch                 = state['train']['epochs']
        trainer.current_step          = state['train']['current_step']
        trainer.last_decay_epoch      = state['train']['last_decay_epoch']
        trainer.lr                    = state['train']['learning_rate']
        trainer.cost_weights          = state['train']['cost_weights']

        return model, trainer, optimizer

def gen_plot(epoch, dataset):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    sns.scatterplot(x="col1", y="col2", data=dataset)
    plt.title("Epoch num: " + str(epoch))

def plotTSNE(epoch, state):
    tsne = TSNE(n_components= 2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(state)
    dataset = pd.DataFrame()
    dataset['col1'] = tsne_results[:, 0]
    dataset['col2'] = tsne_results[:, 1]
    gen_plot(epoch, dataset)

    

