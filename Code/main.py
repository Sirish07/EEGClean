import numpy as np
from lfads import LFADSNET
from data_prepare import *
from loss_function import *
from trainer import *
from utils import *
import config
import os

if __name__ == "__main__":
    print("===========================================")
    print("Retrieving default LFADS model hyperparameters")
    print("===========================================")

    cfg = config.get_arguments()
    EXPERIMENT = f"{cfg.run_name}"
    MODEL_PATH = f"../Results/{EXPERIMENT}"
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    ################################################# Loading Datasets ##################################################
    file_location = cfg.data
    EEG_all = np.load( file_location + 'EEG_all_epochs.npy')
    noise_all = np.load( file_location + 'EOG_all_epochs.npy')

    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = cfg.noise_type)
    model = LFADSNET(cfg).to(cfg.device) 
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr, betas = cfg.betas, eps = cfg.eps)
    trainer = Trainer(cfg)
    if cfg.is_train=="True":
        make_folder(MODEL_PATH)
        trainer.train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, optimizer)
    else: 
        print("===========================================")
        print("Retrieving best model")
        print("===========================================")
        model, trainer, optimizer = load_checkpoint(MODEL_PATH, model, trainer, optimizer)
    
    print("Starting Test")
    trainer.test(model, noiseEEG_test, EEG_test)