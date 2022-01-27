import numpy as np
from data_prepare import *
from baselines import *
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
    RESULTS = f"../results/{EXPERIMENT}"
    MODEL_PATH = f"../models/{EXPERIMENT}"
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    ################################################# Loading Datasets ##################################################
    file_location = cfg.data
    if cfg.noise_type == 'EOG':
        EEG_all = np.load( file_location + 'EEG_all_epochs.npy')
        noise_all = np.load( file_location + 'EOG_all_epochs.npy')
    elif cfg.noise_type == 'EMG':
        EEG_all = np.load( file_location + 'EEG_all_epochs_512hz.npy')                              
        noise_all = np.load( file_location + 'EMG_all_epochs_512hz.npy')
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = cfg.noise_type)
    
    if cfg.model_name == "FcNN":
        model = FcNN(cfg)
        model.apply(init_weights)
    elif cfg.model_name == "LSTM_FFN":
        model = LSTM_FFN(cfg)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    trainer = Trainer(cfg)

    if cfg.is_train=="True":
        make_folder(MODEL_PATH)
        make_folder(RESULTS)
        trainer.train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, optimizer)
    else: 
        print("===========================================")
        print("Retrieving best model")
        print("===========================================")
        model, trainer, optimizer = load_checkpoint(MODEL_PATH, model, trainer, optimizer)
    trainer.test(model, noiseEEG_test, EEG_test)