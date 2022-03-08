import argparse
import numpy as np 
import torch
import torch.nn.functional as F

def parse_arguments():
    """Parsing input arguments"""

    parser = argparse.ArgumentParser("EEG Denoising using LFADS")

    # Training related
    parser.add_argument(
        "--run_name", default = "temp",
        help="name for experiment run"
    )

    parser.add_argument(
        "--model_name", default = "FcNN",
        help="name for experiment run"
    ) 

    parser.add_argument(
        "--is_train", default="True",
        help="If true, then train"
    )

    parser.add_argument(
        "--data", default="../data/",
        help="path to data folder"
    )

    # Input config
    parser.add_argument(
        "--inputs_dim", default=512,
        help="the dimensionality of the data (e.g. number of cells)"
    )

    parser.add_argument(
        "--T", default=1,
        help="number of time-steps in one sequence (i.e. one data point)"
    )

    parser.add_argument(
        "--keep_prob", default=0.97,
        help="keep probability for drop-out layers, if < 1 "
    )

    parser.add_argument(
        "--clip_val", default=10.0,
        help="clips the hidden unit activity to be less than this value"
    )

    parser.add_argument(
        "--max_norm", default=200.0,
        help="maximum gradient norm"
    )

    # optimizer hyperparameters

    parser.add_argument(
        "--lr", default=0.001,
        help="learning rate for ADAM optimizer"
    )

    parser.add_argument(
        "--eps", default=1e-08,
        help="epsilon value for ADAM optimizer"
    )

    parser.add_argument(
        "--betas", default=(0.9,0.999),
        help="beta values for ADAM optimizer"
    )

    parser.add_argument(
        "--device", default='cuda' if torch.cuda.is_available() else 'cpu',
        help="device to use"
    )

    parser.add_argument(
        "--save_variables", default=True,
        help="whether to save dynamic variables"
    )

    parser.add_argument(
        "--seed", default=None,
        help="Set Random Seed"
    )

    parser.add_argument(
        "--maxepochs", default=50,
        help="Total number of epochs"
    )

    parser.add_argument(
        "--batch_size", default=40,
        help="Input batch size"
    )

    parser.add_argument(
        "--combin_num", default=10,
        help="Combing EEG and noise x times"
    )

    parser.add_argument(
        "--noise_type", default="EOG",
        help="Noise type to be mixed"
    )
    return parser

def get_arguments():
    cfg = parse_arguments().parse_known_args()[0]
    for i, ii in cfg.__dict__.items():
        print(i, ii)
    return cfg
