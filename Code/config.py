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

    # model hyperparameters
    parser.add_argument(
        "--g_dim", default=100,
        help="dimensionality of the generator"
    )

    parser.add_argument(
        "--u_dim", default=1,
        help="dimensionality of the inferred inputs to the generator"
    )

    parser.add_argument(
        "--factors_dim", default=20,
        help="dimensionality of the latent factors"
    )

    parser.add_argument(
        "--g0_encoder_dim", default=100,
        help="dimensionality of the encoder for the initial conditions for the generator"
    )

    parser.add_argument(
        "--c_encoder_dim", default=100,
        help="dimensionality of the encoder for the controller"
    )

    parser.add_argument(
        "--controller_dim", default=100,
        help="dimensionality of the controller"
    )

    parser.add_argument(
        "--g0_prior_logkappa", default=0.1,
        help="initial log-variance for the learnable prior over the initial generator state"
    )

    parser.add_argument(
        "--u_prior_logkappa", default=0.1,
        help="initial log-variance for the leanable prior over the inferred inputs to generator"
    )

    parser.add_argument(
        "--keep_prob", default=0.97,
        help="keep probability for drop-out layers, if < 1 "
    )

    parser.add_argument(
        "--clip_val", default=5.0,
        help="clips the hidden unit activity to be less than this value"
    )

    parser.add_argument(
        "--max_norm", default=200.0,
        help="maximum gradient norm"
    )

    # optimizer hyperparameters

    parser.add_argument(
        "--lr", default=0.00005,
        help="learning rate for ADAM optimizer"
    )

    parser.add_argument(
        "--eps", default=1e-08,
        help="epsilon value for ADAM optimizer"
    )

    parser.add_argument(
        "--betas", default=(0.5,0.9),
        help="beta values for ADAM optimizer"
    )

    parser.add_argument(
        "--lr_decay", default=0.95,
        help="learning rate decay factor"
    )

    parser.add_argument(
        "--lr_min", default=1e-5,
        help="minimum learning rate"
    )

    parser.add_argument(
        "--scheduler_on", default=True,
        help="apply scheduler if True"
    )

    parser.add_argument(
        "--scheduler_patience", default=6,
        help="number of steps without loss decrease before weight decay"
    )

    parser.add_argument(
        "--scheduler_cooldown", default=6,
        help="number of steps after weight decay to wait before next weight decay"
    )

    parser.add_argument(
        "--kl_weight_schedule_start", default=0,
        help="optimisation step to start kl_weight increase"
    )

    parser.add_argument(
        "--kl_weight_schedule_dur", default=10000, 
        help="number of optimisation steps to increase kl_weight to 1.0"
    )

    parser.add_argument(
        "--l2_weight_schedule_start", default=0,
        help="optimisation step to start l2_weight increase"
    )

    parser.add_argument(
        "--l2_weight_schedule_dur", default=10000,
        help="number of optimisation steps to increase l2_weight to 1.0"
    )

    parser.add_argument(
        "--l2_gen_scale", default=0.0,
        help="scaling factor for regularising l2 norm of generator hidden weights"
    )

    parser.add_argument(
        "--l2_con_scale", default=0.0,
        help="scaling factor for regularising l2 norm of controller hidden weights"
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
