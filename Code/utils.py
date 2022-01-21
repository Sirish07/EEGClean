import os
import torch.optim as opt
import logging
import torch
import datetime
from tensorboardX import SummaryWriter

def make_optimizer(cfg, model):
    return opt.Adam(model.parameters(), lr = cfg.learning_rate, eps = cfg.epsilon, betas = cfg.betas)

def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def create_summary_writer(cfg):
    EXPERIMENT = f"{cfg.run_name}"
    MODEL_PATH = f"../models/{EXPERIMENT}"
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    writer = SummaryWriter(MODEL_PATH)
    print("Create tensorboard logger")
    return writer





