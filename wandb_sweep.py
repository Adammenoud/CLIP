#imports:
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torchinfo import summary
import sys
import wandb
import os
import yaml
#local:
import utils
import nn_classes
import data_extraction
import hdf5
import train
import SDM_eval
#geoclip
from geoclip import LocationEncoder, GeoCLIP

#seed
np.random.seed(48)
torch.manual_seed(48)
#--------------------------------------------------------------------------------------
#static hyperparameters:
with open("config.yaml") as f:
    static_cfg = yaml.safe_load(f)
#sweep hyperparameters:
#wandb
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')


run = wandb.init(
    project = static_cfg['wandb']['project'],
    entity  = static_cfg['wandb']['entity'],
    config=static_cfg,
    name=static_cfg['run_name']
)
cfg = wandb.config  # static defaults + sweep overrides





# Get dataset paths directly from YAML
dataset = cfg.dataset
try:
    data_path = static_cfg['paths'][dataset]['data']
    dict_path = static_cfg['paths'][dataset]['dict']
except KeyError:
    raise ValueError(f"Dataset '{dataset}' not found in config paths section.")



dictionary=pd.read_csv(dict_path)


print("spliting data")
dataloader, test_dataloader =utils.dataloader_emb(data_path,
                                                  batch_size=cfg.training['batch_size'], 
                                                  shuffle=cfg.shuffle,
                                                  train_ratio=cfg.train_ratio, 
                                                  sort_duplicates=cfg.sort_duplicates, 
                                                  dictionary=dictionary,
                                                  drop_last=cfg.drop_last,
                                                  dataset_type=cfg.dataset_type,
                                                  vectors_name=cfg.vectors_name
                                                  )
if cfg.drop_high_freq:
    sigma=[2**0, 2**4]
else:
    sigma=[2**0, 2**4, 2**8]

if cfg.model_name=="contrastive":
    model=nn_classes.DoubleNetwork_V2(LocationEncoder(sigma=sigma,from_pretrained=cfg.training['pretrained_geoclip_encoder']))
else:
    raise Exception(f"invalid model_name: found '{cfg.model_name}' instead of 'contrastive'")



print("training")
model=train.train(
            model,
            epochs=cfg.training['epochs'],
            dataloader=dataloader,
            batch_size=cfg.training['batch_size'],
            lr=cfg.training['lr'],    
            device=cfg.training['device'],
            save_name=cfg.run_name,
            saving_frequency=cfg.training['saving_frequency'],
            nbr_checkppoints=cfg.training['nbr_checkpoints'], 
            test_dataloader=test_dataloader,
            test_frequency=cfg.training['test_frequency'],
            nbr_tests=cfg.training['nbr_tests'],
            modalities=cfg.training['modalities'], 
            dictionary=dictionary,  #if one wants to use a different dictionary in the "species" case
            )


#tensorboard --logdir=runs
#https://www.gbif.org/occurrence/
#CUDA_VISIBLE_DEVICES=1 python main.py
#CUDA_VISIBLE_DEVICES=1 python -m pdb main.py


