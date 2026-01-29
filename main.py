#imports:
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torchinfo import summary
import sys
import wandb
import os
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
#local:
import utils
import nn_classes
import data_extraction
import hdf5
import train
import SDM_eval
import classifier_training
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
)
cfg = wandb.config  # static defaults + sweep overrides

# Get the run_name for checkpoints and tensorflow
#gets the parameters of the sweep, concatenate to the default name (if using wandb sweep)
with open("config_sweep.yaml") as f:
    sweep_cfg = yaml.safe_load(f)

sweep_keys = sweep_cfg["parameters"].keys()
run_name = utils.get_run_name(static_cfg["run_name"], cfg, sweep_keys)
run.name=run_name #same format on wandb
cfg_dict = dict(cfg)
#dumps the config
file_path = Path(f"Model_saves/{run_name}/config.yaml")
file_path.parent.mkdir(parents=True, exist_ok=True)
with open(file_path, "w") as f:
    yaml.dump(cfg_dict, f, default_flow_style=False)




# Get dataset paths directly from YAML
dataset = cfg.dataset
if cfg.use_species:
    data_type='specie_data'
else:
    data_type='data'

try:
    data_path = static_cfg['paths'][dataset][data_type]
    dict_path = static_cfg['paths'][dataset]['dict']
except KeyError:
    raise ValueError(f"Dataset '{dataset}' with '{data_type}' data type not found in config paths section.")

#reads group from sweep_config
if hasattr(cfg, "group_name"):
    wandb.run.group = cfg.group_name




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
    model_cfg = cfg.model_params['contrastive']
    model=nn_classes.DoubleNetwork_V2(LocationEncoder(sigma=sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder']))
    print("training")
    model=train.train(
                model,
                epochs=cfg.training['epochs'],
                dataloader=dataloader,
                batch_size=cfg.training['batch_size'],
                lr=cfg.training['lr'],    
                device=cfg.training['device'],
                save_name=run_name,
                saving_frequency=cfg.training['saving_frequency'],
                nbr_checkppoints=cfg.training['nbr_checkpoints'], 
                test_dataloader=test_dataloader,
                test_frequency=cfg.training['test_frequency'],
                nbr_tests=cfg.training['nbr_tests'],
                modalities=model_cfg['modalities'], 
                dictionary=dictionary,  #if one wants to use a different dictionary in the "species" case
                )
if cfg.model_name== "classifier":
    model_cfg = cfg.model_params['classifier']
    if model_cfg['loss'] == "cross_entropy":
        loss=F.cross_entropy
    if model_cfg['n_species'] is None:
        n_species=len(dictionary[model_cfg['class_column']].unique())
    else:
        pass #could filter species here, maybe later
    #wandb
    wandb_logger = WandbLogger(
    project=static_cfg['wandb']['project'],
    entity=static_cfg['wandb']['entity'],
    name=run_name)
    #model
    model = nn.Sequential(
                LocationEncoder(sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder']),
                nn.Linear(512,n_species)
                    )
    # model = nn.Sequential(
    #             nn_classes.MultilGaussianEncoding(encoded_size=model_cfg['encoded_size'],sigma=sigma),
    #             nn_classes.MLP(
    #                 in_dim=len(sigma)*model_cfg['encoded_size'],
    #                 hidden=model_cfg['hidden_layers'],
    #                 out_dim=n_species  
    #                 )
    # )
    #(put model in wrapper)
    model=classifier_training.Classifier_train(model=model,
                    dictionary=dictionary,
                    save_name=f"Model_saves/{run_name}",
                    lr=cfg.training['lr'],
                    loss=loss,
                    name_training_loss=f"{model_cfg['loss']} training", 
                    name_val_loss=f"{model_cfg['loss']} validation"
                    )
    #checkpoints
    checkpoint_cb = ModelCheckpoint(
                            dirpath=f"Model_saves/{run_name}/checkpoints",
                            filename=f"{run_name}_checkpoint_{{epoch}}", 
                            save_top_k=-1,  # save all epochs; use 1 for only the best
                            )
    #trainer
    trainer= L.Trainer(max_epochs=cfg.training['epochs'], 
                    accelerator=cfg.training['device'], 
                    devices=1, 
                    default_root_dir=f"Model_saves/{run_name}",   # overwritten by checkpoints?
                    log_every_n_steps=cfg.training['test_frequency'],
                    callbacks=[checkpoint_cb],
                    limit_val_batches=cfg.training['nbr_tests'],
                    logger=wandb_logger,
                    deterministic=True
                    )  
    #train
    trainer.fit(model,dataloader,test_dataloader,)

else:
    raise Exception(f"invalid model_name: found '{cfg.model_name}' instead of 'contrastive' or 'classifier' ")


#tensorboard --logdir=runs
#https://www.gbif.org/occurrence/
#CUDA_VISIBLE_DEVICES=1 python main.py
#CUDA_VISIBLE_DEVICES=1 python -m pdb main.py

