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
import train_contrastive
import SDM_eval
import train_classifier
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
cfg.run_name_clean=static_cfg["run_name"] #saves the basic name also
run.name=run_name #same format on wandb
if cfg.model_name == "classifier": #dim_output for classifier
    model_cfg = cfg.model_params['classifier']
    if model_cfg["target"]== "embeddings":
        dim_output=768
    else:
        dictionary=pd.read_csv(static_cfg['paths'][cfg.dataset]['dict'])
        dim_output=len(dictionary[model_cfg['class_column']].unique())
    cfg.model_params['classifier']['dim_output'] = dim_output
#dumps the config
file_path = Path(f"Model_saves/{run_name}/config.yaml")
file_path.parent.mkdir(parents=True, exist_ok=True)
cfg_dict = dict(cfg)
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
difference=None #compatibility
dataloader, test_dataloader =utils.dataloader_emb(data_path,
                                                  batch_size=cfg.training['batch_size'], 
                                                  shuffle=cfg.shuffle,
                                                  train_ratio=cfg.train_ratio, 
                                                  sort_duplicates=cfg.sort_duplicates, 
                                                  dictionary=dictionary,
                                                  drop_last=cfg.drop_last,
                                                  dataset_type=cfg.dataset_type,
                                                  vectors_name=cfg.vectors_name,
                                                  spe_data_path=static_cfg['paths'][dataset]["specie_data"],
                                                  difference=static_cfg["difference"]
                                                  )
if cfg.drop_high_freq:
    sigma=[2**0, 2**4]
else:
    sigma=[2**0, 2**4, 2**8]


if cfg.model_name=="contrastive":
    model_cfg = cfg.model_params['contrastive']
    model=nn_classes.DoubleNetwork_V2(LocationEncoder(sigma=sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder']),dim_in=static_cfg["model_params"]["contrastive"]["embedding_size"])
    print("training")
    model=train_contrastive.train(
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
elif cfg.model_name== "classifier":
    model_cfg = cfg.model_params['classifier']
    if model_cfg['loss'] == "cross_entropy":
        loss=F.cross_entropy
    elif model_cfg['loss'] == "MSE":
        loss=F.mse_loss
    if model_cfg['n_species'] is None:
        n_species=len(dictionary[model_cfg['class_column']].unique())
        print("n_species:" , n_species)
    else:
        pass #could filter species here, maybe later
    #wandb
    wandb_logger = WandbLogger(
    project=static_cfg['wandb']['project'],
    entity=static_cfg['wandb']['entity'],
    name=run_name)
    #model
    dim_output = cfg.model_params['classifier']['dim_output']    
    model = nn.Sequential(
                LocationEncoder(sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder']),
                nn.ReLU(),
                nn.Linear(512,dim_output)
                    )

    #(put model in wrapper)
    if model_cfg["target"]== "embeddings":
        model=train_classifier.Embeddings_Classifier_train(
            model, 
            f"Model_saves/{run_name}",
            loss=loss,
            name_training_loss="MSE on training set", 
            name_val_loss="MSE on validation set", 
            lr=1e-4
        )
    elif model_cfg["target"]== "specie_names":
        model=train_classifier.Classifier_train(model=model,
                    dictionary=dictionary,
                    save_name=f"Model_saves/{run_name}",
                    lr=cfg.training['lr'],
                    loss=loss,
                    name_training_loss=f"{model_cfg['loss']} training", 
                    name_val_loss=f"{model_cfg['loss']} validation",
                    class_name=model_cfg['class_column']
                    )
    else:
        raise ValueError(f"target '{model_cfg['target']}' invalid: select 'embeddings' or 'specie_names'")
    
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

