#imports:
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import wandb
import os
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from dotenv import load_dotenv
#local:
import utils
import datasets
import nn_classes
import train_contrastive
import train_classifier
#geoclip
from geoclip import LocationEncoder

#seed
np.random.seed(48)
torch.manual_seed(48)
#--------------------------------------------------------------------------------------
#static hyperparameters (those in the config.yaml, not the ones in the sweep_config.yaml):
with open("config.yaml") as f:
    static_cfg = yaml.safe_load(f)
#wandb API:
load_dotenv()  # loads .env if present
WANDB_API_KEY = os.getenv("WANDB_API_KEY") #otherwise: can also use use export WANDB_API_KEY=<your_api_key> in the terminal
if WANDB_API_KEY is None:
    raise RuntimeError("WANDB_API_KEY is not set (neither in .env nor in environment)")

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
        dataframe=pd.read_csv(static_cfg['paths'][cfg.dataset]['dict'])
        dim_output=len(dataframe[model_cfg['class_column']].unique())
    cfg.model_params['classifier']['dim_output'] = dim_output
#dumps the config
utils.dump_config(cfg, run_name)



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


dataframe=pd.read_csv(dict_path)


print("spliting data")
difference=None #compatibility
dataloader, test_dataloader, dataset_type =datasets.dataloader_factory(data_path,
                                                  dict(cfg), 
                                                  dataframe=dataframe,
                                                  )
cfg.dataset_type=dataset_type #saves the dataset type in the config for debugging/clarity

if cfg.drop_high_freq:
    sigma=[2**0, 2**4]
else:
    sigma=[2**0, 2**4, 2**8]

#image embedding size infered from the data
embedding, _, _ = dataloader.dataset[0]
cfg.dataloader_embedding_size = embedding.shape[0]
utils.dump_config(cfg, run_name)#to overwrite the inferred value



# ----------------------------
#Model construction and training:
# ----------------------------
if cfg.model_name=="contrastive":
    #build model
    location_encoder=LocationEncoder(sigma=sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder'])
    model=nn_classes.DoubleNetwork_V2(location_encoder,
                                      dim_in=cfg.dataloader_embedding_size,
                                      dim_output=cfg.model_params['contrastive']['contrastive_embedding_size'],
                                    )
    #train
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
                modalities=cfg.model_params['contrastive']['modalities'], 
                dataframe=dataframe,
                )
    

elif cfg.model_name== "classifier":
    model_cfg = cfg.model_params['classifier']
    if model_cfg['loss'] == "cross_entropy":
        loss=F.cross_entropy
    elif model_cfg['loss'] == "MSE":
        loss=F.mse_loss
    if model_cfg['n_species'] is None:
        n_species=len(dataframe[model_cfg['class_column']].unique())
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
        name_val_loss="MSE on validation set"
        model=train_classifier.Embeddings_Classifier_train(
            model, 
            f"Model_saves/{run_name}",
            loss=loss,
            name_training_loss="MSE on training set", 
            name_val_loss=name_val_loss, 
            lr=1e-4
        )
    elif model_cfg["target"]== "specie_names":
        name_val_loss=f"{model_cfg['loss']} validation"
        model=train_classifier.Classifier_train(model=model,
                    dataframe=dataframe,
                    save_name=f"Model_saves/{run_name}",
                    lr=cfg.training['lr'],
                    loss=loss,
                    name_training_loss=f"{model_cfg['loss']} training", 
                    name_val_loss=name_val_loss,
                    class_name=model_cfg['class_column']
                    )
    else:
        raise ValueError(f"target '{model_cfg['target']}' invalid: select 'embeddings' or 'specie_names'")
    
    #checkpoints
        #periodic checkpoints
    checkpoint_cb = ModelCheckpoint(
                            dirpath=f"Model_saves/{run_name}/checkpoints",
                            filename=f"{run_name}_checkpoint_{{epoch}}", 
                            save_top_k=-1,  # save all epochs; use 1 for only the best
                            )
        #best_model.pt checkpoint
    best_model_cb = ModelCheckpoint(
                            dirpath=f"Model_saves/{run_name}",   
                            filename="best_model",
                            monitor=name_val_loss,                 
                            mode="min",                          
                            save_top_k=1,                      
                        )
    #trainer
    trainer= L.Trainer(max_epochs=cfg.training['epochs'], 
                    accelerator=cfg.training['device'], 
                    devices=1, 
                    default_root_dir=f"Model_saves/{run_name}",   # overwritten by checkpoints?
                    log_every_n_steps=cfg.training['test_frequency'],
                    callbacks=[checkpoint_cb, best_model_cb],
                    limit_val_batches=cfg.training['nbr_tests'],
                    logger=wandb_logger,
                    deterministic=True
                    )  
    #train
    trainer.fit(model,dataloader,test_dataloader,)

else:
    raise Exception(f"invalid model_name: found '{cfg.model_name}' instead of 'contrastive' or 'classifier' ")

