#imports:
from typing import override
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
#local:
import utils
import nn_classes
import data_extraction
import hdf5
import torch
from geoclip import LocationEncoder
import os

from torchvision import models
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
from torchinfo import summary
import utils
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger

# import heartrate
# heartrate.trace(browser=False, port=9999)



class GeneralLoop(L.LightningModule):
    def __init__(self, model, loss, save_name ,name_training_loss="Cross-entropy", name_val_loss="CE on test set", lr=1e-4):
        super().__init__()
        self.model=model
        self.lr=lr
        self.loss=loss
        self.save_name=save_name
        self.name_training_loss=name_training_loss
        self.name_val_loss=name_val_loss
        

    def forward(self, batch):
        return self.model(batch)
    
    def get_target_from_batch(batch):
        return batch[1]  #coordinates here, if image, coords, idx convention. Overwrite if needed.
    
    def training_step(self, batch):
        # training_step defines the train loop.
        x=self(batch)
        target=self.get_target_from_batch(batch)  #Define this function according to the task
        loss= self.loss(x, target)  #Cross entropy, since one hot? Or BCEWithLogitsLoss?
        self.log(self.name_training_loss, loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.lr)
        return optimizer
    
    def validation_step(self, batch):
        x =self(batch)
        target= self.get_target_from_batch(batch)
        print("x:", x.shape, "target:", target.shape)
        loss=self.loss(x, target)
        self.log(self.name_val_loss, loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        '''Implemented manually for consistenyy with previous saves'''
        os.makedirs(self.save_name, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_name, f"model.pt")) #overwrites the same file, so to avoid getting floded by saves
        torch.save(self.optimizers().state_dict(), os.path.join(self.save_name, "optim.pt"))
        # hparams = {                             #json
        #         "save_name" : save_name,
        #         "saving_frequency" : saving_frequency,
        #         "learning_rate": lr,
        #         "batch_size": batch_size,
        #         "epochs": epochs,
        #         "current epoch (if stopped)" : ep,
        #         "saving_frequency":saving_frequency,
        #         "nbr_checkppoints":nbr_checkppoints, 
        #         "test_frequency":test_frequency,
        #         "nbr_tests": nbr_tests}
        # config_path = os.path.join(save_name, "hyperparameters.json")
        # with open(config_path, "w") as f:
        #     json.dump(hparams, f, indent=4)
    
    def every_n_epochs(self):
        pass #Checkpoints?
      
class Classifier_train(GeneralLoop):
    '''Overwirites the forward and target, training specific to classifier
        We train from the hp5 file, it would be more optimal from a dictionary directly
        Model's output must match the number of species
    '''
    def __init__(self, model, dictionary,save_name ,loss=F.cross_entropy,name_training_loss="Cross-entropy", name_val_loss="CE on test set", lr=1e-4):
        super().__init__(model, loss, save_name ,name_training_loss, name_val_loss, lr)

        self.dictionary=dictionary
        class_dict=data_extraction.species_classIdx_dict(dictionary, class_name="scientificName")
        self.register_buffer(
            "class_lookup",
            torch.tensor(
                class_dict["class_idx"].values,
                dtype=torch.long
            )
        )

    @override
    def forward(self, batch):
        _, coords, idx = batch
        return self.model(coords)
    
    @override
    def get_target_from_batch(self, batch):
        _, coords, idx = batch
        idx = idx.long().squeeze()
        return self.class_lookup[idx]



#seed
np.random.seed(48)
torch.manual_seed(48)
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    nbr_epochs=50
    device="cuda"
    batch_size=64
    save_name="classifier_gaussian_enc"
    data_path="embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioCLIP_full_dataset_embeddings.h5"
    #k=256 #number of species to keep. 256 is about 40% on the dataset
    fourier_dim=3*512

    #wandb
    os.environ['WANDB_API_KEY'] = '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'
    wandb.init(
    project="contrastive_learning",   # Replace with your project name
    entity="adammenoud",         # Replace with your W&B username or team
    name=save_name   # Optional: name your run
    )
    wandb_logger = WandbLogger(
    project="contrastive_learning",
    entity="adammenoud",
    name=save_name,
)

    dictionary=pd.read_csv("embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioclip_data_dictionary_all_taxons")
    #dataset=data_extraction.dictionary_data(dictionary)
    train_dataloader, test_dataloader =utils.dataloader_emb(data_path,batch_size=batch_size, shuffle=True,train_ratio=0.8, sort_duplicates=True, dictionary=dictionary)

    #filter: keeps to k species:
    # species_counts = dictionary['scientificName'].value_counts()
    # top_species = species_counts.head(k).index
    # dictionary = dictionary[dictionary['scientificName'].isin(top_species)]


    class_name="scientificName"
    n_species=len(dictionary[class_name].unique())



    model = nn.Sequential(
        nn_classes.AllGaussianEncoding(),
        nn_classes.MLP(
            in_dim=fourier_dim,
            hidden=[256, 256, 256],
            out_dim=n_species,
        )
    )

    model=Classifier_train(model=model,
                    dictionary=dictionary,
                    save_name=save_name,
                    lr=1e-4,
                    loss=F.cross_entropy,
                    name_training_loss="Cross-entropy training", 
                    name_val_loss="Cross-entropy validation"
                    )
    checkpoint_cb = ModelCheckpoint(
    dirpath=f"/home/adam/source/CLIP/Model_saves/{save_name}",
    filename=f"{save_name}_checkpoint_{{epoch}}", 
    save_top_k=-1,  # save all epochs; use 1 if you only want the best
    )

    trainer= L.Trainer(max_epochs=nbr_epochs, 
                    accelerator=device, 
                    devices=1, 
                    default_root_dir=f"experiments/{save_name}", 
                    log_every_n_steps=1,
                    callbacks=[checkpoint_cb],
                    limit_val_batches=10,
                    logger=wandb_logger)  
    trainer.fit(
        model,
        train_dataloader,
        test_dataloader,
    )
