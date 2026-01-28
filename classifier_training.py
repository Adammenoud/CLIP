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
    def __init__(self, model, loss, save_name ,name_training_loss="Cross-entropy training", name_val_loss="Cross-entropy validation", lr=1e-4):
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
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.lr)
        return optimizer
    
    def validation_step(self, batch):
        x =self(batch)
        target= self.get_target_from_batch(batch)
        print("x:", x.shape, "target:", target.shape)
        loss=self.loss(x, target)
        self.log(self.name_val_loss, loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        '''Implemented manually for consistency with previous saves'''
        os.makedirs(self.save_name, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.save_name, f"model.pt")) #overwrites the same file, so to avoid getting floded by saves
        torch.save(self.optimizers().state_dict(), os.path.join(self.save_name, "optim.pt"))
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



#--------------------------------------------------------------------------------------
if __name__ == "__main__":
 pass

    
