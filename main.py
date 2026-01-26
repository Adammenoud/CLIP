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
#local:
import utils
import nn_classes
import data_extraction
import hdf5
import train
import SDM_eval
#geoclip
from geoclip import LocationEncoder, GeoCLIP

# import heartrate
# heartrate.trace(browser=False, port=9999)

#seed
np.random.seed(48)
torch.manual_seed(48)
#--------------------------------------------------------------------------------------


wandb.init(project="contrastive_learning"
           
           
           )
nbr_epochs=120
device="cuda"
batch_size=4096 #"We use batch size |B| as 512 when training on full dataset. For data-efficient settings with 20%, 10%, and 5% of the data, we use |B| as 256, 256, and 128 respectively"
save_name="to_delete_test_compute"
data_path="Embeddings_and_dictionaries/arthropods/embeddings_inaturalist_FR_arthropods.h5"
#data_path="embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/difference_embeddings.h5"
#"embeddings_data_and_dictionaries/Embeddings/swiss_bioclip_embeddings/swiss_data_bioclip.h5"


#Fetching data: see "data_extraction.py"



#"embeddings_data_and_dictionaries/Embeddings/swiss_bioclip_embeddings/swiss_dictionary"
dictionary=pd.read_csv("Embeddings_and_dictionaries/arthropods/dictionary_inaturalist_FR_arthropods")



print("spliting data")
dataloader, test_dataloader =utils.dataloader_emb(data_path,
                                                  batch_size=batch_size, 
                                                  shuffle=True,
                                                  train_ratio=0.8, 
                                                  sort_duplicates=True, 
                                                  dictionary=dictionary,
                                                  drop_last=True,
                                                  dataset="ordered_HDF5Dataset"
                                                  )
dim_hidden=768
dim_output=512
image_encoder=nn.Sequential( nn.Linear(768, dim_hidden),nn.ReLU(),nn.Linear(dim_hidden, dim_output) )
model= GeoCLIP(from_pretrained=False,queue_size=1)
model.image_encoder=image_encoder



print("training")
model=train.train(
            model,
            nbr_epochs,
            dataloader,
            batch_size,
            lr=1e-4,    
            device="cuda",
            save_name=save_name,
            saving_frequency=1,
            nbr_checkppoints=30,   ########
            test_dataloader=test_dataloader,
            test_frequency=1,
            nbr_tests=10,
            modalities=["images","coords"], #either "images","coords","NCEAS" or "species"
            dictionary=None,  #if one wants to use a different dictionary in the "species" case
            )




#tensorboard --logdir=runs
#https://www.gbif.org/occurrence/
#CUDA_VISIBLE_DEVICES=1 python train.py
#CUDA_VISIBLE_DEVICES=1 python -m pdb your_script.py


