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

#local:
import utils
import data_extraction
import hdf5

#seed
np.random.seed(48)
torch.manual_seed(48)
#--------------------------------------------------------------------------------------
nbr_epochs=300
device="cuda"
batch_size=4096
save_name="save_14_to_17-10-25"
data_path="/home/adam/source/CLIP/full_dataset_embeddings.h5"
#Fetching data
'''
print("making dictionary")
path_multimedia="/home/adam/source/CLIP/data_plantnet_obsevations/multimedia.txt"
path_occurences="/home/adam/source/CLIP/data_plantnet_obsevations/occurrence.txt"
dictionary=data_extraction.get_dictionary(3167066, path_occurences, path_multimedia)#TYPO, 317066 instead of 3167066

print("downloading embeddings")
#make embeddings
data_extraction.download_emb(dictionary, dim_emb=768, output_dir="downloaded_embeddings")
print("download finished")
'''

#I got 3167055 embeddings: 11 failed,for instance
#Failed to download https://bs.plantnet.org/image/o/f20e1710cd58e9cc0f5816351719600476293e5b
#Failed to download https://bs.plantnet.org/image/o/21b326f1faf2216108193b5305bb05828ee4551d

#with multi-threading, 47min for 150000 images, *20 for the whole dataset


#TRAIN BABY



dataloader, _=utils.dataloader_emb(data_path,batch_size=batch_size)
#hyperparameters:
#pos. layer sizes
#we first upscale from 2 to dim_fourrier_encoding, with the fourrier encodding
dim_fourier_encoding=64 #multiple of 4!!
dim_hidden=256
dim_emb=128 #this one is actually shared with img embeddings

#dim image layer size: 
#As of now: linear from 768 to dim_emb. We could also have MLP if non-linearity needed


image_encoder=nn.Linear(768,dim_emb).to(device)
pos_encoder=utils.Fourier_MLP(original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=dim_emb).to(device)

model= utils.DoubleNetwork(image_encoder,pos_encoder).to(device)

model=utils.train(model, nbr_epochs, dataloader,batch_size,save_name=save_name).to(device)
#tensorboard --logdir=Nov14_10-46-22_Chronos 