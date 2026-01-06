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
#local:
import utils
import nn_classes
import data_extraction
import hdf5
import train
#geoclip
from geoclip import LocationEncoder

# import heartrate
# heartrate.trace(browser=False, port=9999)

#seed
np.random.seed(48)
torch.manual_seed(48)
#--------------------------------------------------------------------------------------
nbr_epochs=120
device="cuda"
batch_size=4096 #"We use batch size |B| as 512 when training on full dataset. For data-efficient settings with 20%, 10%, and 5% of the data, we use |B| as 256, 256, and 128 respectively"
save_name="test_modality_rework_image_VS_coords"
data_path="embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioCLIP_full_dataset_embeddings.h5"

'''
#Fetching data
print("making dictionary")
path_occurences="embeddings_data_and_dictionaries/data_plantnet_obsevations/occurrence.txt" 
path_multimedia="embeddings_data_and_dictionaries/data_plantnet_obsevations/multimedia.txt"
taxa_cols= ['kingdom','phylum','class', 'order','family','genus','species']
extra_occ_columns=['scientificName','countryCode','higherClassification','vernacularName']+ taxa_cols
dictionary=data_extraction.get_dictionary(3167066, path_occurences, path_multimedia,extra_occ_columns=extra_occ_columns)#3167066 rows
#create a new column with the full bioclip name (not with the vernacular name)
taxa_cols.remove("genus") #For some reason, species contains genus+species ... removing genus gives the correct full taxon
dictionary['taxa_bioclip'] = (
    dictionary[taxa_cols] 
    .astype(str)
    .replace('nan', '')
    .apply(lambda row: ' '.join([v for v in row if v]), axis=1) #separated by only a space, like in bioclip
)
#save
dictionary.to_csv("embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioclip_data_dictionary_all_taxons")
'''

#"embeddings_data_and_dictionaries/Embeddings/swiss_bioclip_embeddings/swiss_dictionary"
dictionary=pd.read_csv("embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioclip_data_dictionary_all_taxons")
'''
print("downloading embeddings")
#make embeddings
data_extraction.download_emb(dictionary, dim_emb=768, output_dir="downloaded_embeddings_Switwerland")
print("download finished")
'''
#I got 3167055 embeddings: 11 failed,for instance:
#Failed to download https://bs.plantnet.org/image/o/f20e1710cd58e9cc0f5816351719600476293e5b
#Failed to download https://bs.plantnet.org/image/o/21b326f1faf2216108193b5305bb05828ee4551d

#with multi-threading, 47min for 150000 images, *20 for the whole dataset


#TRAIN BABY
print("spliting data")
dataloader, test_dataloader =utils.dataloader_emb(data_path,batch_size=batch_size, shuffle=True,train_ratio=0.8, sort_duplicates=True, dictionary=dictionary)
#hyperparameters:
#pos. layer sizes
#we first upscale from 2 to dim_fourrier_encoding, with the fourrier encodding
 #this one is actually shared with img embeddings

#dim image layer size: 
#As of now: linear from 768 to dim_emb. We could also have MLP if non-linearity needed
#GeoCLIP: followed by "two trainable linear layers h1 and h2 having dimensions of 768 and 512 respectively"
'''Basic architechture
dim_fourier_encoding=64 #multiple of 4!!
dim_hidden=256
dim_emb=128
image_encoder=nn.Linear(768,dim_emb)
pos_encoder=utils.Fourier_MLP(2,dim_fourier_encoding,hidden_dim=dim_hidden,output_dim=dim_emb)
model= utils.DoubleNetwork(image_encoder,pos_encoder).to(device)
print("training")
model=utils.train(
            model,
            nbr_epochs,
            dataloader,
            batch_size,
            lr=1e-4,
            device="cuda",
            save_name=save_name,
            saving_frequency=1,
            nbr_checkppoints=60, 
            test_dataloader=test_dataloader,
            test_frequency=1,
            nbr_tests=10
            )
            '''

# #tensorboard --logdir=runs
# device="cuda"
# data_path="embeddings_data_and_dictionaries/bioCLIP_full_dataset_embeddings.h5"
# fourier_dim=64
# covariate_dim=13
# hidden_dim=256
fourier_dim=104    #13*8
hidden_dim=256
dim_emb=128
cov_dim=13
pos_encoder = nn_classes.Fourier_MLP(original_dim=cov_dim,fourier_dim=fourier_dim, hidden_dim=hidden_dim, output_dim=dim_emb,scales=None)

#pos_encoder= nn_classes.RFF_MLPs(original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=dim_emb,M=8,sigma_min=2,sigma_max=256, number_layers=4)
#pos_encoder=utils.RFF_MLPs( original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=512,M=8,sigma_min=1,sigma_max=256).to(device)

model= nn_classes.DoubleNetwork_V2(pos_encoder,dim_hidden=768,dim_output=dim_emb).to(device)


#pos_encoder= nn_classes.Cov_Fourier_MLP(fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=dim_emb, covariate_dim=covariate_dim)
#pos_encoder=nn_classes.Fourier_MLP(original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=dim_emb)

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
            )


#https://www.gbif.org/occurrence/


