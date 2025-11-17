#Test the clip model


import utils
import importlib
import torch.nn as nn
import torch
importlib.reload(utils)
from tqdm import tqdm

#for having the right model; print the model if unsurea about the layers
dim_fourier_encoding=64 #multiple of 4!!
dim_hidden=256
dim_emb=128 #this one is actually shared with img embeddings
device="cuda"
data_path="/home/adam/source/CLIP/full_dataset_embeddings.h5"
#dim image layer size: 
#As of now: linear from 768 to dim_emb. We could also have MLP if non-linearity needed

image_encoder=nn.Linear(768,dim_emb).to(device)
pos_encoder=utils.Fourier_MLP(original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=dim_emb).to(device)

model= utils.DoubleNetwork(image_encoder,pos_encoder).to(device)
model.load_state_dict(torch.load("/home/adam/source/CLIP/save_14_to_17-10-25/model.pt", weights_only=True))
model.eval()

nbr_iter=316628
mean_sim, mean_asim, std_sim, std_asim = utils.test_similarity(data_path,model, nbr_samples=2,device="cuda",nbr_iter=nbr_iter,plot_sims=True)
print("Mean:", mean_sim, mean_asim)
print("Std:", std_sim, std_asim)
#Mean: tensor(0.1810, device='cuda:0', grad_fn=<DivBackward0>) tensor(0.0827, device='cuda:0', grad_fn=<DivBackward0>)
#Std: tensor(0.0645, device='cuda:0', grad_fn=<SqrtBackward0>) tensor(0.0726, device='cuda:0', grad_fn=<SqrtBackward0>)

