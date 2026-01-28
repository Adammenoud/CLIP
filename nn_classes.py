
from torchvision import models
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
from torchinfo import summary
import utils
import numpy as np
from geoclip import LocationEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import open_clip
from geoclip import LocationEncoder

def get_resnet(dim_emb, device="cuda"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) #can also set weights to None
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, dim_emb) 
    model = model.to(device)
    return model  

class LocationEncoderAdapter(torch.nn.Module):
    '''Just so it also takes idx in the forward'''
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, coordinates, idx=None):
        return self.encoder(coordinates)


def detach_k(sim_matrix, k):
    # sim_matrix: (n, n)
    top_k_values, top_k_indices = torch.topk(sim_matrix, k, dim=1)

    #mask: true on the top k elements (for each sample), false otherwise
    mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    mask.scatter_(1, top_k_indices, True)

    sim_matrix = sim_matrix * (~mask) + sim_matrix.detach() * mask  #combine

    return sim_matrix


def CE_loss(logits, device="cuda", detach_k_top=None):
    '''logits of size (n, n), containing the cosine similarities (n would be batch size for CLIP)
    detach_detach_k_top should be a tuple (k_images,k_coords) of the number of gradient to detach for each modality
    '''
    # Create "labels"
    labels = torch.arange(logits.size(0)).to(device)

    loss_fct = nn.CrossEntropyLoss()

    if detach_k_top is not None:
        if detach_k_top[0] is not None:  # detach top-k from image-to-text logits
            logits_i2t = detach_k(logits, detach_k_top[0])  # detach along text dim
        else:
            logits_i2t = logits

        if detach_k_top[1] is not None:  # detach top-k from text-to-image logits
            logits_t2i = detach_k(logits.T, detach_k_top[1])  # detach along image dim
        else:
            logits_t2i = logits.T
    else:
        logits_i2t = logits
        logits_t2i = logits.T
    
    loss_i2t = loss_fct(logits_i2t, labels)
    loss_t2i = loss_fct(logits_t2i, labels)
    #average
    loss = (loss_i2t + loss_t2i) / 2
            
    return loss

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256, 256], out_dim=30, drop_last=False):
        super().__init__()
        self.drop_last=drop_last
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))  # linear output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if not self.drop_last:
            x=self.net(x)
        else:
            x = self.net[:-1](x)
        return x

class CustomMLP(nn.Module): #to encoode location (obsolete)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomMLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        return x
class GeoCLIP_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        """
        num_layers = number of hidden layers (each is Linear + ReLU)
        """
        super().__init__()

        layers = []

        # First layer maps input_dim → hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Add (num_layers-1) more hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Sequential hidden layers
        self.hidden = nn.Sequential(*layers)

        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    

class Fourier_MLP(nn.Module): #fourrier encoding followed by 2 layer MLP
    def __init__(self, original_dim, fourier_dim, hidden_dim, output_dim,scales=None, device="cuda"): #fourrier_dim must be a multiple of 2*originaldim
        '''The scales range from 1 to 2**-(fourrier_dim/4) if not specified'''
        super(Fourier_MLP, self).__init__()
        
        self.scales=scales
        self.device=device
        self.original_dim=original_dim
        self.fourier_dim=fourier_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.MLP=CustomMLP(fourier_dim, hidden_dim, output_dim)
        self.fourier_enc=Fourier_enc(fourier_dim)
        

    def forward(self, coord, idx=None):
        x=self.fourier_enc(coord,self.scales)
        x=self.MLP(x)
        return x
            
class Fourier_enc(nn.Module):
    def __init__(self, encoding_dim, device="cuda"):
        super(Fourier_enc, self).__init__()
        self.encoding_dim=encoding_dim
        self.device=device
    def forward(self, x, scales=None):
        x=x*2*torch.pi/360
        # x: (batch_size, 2)
        batch_size, input_dim = x.shape
        # Default scales if not provided
        if scales is None:
            # Half of fourier_dim for sin, half for cos
            scales = torch.arange(self.encoding_dim // (2 * input_dim), dtype=torch.float32).to(self.device)
        # Expand x to shape (batch_size, input_dim, 1)
        x_expanded = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        # Compute scaled inputs: (batch_size, input_dim, num_scales)
        scaled = x_expanded * (2.0 ** scales)  # broadcasting
        # Compute sine and cosine encodings
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        # Concatenate sin and cos along last dimension
        encoded = torch.cat([sin_enc, cos_enc], dim=-1)  # (batch_size, input_dim, num_scales*2)
         # Flatten last two dimensions to get final shape: (batch_size, fourier_dim)
        encoded = encoded.view(batch_size, -1)
        return encoded








class RFF_MLPs(nn.Module):
    '''as in geoCLIP, RFF encodding, '''
    def __init__(self, original_dim=2, fourier_dim=512, hidden_dim=1024, output_dim=512,M=3,sigma_min=1,sigma_max=256,number_layers=4,device="cuda"):
        '''fourrier_dim a multiple of original_dim (so even number)'''
        super(RFF_MLPs, self).__init__()
        self.device=device
        self.number_layers=number_layers
        self.original_dim=original_dim
        self.fourier_dim=fourier_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.M = M


        self.sigmas=self.compute_sigmas(sigma_min,sigma_max,M)

        self.MLP_list = nn.ModuleList([GeoCLIP_MLP(fourier_dim, hidden_dim, output_dim,num_layers=number_layers) for _ in range(M)]) #syntax for lists of modules

        self.R_list = []
        for i in range(M):#forum: register_buffer => Tensor which is not a parameter, but should be part of the modules state
            R = torch.randn((fourier_dim//original_dim, original_dim)) * self.sigmas[i]
            self.register_buffer(f'R_{i}', R)
            self.R_list.append(getattr(self, f'R_{i}'))

        
    def RFF(self, R, coords):
        '''coords: (batch_size, 2)'''
        R = R.to(self.device)
        x=2*torch.pi*R@coords.T #shape (fourrier_dim/2 , batch_size)
        cos=torch.cos(x) 
        sin=torch.sin(x)
        return torch.cat((cos, sin), dim=0).T # (batch_size, fourier_dim)

    def compute_sigmas(self, sigma_min, sigma_max, M):
        '''Formula for sigmas in geoCLIP paper'''
        i = torch.arange(1, M+1, dtype=torch.float32) 
        log_sigma_min = torch.log2(torch.tensor(sigma_min))
        log_sigma_max = torch.log2(torch.tensor(sigma_max))
        sigmas = 2 ** (log_sigma_min + (i - 1) * (log_sigma_max - log_sigma_min) / (M - 1))
        return sigmas
    def forward(self, coords):
        '''sums the output of all the MLPs'''
        out = torch.zeros(coords.size(0), self.output_dim, device=coords.device)
        for i in range(self.M):
            rff = self.RFF(self.R_list[i], coords)  # (batch_size, fourier_dim)
            out += self.MLP_list[i](rff)           # (batch_size, output_dim)
        return out


class DoubleNetwork(nn.Module):
    def __init__(self, image_encoder, pos_encoder,device="cuda",temperature=0.07):
        super().__init__()
        self.image_encoder=image_encoder
        self.pos_encoder=pos_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature))).to(device)
        self.device=device

    def forward(self, images, coordinates): #takes a batch of images and their labels
        #returns the normalized pos/image similarity matrix
        image_emb=self.image_encoder(images)
        pos_emb=self.pos_encoder(coordinates)
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        pos_emb = pos_emb / pos_emb.norm(dim=1, keepdim=True)
        # Compute cosine similarity (dot product here)
        logits = image_emb @ pos_emb.t()*self.logit_scale.exp()
        return logits
    
class DoubleNetwork_V2(nn.Module):
    def __init__(self, pos_encoder=LocationEncoder(from_pretrained=False),dim_hidden=768,dim_output=512,device="cuda",temperature=0.07):
        super().__init__()
        self.pos_encoder=pos_encoder
        #self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature))).to(device) # This used to freeze the gradient... Why?
        self.logit_scale = nn.Parameter(torch.tensor([1.0 / temperature], dtype=torch.float32, device=device).log())

        self.device=device


        self.lin1 = nn.Linear(768, dim_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(dim_hidden, dim_output)


    def forward(self, image, coordinates, idx=None): #takes a batch of images and their labels
        '''returns the normalized pos/image similarity matrix'''
        x = self.lin1(image)
        x = self.relu(x)
        image_emb = self.lin2(x) #see geoCLIP paper
        
        pos_emb=self.pos_encoder(coordinates)    #, idx)
        
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        pos_emb = pos_emb / pos_emb.norm(dim=1, keepdim=True)
        # Compute cosine similarity (dot product here, since vectors are normalized)
        logits = image_emb @ pos_emb.t()*self.logit_scale.exp()
        return logits
    


class Cov_Fourier_MLP(nn.Module):
    def __init__(self, fourier_dim, hidden_dim, output_dim,covariate_dim,scales=None, device="cuda",dictionary=None): #fourrier_dim must be a multiple of 2*originaldim
        '''The scales range from 1 to 2**-(fourrier_dim/4) if not specified'''
        super(Cov_Fourier_MLP, self).__init__()

        self.fourier_enc=Fourier_enc(fourier_dim)
        self.MLP=CustomMLP(fourier_dim+covariate_dim, hidden_dim, output_dim)
        self.scales=scales
        self.device=device
        self.dictionary=dictionary

    def forward(self, coords , idx_list=None, covariates=None, ):
        ''' covariates: (batch_size, covariate_dim)
            coords:     (batch_size, 2)             '''
        # if covariates is None:
        #     countries=utils.get_gbif_covariates(self.dictionary, idx_list, covariate_list=['countryCode'])
        #     is_in_CH=(countries=="CH")

        #     coords_numpy=coords.cpu().numpy()
        #     covariates_dict=utils.NCEAS_covariates(coords_numpy[:,1], coords_numpy[:,0]) #lon, lat
        #     keys = ["bcc","calc","ccc","ddeg","nutri","pday","precyy","sfroyy","slope","sradyy","swb","tavecc","topo"] #it is ugly code but it keeps the order, if needed
        #     cov_np = np.stack([covariates_dict[k] for k in keys], axis=1)  # (batch, num_covariates)
        #     covariates = torch.from_numpy(cov_np).float().to(self.device)
        #     covariates = torch.zeros(covariates.shape).to(self.device) #testing without covariates
        #     covariates = torch.normal(covariates, 1).to(self.device) #testing with random covariates
        if covariates is None:
            # 1. Determine which indices are in CH
            if idx_list is  not None:
                countries = utils.get_gbif_covariates(
                    self.dictionary, idx_list, covariate_list=['countryCode']
                )
                is_in_CH = (countries == "CH")
                is_in_CH_t = torch.from_numpy(is_in_CH).squeeze(1).to(self.device)  # (batch,)
            else:
                is_in_CH_t = np.array([True] * coords.shape[0])  # Default to all True if idx_list is None: assume all in CH
            

            # 2. Number of covariates
            keys = ["bcc","calc","ccc","ddeg","nutri","pday","precyy",
                    "sfroyy","slope","sradyy","swb","tavecc","topo"]
            num_cov = len(keys)
            batch_size = coords.shape[0]
            # 3. Initialize random covariates for everyone
            # covariates = torch.normal(
            #     mean=0.0,
            #     std=1.0,
            #     size=(batch_size, num_cov),
            #     device=self.device)
            covariates = torch.zeros(
                batch_size, num_cov, device=self.device
                )
            # 4. Compute real covariates ONLY for CH points
            if is_in_CH_t.any():
                coords_numpy = coords[is_in_CH_t].cpu().numpy()
                covariates_dict = utils.NCEAS_covariates(
                    coords_numpy[:, 1], coords_numpy[:, 0]  # lon, lat
                    )
                cov_np_CH = np.stack(
                    [covariates_dict[k] for k in keys], axis=1
                )
                cov_CH = torch.from_numpy(cov_np_CH).float().to(self.device)
                # 5. Overwrite random values with real covariates for CH rows
                covariates[is_in_CH_t] = cov_CH

        x=self.fourier_enc(coords, self.scales)
        x=torch.cat((x, covariates), dim=1) #(batch_size, covariate_dim+fourier_dim)
        x= self.MLP(x)
        return x
                          
class GeneralCrossEntropyLoss(nn.Module):
    """
    For 2 general probability distributions (in torch's CrossEntropy, one on them is assumed to be deterministic, i.e. just a label)
    """
    def __init__(self):
        super().__init__()
    def forward(self, logits, target):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * log_probs).sum(dim=1).mean()
        return loss
    
from geoCLIP_classes import GaussianEncoding

class AllGaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features.
        The dimension of output is encoded_size*3
    """

    def __init__(self,
                 input_size=2,
                 encoded_size=256):
        """
        Args:
        """
        super().__init__()
        self.g1=GaussianEncoding(sigma = 2**0,input_size= input_size,encoded_size=encoded_size)
        self.g2=GaussianEncoding(sigma = 2**4,input_size= input_size,encoded_size=encoded_size)
        self.g3=GaussianEncoding(sigma = 2**8,input_size= input_size,encoded_size=encoded_size)

    def forward(self, x):
        x=torch.cat((self.g1(x),self.g2(x),self.g3(x)), dim=1)
        return x