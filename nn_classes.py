import torch.nn as nn
import torch
import numpy as np
from geoclip import LocationEncoder
import yaml

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
    '''
    Contrastive model with a projection head on the image branch only, as in the geoCLIP paper.
    The LocationEncoder can be changed in the constructor.
    '''
    def __init__(self, pos_encoder=LocationEncoder(from_pretrained=False),dim_in=768,dim_hidden=768,dim_output=512,device="cuda",temperature=0.07):
        super().__init__()
        self.pos_encoder=pos_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature))).to(device) # This used to freeze the gradient: moving the tensor breaks its "parameter wrapper"...
        self.logit_scale = nn.Parameter(torch.tensor([1.0 / temperature], dtype=torch.float32, device=device).log())

        self.device=device


        self.lin1 = nn.Linear(dim_in, dim_hidden)
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
    


######-------RFF encoding, from the geoclip github
def sample_b(sigma: float, size: tuple):
    return torch.randn(size) * sigma

def gaussian_encoding(v, b):
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1) 

class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features
        More info about it (e.g. full docstring on the geoclip github)
    """
    def __init__(self, sigma = None,
                 input_size = None,
                 encoded_size = None,
                 b  = None):
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')

            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v):
        return gaussian_encoding(v, self.b)

class MultilGaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features.
       The dimension of output is encoded_size * len(sigma)
    """

    def __init__(self,
                 input_size=2,
                 encoded_size=256,
                 sigma=[2**0, 2**4, 2**8]):
        super().__init__()

        self.encoders = nn.ModuleList([
            GaussianEncoding(
                sigma=s,
                input_size=input_size,
                encoded_size=encoded_size
            )
            for s in sigma
        ])

    def forward(self, x):
        return torch.cat([encoder(x) for encoder in self.encoders], dim=1)



def build_model(cfg):
    '''
    Takes a config path and constructs the model (without loading the weights).
    Follows the logic of the main but do not have all the details making them trainable (run_name, dataframe etc...) 
    Use for downstream evaluation only
    drop_last is used in the case of a classifier predicting species only.
    '''
    model=None
    #shared parameters
    if cfg["drop_high_freq"]:
            sigma=[2**0, 2**4]
    else:
            sigma=[2**0, 2**4, 2**8]
    #model branching
    if cfg["model_name"]=="contrastive":
        try:
            embedding_size = cfg["model_params"]["contrastive"]["embedding_size"]
        except KeyError:
            embedding_size = 768
        model=DoubleNetwork_V2(LocationEncoder(sigma=sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder']),dim_in=embedding_size)
    elif cfg["model_name"]=="classifier":
        dim_output = cfg["model_params"]['classifier']['dim_output']    
        model = nn.Sequential(
                LocationEncoder(sigma,from_pretrained=cfg['model_params']['pretrained_geoclip_encoder']),
                nn.ReLU(),
                nn.Linear(512,dim_output)
                    )
    else: 
        print("could not build the model from the config")
    return model.to("cuda")

def load_model(model_path,config_path,device="cuda"):
    with open(config_path) as f:
        cfg=yaml.safe_load(f)
    model=build_model(cfg)
    state_dict = torch.load(model_path, map_location='cuda',weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]  #if it is a lightning checkpoint instead, gets the statedict
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()} #gets rid of the model. added by lightning
    state_dict.pop('class_lookup', None) # Remove 'class_lookup' if it exists, since we build the model without it (we do not use the last layer)
    model.load_state_dict(state_dict)
    model=model.to(device)
    return model

def load_pos_enc(model_path,config_path):
    model=load_model(model_path,config_path)
    with open(config_path) as f:
        cfg=yaml.safe_load(f)
    if cfg["model_name"]=="contrastive": #get the right encoder
        pos_encoder=model.pos_encoder
    else:
        pos_encoder=model[0] 
    return pos_encoder