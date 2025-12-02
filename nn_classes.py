
from torchvision import models
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
from torchinfo import summary


def get_resnet(dim_emb, device="cuda"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) #can also set weights to None
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, dim_emb) 
    model = model.to(device)
    return model  

def CE_loss(logits, device="cuda"):
    '''logits of size (n, n), containing the cosine similarities (n would be batch size for CLIP)'''
    # Create labels
    labels = torch.arange(logits.size(0)).to(device)

    loss_fn = nn.CrossEntropyLoss()

    # Image-to-Text
    loss_i2t = loss_fn(logits, labels)

    # Text-to-Image (transpose logits)
    loss_t2i = loss_fn(logits.T, labels)

    # Total symmetric loss
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss


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

        

    def forward(self, coord):
        x=self.fourier_enc(coord,self.scales)
        x=self.MLP(x)
        return x
            

    def fourier_enc(self, x, scales=None):
        # x: (batch_size, 2)
        batch_size, input_dim = x.shape
        # Default scales if not provided
        if scales is None:
            # Half of fourier_dim for sin, half for cos
            scales = torch.arange(self.fourier_dim // (2 * input_dim), dtype=torch.float32).to(self.device)
        # Expand x to shape (batch_size, input_dim, 1)
        x_expanded = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        # Compute scaled inputs: (batch_size, input_dim, num_scales)
        scaled = x_expanded / (2.0 ** scales)  # broadcasting
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
    def __init__(self, pos_encoder,dim_hidden=768,dim_output=512,device="cuda",temperature=0.07):
        super().__init__()
        self.pos_encoder=pos_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature))).to(device)
        self.device=device


        self.lin1 = nn.Linear(768, dim_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(dim_hidden, dim_output)

    def image_encode(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


    def forward(self, image, coordinates): #takes a batch of images and their labels
        '''returns the normalized pos/image similarity matrix'''
        x = self.lin1(image)
        x = self.relu(x)
        image_emb = self.lin2(x) #see geoCLIP paper
        
        pos_emb=self.pos_encoder(coordinates)
        
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        pos_emb = pos_emb / pos_emb.norm(dim=1, keepdim=True)
        # Compute cosine similarity (dot product here)
        logits = image_emb @ pos_emb.t()*self.logit_scale.exp()
        return logits
    



                          

def train(doublenetwork,
          epochs,
          dataloader,
          batch_size,
          lr=1e-4,
          device="cuda",
          save_name=None,
          saving_frequency=1,
          nbr_checkppoints=None, 
          test_dataloader=None,
          test_frequency=1,
          nbr_tests=10
          ):
    '''nbr_chepoints < epochs, please'''
    #### initialization
    doublenetwork = doublenetwork.to(device)
    doublenetwork.train()
    writer =  SummaryWriter(log_dir=os.path.join("runs", f"{save_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))
    optimizer=torch.optim.AdamW(doublenetwork.parameters(),lr=lr)
    l = len(dataloader)
    current_checkpoint=1
    #### training loop
    for ep in range(epochs):
        pbar = tqdm(dataloader)
        for i, (images, labels,_) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            logits=doublenetwork(images,labels) #logits of cosine similarities
            loss=CE_loss(logits,device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(CE=loss.item())
            writer.add_scalar("Cross-entropy", loss.item(), global_step=ep * l + i)
    ################################# logs and savings
        if ep % saving_frequency == 0 and (save_name is not None):
            os.makedirs(save_name, exist_ok=True)
            torch.save(doublenetwork.state_dict(), os.path.join(save_name, f"model.pt")) #overwrites the same file, so to avoid getting floded by saves
            torch.save(optimizer.state_dict(), os.path.join(save_name, f"optim.pt"))
            hparams = {                             #json
                    "save_name" : save_name,
                    "saving_frequency" : saving_frequency,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "current epoch (if stopped)" : ep,
                    "saving_frequency":saving_frequency,
                    "nbr_checkppoints":nbr_checkppoints, 
                    "test_frequency":test_frequency,
                    "nbr_tests": nbr_tests}
            config_path = os.path.join(save_name, "hyperparameters.json")
            with open(config_path, "w") as f:
                json.dump(hparams, f, indent=4)
            summary_str = str(summary(doublenetwork, input_size=[images.shape, labels.shape])) #txt
            with open(os.path.join(save_name, "model_summary.txt"), "w") as f:
                f.write(summary_str)
        if save_name is not None and nbr_checkppoints is not None and ep % (epochs//nbr_checkppoints)==0 and ep!= 0:
            checkpoint_name = f"{save_name}_checkpoint_{current_checkpoint}"
            os.makedirs(checkpoint_name, exist_ok=True)
            torch.save(doublenetwork.state_dict(), os.path.join(checkpoint_name, f"model.pt")) #overwrites the same file, so to avoid getting floded by saves
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_name, f"optim.pt"))
            current_checkpoint +=1
        if test_dataloader is not None and ep % test_frequency == 0:
            doublenetwork.eval()
            with torch.no_grad():
                total_loss = 0.0
                for _ in range(nbr_tests):
                    test_images, test_labels, _ = next(iter(test_dataloader))
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    test_logits = doublenetwork(test_images, test_labels)
                    test_loss = CE_loss(test_logits, device=device)
                    total_loss += test_loss.item()

                avg_loss = total_loss / nbr_tests
                writer.add_scalar("CE on test set", avg_loss, global_step=ep)

        doublenetwork.train()
    #######################

    doublenetwork.eval()
    writer.close()
    return doublenetwork
