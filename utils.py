import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset,random_split, Subset
import data_extraction
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json
from pyproj import Transformer
import h5py


def noise(input, std):
    noise = np.random.normal(0, std, input.shape)  # Gaussian noise
    noisy = input + noise
    return noisy

def show_image(img): #torch, numpy or PIL
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
    plt.figure(figsize=(4,4))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # hide axes
    plt.show()


class ImageFromDictionary(Dataset):
    def __init__(self, dictionary, image_folder_path ,label_names=('decimalLatitude','decimalLongitude'), transform=None):
        """
        dictionary must have column 'gbifID',
        and columns in label_names for the labels.

        It uses the same idx than the one to name the image files, so please do not re-index the dictionary

        image_folder_path must be the path to a folder containing the images, Not like ImageFolder!
        """
        self.dictionary = dictionary
        self.label_names = label_names
        self.transform = transform
        self.image_folder_path=image_folder_path

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        img_path = f"{self.image_folder_path}/{idx}_{self.dictionary.loc[idx, 'gbifID']}.jpg"
        labels = self.dictionary.loc[idx, list(self.label_names)].to_numpy(dtype='float32')
        labels = torch.tensor(labels)       

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, labels
    

def dataloader_img(dictionary,image_folder_path,image_size,batch_size,shuffle=True):
        transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),  #as in HF
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #to [-1,1]
        ])
        dataset = ImageFromDictionary(dictionary, image_folder_path,label_names=('decimalLatitude','decimalLongitude'), transform=transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

'''
def dataloader_emb(file_path,batch_size,shuffle=False,train_proportion=0.8): #from a h5 file
    dataset=data_extraction.HDF5Dataset(file_path)
    train_size = int(train_proportion * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(48)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
    '''

def dataloader_emb(file_path,batch_size,shuffle=False,train_ratio=0.8, sort_duplicates=False, dictionary=None): #from a h5 file
    '''solves to the duplication problem'''

    dataset=data_extraction.HDF5Dataset(file_path)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(48)
    if sort_duplicates:
        train_dataset, test_dataset=group_split(dataset,file_path, dictionary,generator, train_ratio)
    else:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
def group_split(dataset, file_path, dictionary,generator, train_ratio=0.8):
    """Split dataset into train/test SUCH THAT samples with the same ID stay together.
    Technically, there may be some variance to the training set proportion since not all Gbif indices have the same number of images."""
    n = len(dataset)
    with h5py.File(file_path, "r") as f:
        dict_indices = f["dict_idx"][:].flatten()
    # 1 — Extract IDs for each sample
    ids = dictionary.loc[dict_indices, "gbifID"].values  
    # 2 — Convert unique IDs to integers
    unique_ids = sorted(set(ids))
    id_to_int = {id_: i for i, id_ in enumerate(unique_ids)}
    int_ids = torch.tensor([id_to_int[id_] for id_ in ids])
    # 3 — Shuffle group IDs
    num_groups = len(unique_ids)
    perm = torch.randperm(num_groups, generator=generator)
    # 4 — Split group IDs
    test_size = int(num_groups * (1-train_ratio))
    test_groups = set(perm[:test_size].tolist())
    train_groups = set(perm[test_size:].tolist())
    # 5 — Assign samples
    train_indices = [i for i in range(n) if int_ids[i].item() in train_groups]
    test_indices  = [i for i in range(n) if int_ids[i].item() in test_groups]
    # 6 — Build subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


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
class GeoCLIP_MLP(nn.Module): #to encoode location (obsolete)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeoCLIP_MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        x = self.relu4(x)
        x = self.lin4(x)
        x = self.output(x)

        return x
    
    

class Fourier_MLP(nn.Module): #fourrier encoding followed by 2 layer MLP
    def __init__(self, original_dim, fourier_dim, hidden_dim, output_dim, device="cuda"): #fourrier_dim must be a multiple of 2*originaldim
        super(Fourier_MLP, self).__init__()

        self.device=device
        self.original_dim=original_dim
        self.fourier_dim=fourier_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.MLP=CustomMLP(fourier_dim, hidden_dim, output_dim)

        

    def forward(self, coord):
        x=self.fourier_enc(coord)
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
    def __init__(self, original_dim=2, fourier_dim=512, hidden_dim=1024, output_dim=512,M=3,sigma_min=1,sigma_max=256,device="cuda"):
        '''fourrier_dim a multiple of original_dim (so even number)'''
        super(RFF_MLPs, self).__init__()
        self.device=device
        self.original_dim=original_dim
        self.fourier_dim=fourier_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.M = M


        self.sigmas=self.compute_sigmas(sigma_min,sigma_max,M)

        self.MLP_list = nn.ModuleList([GeoCLIP_MLP(fourier_dim, hidden_dim, output_dim) for _ in range(M)]) #syntax for lists of modules

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
    writer =  SummaryWriter()
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
            hparams = {
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


import time
import torch
from tqdm import tqdm

def minitrain(doublenetwork, dataloader, device="cuda"):
    """
    Run a few batches and print time per operation.
    Useful for profiling bottlenecks.
    """
    doublenetwork = doublenetwork.to(device)
    doublenetwork.train()
    
    optimizer = torch.optim.AdamW(doublenetwork.parameters(), lr=1e-4)
    
    # Only 1 epoch, 5 batches
    for ep in range(1):
        pbar = tqdm(dataloader)
        for i, (images, labels, _) in enumerate(pbar):
            if i >= 5:  # only profile first 5 batches
                break

            t0 = time.time()
            # Move data to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t1 = time.time()
            print(f"[Batch {i}] to(device) time: {t1-t0:.4f}s")

            # Forward pass
            logits = doublenetwork(images, labels)
            t2 = time.time()
            print(f"[Batch {i}] forward time: {t2-t1:.4f}s")

            # Loss computation
            loss = CE_loss(logits, device=device)
            t3 = time.time()
            print(f"[Batch {i}] loss computation time: {t3-t2:.4f}s")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            t4 = time.time()
            print(f"[Batch {i}] backward time: {t4-t3:.4f}s")

            # Optimizer step
            optimizer.step()
            t5 = time.time()
            print(f"[Batch {i}] optimizer step time: {t5-t4:.4f}s")

            # Logging (optional)
            # writer.add_scalar("Cross-entropy", loss.item(), global_step=ep * len(dataloader) + i)
            t6 = time.time()
            print(f"[Batch {i}] logging time: {t6-t5:.4f}s")

            print(f"[Batch {i}] total batch time: {t6-t0:.4f}s\n")

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
    def __init__(self, pos_encoder,device="cuda",temperature=0.07):
        super().__init__()
        self.pos_encoder=pos_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/temperature))).to(device)
        self.device=device

        self.lin1=nn.Linear(768,768)#this is the image encoder
        self.relu=nn.ReLU()
        self.lin2=nn.Linear(768,512)

    def forward(self, image, coordinates): #takes a batch of images and their labels
        '''returns the normalized pos/image similarity matrix'''
        
        image_emb=self.lin1(image)   #see geoCLIP paper
        image_emb=self.relu(image_emb)
        image_emb=self.lin2(image_emb)


        pos_emb=self.pos_encoder(coordinates)
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        pos_emb = pos_emb / pos_emb.norm(dim=1, keepdim=True)
        # Compute cosine similarity (dot product here)
        logits = image_emb @ pos_emb.t()*self.logit_scale.exp()
        return logits
    


''' Syntax for model saving
Save:
torch.save(model.state_dict(), PATH)

Load:
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, weights_only=True)) #Model must be initialized BEFORE loading optimizer state. No weight_only argument for optimizers
model.eval()
'''

def test_similarity(data_file_name,doublenetwork, nbr_samples=2,device="cuda",nbr_iter=10, plot_sims=False):
    '''Picks random samples and gives the (average) similarity between the image emmbedding and the coordinate embedding'''
    _ , dataloader=dataloader_emb(data_file_name,batch_size=nbr_samples,shuffle=True) #test set
    doublenetwork.eval()

    results = []
    data_iter = iter(dataloader)  # manual iterator

    for _ in tqdm(range(nbr_iter), desc="Computing similarities"):
        try:
            batch = next(data_iter)
        except StopIteration:
            # reinitialize iterator if dataset is smaller than nbr_iter
            data_iter = iter(dataloader)
            batch = next(data_iter)
        emb_vectors, coord , _ = batch
        emb_vectors = emb_vectors.to(device)
        coord = coord.to(device)
        logits=doublenetwork(emb_vectors,coord)

        logits=logits/doublenetwork.logit_scale.exp()

        sim=logits.diagonal().mean()

        non_diag_mask = ~torch.eye(logits.size(0), dtype=torch.bool).to(device)
        assim = logits[non_diag_mask].mean()
        score=(sim,assim)
        if not torch.isnan(sim) and not torch.isnan(assim): #NaN if only one value in the batch for instance; in that case, drop that value
            results.append(score)
    # Compute mean and std for each element in the tuple
    mean = tuple(
        sum(x[i] for x in results) / len(results) for i in range(2)
    )
    std = tuple(
        torch.sqrt(sum((x[i] - mean[i])**2 for x in results) / len(results))
        for i in range(2)
    )
    if plot_sims:
        sim_list, asim_list = zip(*results) #transposes list of tuples into tuple of lists
        sim_list = [x.item() for x in sim_list]
        asim_list = [x.item() for x in asim_list]
        plot(sim_list,"Similarity for coresponding pairs")
        plot(asim_list,"Similarity for non-coresponding pairs")


    return mean[0], mean[1], std[0] , std[1] #respectively similarity and asimilarity
        
def plot(data, title,bins=50):
    #data=data.cpu().numpy()
    plt.hist(data, bins=bins, edgecolor='black',alpha=0.5)  # bins control the granularity
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.savefig(title)

def coord_trans(x, y, order="CH_to_normal"):

    if order == "CH_to_normal":
        transformer = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True)
        x_out, y_out = transformer.transform(x, y)  # X=Easting, Y=Northing
    elif order == "normal_to_CH":
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:21781", always_xy=True)
        x_out, y_out = transformer.transform(x, y)  # X=Longitude, Y=Latitude
    else:
        raise ValueError("order must be either 'CH_to_normal' or 'normal_to_CH'")
    return x_out, y_out