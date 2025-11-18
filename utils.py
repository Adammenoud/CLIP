import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset,random_split
import data_extraction
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json
from pyproj import Transformer


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

def dataloader_emb(file_path,batch_size,shuffle=False,train_proportion=0.8): #from a h5 file
    dataset=data_extraction.HDF5Dataset(file_path)
    train_size = int(train_proportion * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(48)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader



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

    

def train(doublenetwork,epochs,dataloader,batch_size,lr=1e-4,device="cuda",save_name=None,saving_frequency=1):
    doublenetwork = doublenetwork.to(device)
    doublenetwork.train()
    writer =  SummaryWriter()
    optimizer=torch.optim.AdamW(doublenetwork.parameters(),lr=lr)
    l = len(dataloader)
    
    for ep in range(epochs):
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            logits=doublenetwork(images,labels) #logits of cosine similarities
            loss=CE_loss(logits,device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(CE=loss.item())
            writer.add_scalar("Cross-entropy", loss.item(), global_step=ep * l + i)
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
                    "current epoch (if stopped)" : ep}
                config_path = os.path.join(save_name, "hyperparameters.json")
                with open(config_path, "w") as f:
                    json.dump(hparams, f, indent=4)
    doublenetwork.eval()
    writer.close()
    return doublenetwork

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
        emb_vectors, coord = batch
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


def CH_to_coord(east, north): 
    '''Mind the order! It swaps to respect both order conventions
    ETH doc: https://ia.arch.ethz.ch/lat-lon-to-ch-coordinates/'''
    transformer_CH_to_WGS = Transformer.from_crs(
    "EPSG:21781",  # CH1903 / LV03
    "EPSG:4326",   # WGS84
    always_xy=True
    )
    lons, lats = transformer_CH_to_WGS.transform(east, north)
    return lats,lons