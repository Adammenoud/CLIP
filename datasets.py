import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset,random_split, Subset
import data_extraction
import h5py



class ImageFromDataframe(Dataset):
    def __init__(self, dataframe, image_folder_path ,label_names=('decimalLatitude','decimalLongitude'), transform=None):
        """
        dataframe must have column 'gbifID',
        and columns in label_names for the labels.

        It uses the same idx than the one to name the image files, so please do not re-index the dataframe

        image_folder_path must be the path to a folder containing the images, Not like ImageFolder!
        """
        self.dataframe = dataframe
        self.label_names = label_names
        self.transform = transform
        self.image_folder_path=image_folder_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f"{self.image_folder_path}/{idx}_{self.dataframe.loc[idx, 'gbifID']}.jpg"
        labels = self.dataframe.loc[idx, list(self.label_names)].to_numpy(dtype='float32')
        labels = torch.tensor(labels)       

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, labels
    
def dataloader_img(dataframe,image_folder_path,image_size,batch_size,shuffle=True):
        transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),  #as in HF
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #to [-1,1]
        ])
        dataset = ImageFromDataframe(dataframe, image_folder_path,label_names=('decimalLatitude','decimalLongitude'), transform=transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader


def dataloader_factory(file_path,config, dataframe=None):
    '''
    solves to the duplication problem, but needs to take a dataframe with gbifID column.
    Also possible to use naively without the dataframe by setting sort_duplicates=False.
    '''
    # Gets some parameters from the config
    vectors_name= config["vectors_name"]
    spe_data_path=config['paths'][config["dataset"]]["specie_data"]

    #Picks the right dataset class from the config
    if config["use_species"]:
        dataset=data_extraction.HDF5Dataset(file_path,data_name=vectors_name)
        dataset_type="HDF5Dataset"
    elif config["mixed_embeddings"]:
        if dataframe is None or spe_data_path is None:
            raise Exception(f" 'None' arguments passed to 'mixed_HDF5Dataset' ")
        dataset=data_extraction.mixed_HDF5Dataset(file_path_images=file_path,file_path_spe=spe_data_path, dataframe=dataframe, data_name_images=vectors_name,mixed_data_method=config["mixed_data_method"])
        dataset_type="mixed_HDF5Dataset"
    else:
        dataset=data_extraction.ordered_HDF5Dataset(file_path,dataframe,data_name=vectors_name)
        dataset_type="ordered_HDF5Dataset"
   
    #get the sizes of the train and test sets
    train_size = int(config["train_ratio"] * len(dataset))
    test_size = len(dataset) - train_size

    #ensure reproducibility
    generator = torch.Generator().manual_seed(48)

    #sorts the duplicates: several images corresponding to one occurence
    if config["sort_duplicates"]:
        if dataset_type=="HDF5Dataset": #The HDF5 class does not have a dataframe. The indices are stored in the HDF5 data (the hdf5 data is not necessarly ordered).
            with h5py.File(file_path, "r") as f:
                dict_indices = f["dict_idx"][:].flatten()
        else:
            dict_indices = dataset.dataframe.index
        train_dataset, test_dataset=group_split(dataset,dict_indices, dataframe,generator, config["train_ratio"])
    else:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)

    #creates the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=config["shuffle"],drop_last=config["drop_last"],num_workers=config["dataloader_workers"])
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=config["shuffle"],drop_last=config["drop_last"],num_workers=config["dataloader_workers"])
    return train_loader, test_loader, dataset_type

def group_split(dataset, dict_indices, dataframe,generator, train_ratio=0.8):
    """Split dataset into train/test SUCH THAT samples with the same ID stay together.
    Technically, there may be some variance to the training set proportion since not all Gbif indices have the same number of images."""
    n = len(dataset)
    # 1 — Extract IDs for each sample
    ids = dataframe.loc[dict_indices, "gbifID"].values  
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