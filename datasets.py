import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset,random_split, Subset
from tqdm import tqdm
import pandas as pd
#local
import datasets
import h5py


#-------------------------------------------
#Dataset classes
#-------------------------------------------
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


class HDF5Dataset(Dataset):
    def __init__(self, file_path, data_name="vectors", label_name="coordinates"):
        self.file_path = file_path
        self.data_name = data_name
        self.label_name = label_name
        self.file = None  # file will be opened per worker

        # Open once just to get length
        with h5py.File(file_path, "r") as f:
            self.length = f[data_name].shape[0]
            self.has_idx = "dict_idx" in f

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # open file once per worker
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")

        data = self.file[self.data_name][idx]
        label = self.file[self.label_name][idx]
        dict_idx = self.file["dict_idx"][idx] if self.has_idx else -1

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(dict_idx, dtype=torch.int64)
        )

class ordered_HDF5Dataset(Dataset):
    '''
    This class assumes that the HDF5 indices correspond to indices of the dataframe.
    (This requirement is fullfilled using the most recent download_emb function.)

    It allows taking a subset by first filtering the dataframe.
    Note that some downloads may have failed, thus some rows of the dataframe will also not be used in the dataset.
    The rows that are filled with "None" should be identifiable from a mask hdf5 dataset, here "valid".

    The names of the hdf5 datasets downloaded with download_embeddings are: "vectors_bioclip", "vectors_dino", "coordinates"
    '''

    def __init__(self, file_path, dataframe, data_name="vectors", label_name="coordinates",valid_name="valid", do_valid=True):
        self.file_path = file_path
        self.data_name = data_name
        self.label_name = label_name
        self.valid_name = valid_name

        self.dataframe= dataframe
        self.file = None  # file will be opened per worker

        if do_valid:
            # Open once to read the valid mask
            with h5py.File(file_path, "r") as f:
                valid_mask = f[self.valid_name][:]
            # Filter the dataframe to only keep valid rows
            self.dataframe = self.dataframe.loc[valid_mask[self.dataframe.index]]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # open file once per worker
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        
        hdf5_idx = self.dataframe.index[idx] #get the right index: idx arguments is in 0,..., len-1, hdf5_idx is a dataframe/hdf5 index.


        data = self.file[self.data_name][hdf5_idx]
        label = self.file[self.label_name][hdf5_idx]

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.int64) #return the idx of dataframe, the one that has meaningful metadata associated to.
        )


class double_HDF5Dataset(Dataset):
    '''
    return both species embeddings and images embeddings in the __getitem__ (as well as ixd/coords).
    Specie embeddings have to be ordered as well (according to some common dataframe), otherwise there will be an error.
    '''

    def __init__(self, file_path_images,file_path_spe, dataframe, data_name_images="vectors_bioclip",data_name_spe="vectors", label_name="coordinates",valid_name="valid", do_valid=True):
        self.images_dataset=ordered_HDF5Dataset(file_path_images, dataframe, data_name=data_name_images, label_name=label_name,valid_name=valid_name, do_valid=do_valid)
        self.spe_dataset=HDF5Dataset(file_path_spe, data_name=data_name_spe, label_name=label_name)
        self.dataframe=dataframe

    def __len__(self):
        return len(self.images_dataset) #spe_data should have all entries
    
    def __getitem__(self, idx):
        #get 
        image_emb, image_coords ,image_idx = self.images_dataset[idx]
        spe_emb, _, spe_idx= self.spe_dataset[image_idx]
        if spe_idx != image_idx:
            raise ValueError("The indices returned by the image dataset do not correspond to those from the species dataset. (The species embedding dataset should be ordered, make sure this is the case).")

        return image_emb, spe_emb, image_coords, spe_idx
    

class mixed_HDF5Dataset(Dataset):
    '''
    combines species embeddings and images embeddings.
    Either simply concatenate, or concatenate species and the normalized image-specie vector, if difference=True
    Specie embeddings have to be ordered as well, otherwise there will be an error.

    Note that the vectors are normalized before taking any operation, since the scale should not matter, only the direction.
    When taking a difference, wwe normalizer with respect to the average difference norm, to avoid having very small values, but keep the magnitude information.
        (It might take some time to compute, but it is done only once at the initialisation.)
    '''

    def __init__(self, file_path_images,file_path_spe, dataframe, data_name_images="vectors_bioclip",data_name_spe="vectors", label_name="coordinates",valid_name="valid", do_valid=True,mixing_method="concatenate"):
        self.double_HDF5Dataset=double_HDF5Dataset(file_path_images,file_path_spe, dataframe, data_name_images=data_name_images,data_name_spe=data_name_spe, label_name=label_name,valid_name=valid_name, do_valid=do_valid)
        self.mixing_method=mixing_method
        self.dataframe=dataframe

        if mixing_method in ["difference", "concatenate_difference"]:
            self.avg_difference_norm=self.compute_avg_difference_norm()
            
    def __len__(self):
        return len(self.double_HDF5Dataset) #spe_data should have all entries
    
    def __getitem__(self, idx):

        #get 
        image_emb,spe_emb, coords, idx = self.double_HDF5Dataset[idx]

        #normalize
        image_emb= torch.nn.functional.normalize(image_emb,dim=0)
        spe_emb= torch.nn.functional.normalize(spe_emb,dim=0)

        #does the operation picked by 'mixing_method'
        if self.mixing_method=="concatenate":
            final_emb=torch.cat((spe_emb,image_emb),dim=0) # the size is (768,) for now (__getitem__ method gets single item, not a batch)
        elif self.mixing_method=="sum":
            final_emb=spe_emb+image_emb # the size is (768,) for now (__getitem__ method gets single item, not a batch)
        elif self.mixing_method in ["difference", "concatenate_difference"]:
            difference_emb=image_emb-spe_emb
            difference_emb= difference_emb/self.avg_difference_norm
            if self.mixing_method=="concatenate_difference":
                final_emb=torch.cat((spe_emb,difference_emb),dim=0)
            else:
                final_emb=difference_emb
        else:
            raise ValueError(f"Invalid mixing method: {self.mixing_method}. Supported methods are 'concatenate', 'difference', and 'concatenate_difference'.")

        return final_emb, coords, idx
    
    def compute_avg_difference_norm(self):
        dataloader= DataLoader(self.double_HDF5Dataset, batch_size=1024, shuffle=False, num_workers=16)
        total_norm=0
        pbar = tqdm(dataloader, desc="Computing average difference norm")
        for image_emb, spe_emb, _, _ in pbar:
            image_emb= torch.nn.functional.normalize(image_emb,dim=1)
            spe_emb= torch.nn.functional.normalize(spe_emb,dim=1)
            difference_emb=image_emb-spe_emb
            batch_norms=torch.norm(difference_emb, dim=1)
            total_norm+=batch_norms.sum().item()
        avg_norm=total_norm/len(self.double_HDF5Dataset)
        return avg_norm








class dataframe_data(Dataset):
    def __init__(self, dataframe, idx_mapping=None, data_name=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing your data.
            idx_mapping (dict, optional): Mapping from dataset idx to row idx in dataframe.
            data_name (str, optional): Name of column to use as 'data'. If None, data will be None.
        """
        self.df = dataframe
        self.data_name = data_name
        self.idx_mapping = idx_mapping
        self.has_idx = idx_mapping is not None
        self.length = len(idx_mapping) if self.has_idx else len(dataframe)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Map idx through dataframe if provided
        actual_idx = self.idx_mapping[idx] if self.has_idx else idx
        row = self.df.iloc[actual_idx]

        # Prepare data
        data = row[self.data_name] if self.data_name is not None else None
        if data is not None:
            data = torch.tensor(data, dtype=torch.float32)

        # Prepare label tensor
        label = torch.tensor([row['decimalLatitude'], row['decimalLongitude']], dtype=torch.float32)

        # Correct dict_idx to mimic HDF5Dataset
        dict_idx_value = idx if not self.has_idx else actual_idx
        dict_idx = torch.tensor(dict_idx_value, dtype=torch.int64)

        return data, label, dict_idx
    


#-------------------------------------------
#Dataloader classes
#-------------------------------------------
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
    if dataframe is None:
        dict_path = config['paths'][config["dataset"]]['dict']
        dataframe=pd.read_csv(dict_path)

    #Picks the right dataset class from the config
    if config["use_species"] and config["mixed_embeddings"]:
        raise ValueError("Cannot use both 'use_species' and 'mixed_embeddings' options at the same time, please choose one of the two.")
    if config["use_species"]:
        dataset=datasets.HDF5Dataset(file_path,data_name=vectors_name)
        dataset_type="HDF5Dataset"
    elif config["mixed_embeddings"]:
        dataset=datasets.mixed_HDF5Dataset(file_path_images=file_path,file_path_spe=spe_data_path, dataframe=dataframe, data_name_images=vectors_name,mixing_method=config["mixed_data_method"])
        dataset_type="mixed_HDF5Dataset"
    else:
        dataset=datasets.ordered_HDF5Dataset(file_path,dataframe,data_name=vectors_name)
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