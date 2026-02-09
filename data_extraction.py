import pandas as pd
import numpy as np
import os
import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import hdf5
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
import utils
import time
from requests.exceptions import RequestException
import threading
import matplotlib.pyplot as plt


#data available at https://api.gbif.org/v1/occurrence/download/request/0015029-251025141854904.zip

def get_dictionary(nbr_images, path_occurences, path_multimedia, extra_occ_columns=None, sep="\t"):
    # Load (takes about 3 min)
    occurrences = pd.read_csv(path_occurences, sep=sep, low_memory=False)
    multimedia = pd.read_csv(path_multimedia, sep=sep, low_memory=False)

    # Keep:
    #first entries
    if nbr_images is not None:
        multimedia = multimedia.head(nbr_images)
    #specific columns (here identifier, to be able to download images)
    multimedia = multimedia[['gbifID', 'identifier']]

    # Merges with occurrences to get coordinates (can also add more covariates)
    occ_cols = ['gbifID', 'decimalLatitude', 'decimalLongitude']
    if extra_occ_columns:
        occ_cols += extra_occ_columns
    merged = pd.merge(
        multimedia,
        occurrences[occ_cols],
        on='gbifID',
        how='left'
    )
    return merged
#"individualCount", "coordinateUncertaintyInMeters", "coordinatePrecision", "taxonID", "scientificNameID" , "acceptedNameUsageID" , "parentNameUsageID", "originalNameUsageID", "scientificName"
#class	order	superfamily	family	subfamily	tribe	subtribe	genus	genericName	subgenus

def download_imgs(dictionary, output_dir="downloaded_images/all_images"):
    """
    Mind the name under which images are registered:
    {idx}_{row['gbifID']}.jpg
    """
    os.makedirs(output_dir, exist_ok=True)

    # Wrap iterrows with tqdm
    for idx, row in tqdm(dictionary.iterrows(), total=len(dictionary), desc="Downloading images"):
        url = row['identifier']
        if pd.notna(url):
            response = requests.get(url)
            if response.status_code == 200:
                file_path = f"{output_dir}/{idx}_{row['gbifID']}.jpg"
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {url}")

'''
def download_emb(dictionary, dim_emb,output_dir="downloaded_embeddings",device="cuda"):
    #Mind the name under which images are registered:  {idx}_{row['gbifID']}.jpg

    # Create folder for embeddings
    hdf5.create_HDF5_file(vector_length=dim_emb,name=output_dir)
    file = hdf5.open_HDF5(output_dir + ".h5")

    #get DINOv2
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    # Loop over rows
    for idx, row in tqdm(dictionary.iterrows(), total=dictionary.shape[0], desc="Downloading embeddings"):
        url = row['identifier']
        if pd.notna(url):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                # get image
                image = Image.open(response.raw).convert("RGB")
                #encode image
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                cls_embedding = last_hidden_states[:, 0, :].cpu().detach().numpy()#get only the cls
                # get label (coordinates)
                coordinates = np.array([[row['decimalLatitude'], row['decimalLongitude']]])  # shape (1,2)
                #appends
                hdf5.append_HDF5(vectors_to_add=cls_embedding, label_to_add=coordinates,file=file, data_name="vectors",label_name="coordinates")
            else:
                print(f"Failed to download {url}")
    file.close()
'''
def get(url):
    url = url.replace("original", "medium")
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()  # raises HTTPError for 4xx / 5xx
    return response
def log_error(row, error_msg, ERROR_FILE,error_lock):
    with error_lock:
        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(f"{row.to_dict()}\t{error_msg}\n")

def process_row(row, processor_dino, processor_bioclip, model_dino,model_bioclip, device, dictionary, tokenizer_bioclip,ERROR_FILE,error_lock):
    """Download image, compute embedding, return embedding and coordinates.
        Each coordinate is a 2 elements array with order lat, lon
    """
    url = row['identifier']
    if pd.isna(url):
        return None

    try:
        try:
            response=get(url)
        except RequestException as e:
            print(f"Request error for {url}: {e}. Waiting 1s...")
            time.sleep(1)
            try:
                response=get(url) #write log error
            except Exception as e2:
                log_error(row, f"Second attempt failed: {e2}", ERROR_FILE,error_lock)
                return None

        image = Image.open(response.raw).convert("RGB")
        #DINO:
        inputs = processor_dino(images=image, return_tensors="pt") #for DINOv2
        inputs = {k: v.to(device) for k, v in inputs.items()} #for DINOv2
        #bioclip_
        image_tensor = processor_bioclip(image).unsqueeze(0).to(device)
        with torch.no_grad(): #encode
            outputs_dino = model_dino(**inputs)  #for DINO-V2 (normal forward method)
            outputs_dino=outputs_dino.last_hidden_state[:, 0, :]
            outputs_bioclip = model_bioclip.encode_image(image_tensor) #for bioCLIP (image encoder)

        cls_embedding_dino = outputs_dino.cpu().numpy()
        cls_embedding_bioclip = outputs_bioclip.cpu().numpy()
        coordinates = np.array([[row['decimalLatitude'], row['decimalLongitude']]])
        #if difference:
            #specie_emb= utils.get_species_emb(np.array([row.name]), dictionary, model,tokenizer, column_name="taxa_bioclip", device="cuda")
            #cls_embedding=cls_embedding - specie_emb.cpu().numpy()
        return cls_embedding_bioclip,cls_embedding_dino , coordinates
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def download_emb(dictionary, dim_emb, output_dir="downloaded_embeddings", device="cuda", max_workers=8):
    """Download images and compute embeddings in parallel.
        Each coordinate is a 2 elements array with order lat, lon
        File registered as output_dir + ".h5"
        assumes the dictionary has every index
    """
    #error management
    error_lock = threading.Lock()
    ERROR_FILE = output_dir+ "_error_rows.txt"
    #Create HDF5
    file = h5py.File(output_dir + ".h5", "w")
    N = len(dictionary)
    vectors_bioclip = file.create_dataset(
        "vectors_bioclip", shape=(N, dim_emb), dtype="float32")
    vectors_dino = file.create_dataset(
        "vectors_dino", shape=(N, dim_emb), dtype="float32")
    coords = file.create_dataset(
        "coordinates", shape=(N, 2), dtype="float32")
    valid = file.create_dataset("valid", shape=(N,), dtype="bool")

    #load the models
    processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    model_dino.eval()

    model_bioclip, preprocess_train, processor_bioclip = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    model_bioclip = model_bioclip.to(device)
    model_bioclip.eval()
    tokenizer_bioclip = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')

    # Parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_row, row, processor_dino, processor_bioclip, model_dino,model_bioclip, device, dictionary, tokenizer_bioclip,ERROR_FILE,error_lock): idx  #syntax: .submit(function, arg1, arg2, ...). Basically, an ashychronous "for" loop on idx, row that tracks the output of process_row, and index it on idx
                         for idx, row in dictionary.iterrows()}

        for future in tqdm(as_completed(future_to_idx), total=dictionary.shape[0], desc="Downloading embeddings"):
            res = future.result()
            idx = future_to_idx[future]  # original DataFrame index
            if res is not None:
                cls_embedding_bioclip,cls_embedding_dino , coordinates = res
                vectors_bioclip[idx] = cls_embedding_bioclip
                vectors_dino[idx] = cls_embedding_dino
                coords[idx] = coordinates
                valid[idx] = True
            else:
                valid[idx] = False
                print(f"Warning: failed to process row {idx}, URL: {dictionary.loc[idx, 'identifier']}")

    file.close()

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

class mixed_HDF5Dataset(Dataset):
    '''
    combines species embeddings and images embeddings.
    Either simply concatenate, or concatenate species and the normalized image-specie vector, if difference=True
    Specie embeddings have to be ordered as well, otherwise there will be an error.
    '''

    def __init__(self, file_path_images,file_path_spe, dictionary, data_name_images="vectors_bioclip",data_name_spe="vectors", label_name="coordinates",valid_name="valid", do_valid=True,difference=False):
        self.images_dataset=ordered_HDF5Dataset(file_path_images, dictionary, data_name=data_name_images, label_name=label_name,valid_name=valid_name, do_valid=do_valid)
        self.spe_dataset=HDF5Dataset(file_path_spe, data_name=data_name_spe, label_name=label_name)
        self.difference=difference
        self.dictionary=dictionary

    def __len__(self):
        return len(self.images_dataset) #spe_data should have all entries
    
    def __getitem__(self, idx):
        image_emb, image_coords ,image_idx = self.images_dataset[idx]
        spe_emb, _, spe_idx= self.spe_dataset[image_idx]
        if spe_idx != image_idx:
            raise ValueError("The indices returned by the image dataset do not correspond to those from the species dataset.")
        
        if self.difference:
            image_emb=image_emb-spe_emb
            image_emb= torch.nn.functional.normalize(image_emb,dim=0) # the size is (768,) for now (gets single item)

        return torch.cat((spe_emb,image_emb),dim=0), image_coords, spe_idx



class ordered_HDF5Dataset(Dataset):
    '''
    This class assumes that the HDF5 indices correspond to indices of the dictionary.
    (This requirement is fullfilled using the most recent download_emb function.)

    It allows taking a subset by first filtering the dictionary.
    Note that some downloads may have failed, thus some rows of the dictionary will also not be used in the dataset.
    The rows that are filled with "None" should be identifiable from a mask hdf5 dataset, here "valid".

    The names of the hdf5 datasets downloaded with download_embeddings are: "vectors_bioclip", "vectors_dino", "coordinates"
    '''

    def __init__(self, file_path, dictionary, data_name="vectors", label_name="coordinates",valid_name="valid", do_valid=True):
        self.file_path = file_path
        self.data_name = data_name
        self.label_name = label_name
        self.valid_name = valid_name

        self.dictionary= dictionary
        self.file = None  # file will be opened per worker

        if do_valid:
            # Open once to read the valid mask
            with h5py.File(file_path, "r") as f:
                valid_mask = f[self.valid_name][:]
            # Filter the dictionary to only keep valid rows
            self.dictionary = self.dictionary.loc[valid_mask[self.dictionary.index]]

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # open file once per worker
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        
        hdf5_idx = self.dictionary.index[idx] #get the right index: idx arguments is in 0,..., len-1, hdf5_idx is a dictionary/hdf5 index.


        data = self.file[self.data_name][hdf5_idx]
        label = self.file[self.label_name][hdf5_idx]

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.int64) #return the idx of dictionary, the one that has meaningful metadata associated to.
        )




class dictionary_data(Dataset):
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
        # Map idx through dictionary if provided
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

def get_species_emb(dictionary, output_dir, batch_size=4096):
    #Load bioclip model
    bioclip, preprocess_train, processor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    bioclip = bioclip.to("cuda")
    bioclip.eval()

    hdf5.create_HDF5_file(vector_length=768, name=output_dir)
    file = hdf5.open_HDF5(output_dir + ".h5")

    indices = dictionary.index.tolist()

    for start in tqdm(range(0, len(indices), batch_size), desc="Processing batches"):
        batch_indices = indices[start:start + batch_size]
        batch_df = dictionary.loc[batch_indices]

        coordinates = batch_df[['decimalLatitude', 'decimalLongitude']].to_numpy()
        vectors_to_add = utils.get_species_emb(np.array(batch_indices), dictionary, bioclip, tokenizer)

        # If torch tensors, convert to numpy
        if hasattr(vectors_to_add, "cpu"):
            vectors_to_add = vectors_to_add.cpu().numpy()

        hdf5.append_HDF5(
            vectors_to_add=vectors_to_add,
            label_to_add=coordinates,
            file=file,
            data_name="vectors",
            label_name="coordinates",
            dict_idx=np.array(batch_indices).reshape(-1, 1)
        )

    file.close()

def one_hot(indices, spe_classIdx_dict):
    '''
    takes the indices from the original dictionary/hp5 file, returns one hot vectors using the sepcies/class index dictionary
    '''
    n_species = len(spe_classIdx_dict)
    indices=utils.to_numpy(indices).flatten()

    class_indices = spe_classIdx_dict.loc[indices, 'class_idx'].values
    one_hot_vectors = np.zeros((len(indices), n_species), dtype=np.float32)
    one_hot_vectors[np.arange(len(indices)), class_indices] = 1.0
    return one_hot_vectors

def species_classIdx_dict(dictionary, class_name="scientificName"):
    class_codes = pd.Categorical(dictionary[class_name]).codes
    df_codes = pd.DataFrame({'class_idx': class_codes}, index=dictionary.index)
    return df_codes



def plot_image_from_index(data_dict, index, return_only=False):
    url = data_dict[index]["identifier"]
    response = get(url, stream=True)
    response.raise_for_status()

    image = Image.open(response.raw).convert("RGB")

    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__== "__main__":
    
    #INaturalist data (from the filtered csv's):
    # print("making dictionary")
    # path_occurences="Data/filtered_inaturalist/occurrence_plants.txt"
    # path_multimedia="Data/filtered_inaturalist/multimedia_plants.txt"
    # taxa_cols= ['kingdom','phylum','class', 'order','family','genus','species']
    # extra_occ_columns=['scientificName','countryCode']+ taxa_cols
    # dictionary=get_dictionary(None, path_occurences, path_multimedia,extra_occ_columns=extra_occ_columns,sep=",")#3167066 rows
    # #create a new column with the full bioclip name (not with the vernacular name)
    # taxa_cols.remove("genus") #For some reason, species contains genus+species ... removing genus gives the correct full taxon
    # dictionary['taxa_bioclip'] = (
    #     dictionary[taxa_cols]
    #     .astype(str)
    #     .replace('nan', '')
    #     .apply(lambda row: ' '.join([v for v in row if v]), axis=1) #separated by only a space, like in bioclip
    # )
    # #save
    # dictionary.to_csv("Embeddings_and_dictionaries/plants/dictionary_inaturalist_FR_plants")


    # print("downloading arthropods embeddings")
    # #make embeddings
    # dictionary= pd.read_csv("Embeddings_and_dictionaries/dictionary_inaturalist_FR_arthropods")
    # #dictionary= pd.read_csv("embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioclip_data_dictionary_all_taxons")
    # dictionary=dictionary
    # download_emb(dictionary, dim_emb=768, output_dir="embeddings_inaturalist_FR_arthropods",max_workers=64)
    # print("download arthropods finished")

    print("downloading plants embeddings")
    #make embeddings
    dictionary= pd.read_csv("Embeddings_and_dictionaries/plants/dictionary_inaturalist_FR_plants")
    download_emb(dictionary, dim_emb=768, output_dir="embeddings_inaturalist_FR_plants",max_workers=16)
    print("download plants finished")

    # print("downloading plants embeddings")
    # #make embeddings
    # dictionary= pd.read_csv("Embeddings_and_dictionaries/dictionary_inaturalist_FR_mushrooms")
    # dictionary=dictionary
    # download_emb(dictionary, dim_emb=768, output_dir="embeddings_inaturalist_FR_mushrooms",max_workers=64)
    # print("download mushrooms finished")

    # #Arthropods
    # dictionary= pd.read_csv("Embeddings_and_dictionaries/arthropods/dictionary_inaturalist_FR_arthropods")
    # get_species_emb(dictionary, "Embeddings_and_dictionaries/arthropods/species_embeddings_inaturalist_FR_arthropods", batch_size=4096)

    # #Mushrooms
    # dictionary= pd.read_csv("Embeddings_and_dictionaries/mushrooms/dictionary_inaturalist_FR_mushrooms")
    # get_species_emb(dictionary, "Embeddings_and_dictionaries/mushrooms/species_embeddings_inaturalist_FR_mushrooms", batch_size=4096)

    # #Plants
    # dictionary= pd.read_csv("Embeddings_and_dictionaries/plants/dictionary_inaturalist_FR_plants")
    # get_species_emb(dictionary, "Embeddings_and_dictionaries/plants/species_embeddings_inaturalist_FR_plants", batch_size=4096)
