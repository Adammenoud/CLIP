import pandas as pd
import numpy as np
import os
import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import hdf5
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.utils.data import Dataset, DataLoader



#data available at https://api.gbif.org/v1/occurrence/download/request/0015029-251025141854904.zip

def get_dictionary(nbr_images, path_occurences, path_multimedia, extra_occ_columns=None):
    # Load (takes about 3 min)
    occurrences = pd.read_csv(path_occurences, sep="\t", low_memory=False)
    multimedia = pd.read_csv(path_multimedia, sep="\t", low_memory=False)

    # Keep:
    #first entries
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
    '''Mid the name under which images are registered: 
            {idx}_{row['gbifID']}.jpg
            '''
    # Create folder for images
    os.makedirs(output_dir,exist_ok=True)

    # Loop over rows
    for idx, row in dictionary.iterrows():
        url = row['identifier']
        if pd.notna(url):
            response = requests.get(url)
            if response.status_code == 200:
                # Save as idx_gbifID.jpg
                file_path = f"{output_dir}/{idx}_{row['gbifID']}.jpg" #careful: there will be gaps in idx if the data is filtered (indices will not be from 0 to nbr_images-1)
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


def process_row(row, processor, model, device):
    """Download image, compute embedding, return embedding and coordinates."""
    url = row['identifier']
    if pd.isna(url):
        return None

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"Failed to download {url}")
            return None

        image = Image.open(response.raw).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs) #
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        coordinates = np.array([[row['decimalLatitude'], row['decimalLongitude']]])
        return cls_embedding, coordinates
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def download_emb(dictionary, dim_emb, output_dir="downloaded_embeddings", device="cuda", max_workers=8):
    """Download images and compute embeddings in parallel."""
    # Create HDF5 file
    hdf5.create_HDF5_file(vector_length=dim_emb, name=output_dir)
    file = hdf5.open_HDF5(output_dir + ".h5")

    # Load DINOv2 model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    model.eval()

    #model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    #tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    #to encode:
    #image_features = model.encode_image(image_tensor)
    #text_features  = model.encode_text(text_tokens)



    # Parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_row, row, processor, model, device): idx  #syntax: .submit(function, arg1, arg2, ...). Basically, an ashychronous "for" loop on idx, row that tracks the output of process_row, and index it on idx
                         for idx, row in dictionary.iterrows()}

        for future in tqdm(as_completed(future_to_idx), total=dictionary.shape[0], desc="Downloading embeddings"):
            res = future.result()
            if res is not None:
                cls_embedding, coordinates = res
                hdf5.append_HDF5(vectors_to_add=cls_embedding, label_to_add=coordinates, file=file,
                                 data_name="vectors", label_name="coordinates")

    file.close()


class HDF5Dataset(Dataset):
    def __init__(self, file_path, data_name="vectors",label_name="coordinates"):
        self.file_path = file_path

        self.dataset_name = data_name
        self.label_name=label_name
        # Open file once to read shape
        file=hdf5.open_HDF5(file_path)
        self.length = file[data_name].shape[0]
        file.close()


    def __len__(self):
        return self.length  #technically, only shows length of "vectors"

    def __getitem__(self, idx):
        # Open file *every time* to avoid multi-threading issues (only noticed a small difference in performance)
        file=hdf5.open_HDF5(self.file_path)
        data = file[self.dataset_name][idx]  # Load only one sample
        label=file[self.label_name][idx]

        data_return= torch.tensor(data, dtype=torch.float32)
        label_return=torch.tensor(label, dtype=torch.float32)
        file.close()
        return (data_return, label_return)



