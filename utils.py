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
from pyproj import Transformer
import h5py
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely import geometry as geom
import regionmask
import rasterio
import open_clip
import pandas as pd


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

def dataloader_emb(file_path,batch_size,shuffle=False,train_ratio=0.8, sort_duplicates=False, dictionary=None,drop_last=True): #from a h5 file
    '''solves to the duplication problem, but need to take a dictionary with gbifID column.'''

    dataset=data_extraction.HDF5Dataset(file_path)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(48)
    if sort_duplicates:
        train_dataset, test_dataset=group_split(dataset,file_path, dictionary,generator, train_ratio)
    else:
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
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










def test_similarity(data_file_name, doublenetwork, nbr_iter=1000, nbr_samples=2,device="cuda", plot_sims=True,sort_duplicates=True, dictionary_path="embeddings_data_and_dictionaries/data_dictionary_sciName"):
    '''Picks random samples and gives the (average) similarity between the image emmbedding and the coordinate embedding
    dictionary is used to get the dataloader. It can be set to None, together with
    '''
    if dictionary_path is not None:
        dictionary=pd.read_csv(dictionary_path)
    else:
        dictionary=None
    _ , dataloader=dataloader_emb(data_file_name,batch_size=nbr_samples,shuffle=True,sort_duplicates=sort_duplicates, dictionary=dictionary) #test set
    doublenetwork.eval()

    data_iter = iter(dataloader)  # manual iterator
    all_sim_values = []
    all_asim_values = []


    for _ in tqdm(range(nbr_iter), desc="Computing similarities"):
        try:
            batch = next(data_iter)
        except StopIteration:
            # reinitialize iterator if dataset is smaller than nbr_iter
            data_iter = iter(dataloader)
            batch = next(data_iter)
        emb_vectors, coord , idx = batch
        emb_vectors = emb_vectors.to(device)
        coord = coord.to(device)
        idx=idx = idx.numpy().reshape(-1).tolist()
        logits=doublenetwork(emb_vectors,coord, idx)

        logits=logits/doublenetwork.logit_scale.exp()
        #get values
        sim=logits.diagonal()
        non_diag_mask = ~torch.eye(logits.size(0), dtype=torch.bool).to(device)
        asim = logits[non_diag_mask]
        #Nan
        sim = sim[~torch.isnan(sim)]
        asim = asim[~torch.isnan(asim)]
        #log
        all_sim_values.append(sim)
        all_asim_values.append(asim)
    # concatenate everything
    all_sim_values = torch.cat(all_sim_values)         # shape (N_matches,)
    all_asim_values = torch.cat(all_asim_values) 
    # Compute mean and std
    mean_sim  = all_sim_values.mean().item()
    mean_asim = all_asim_values.mean().item()
    std_sim   = all_sim_values.std().item()
    std_asim  = all_asim_values.std().item()
    if plot_sims:
        plot_both(all_sim_values.cpu().tolist(),
                all_asim_values.cpu().tolist(),
                "Similarity vs Asimilarity Distribution")
    return mean_sim, mean_asim, std_sim, std_asim
        
def plot_both(sim_data, asim_data, title, bins=50):
    plt.figure(figsize=(8, 5))

    plt.hist(sim_data, bins=bins, density=True, alpha=0.5, label="Corresponding pairs")
    plt.hist(asim_data, bins=bins, density=True, alpha=0.5, label="Random pairs")

    plt.xlabel("Similarity score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(title + ".png")
    plt.close()


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

def coord_trans_shift(x,y, order="CH_to_normal"):
    "converts from the NCEAS dataset coordinates (they have a shift) to regular lat, lon (and vice versa)"
    shift_x, shift_y= (1011627.4909483634, -100326.1477937577) #See "coordinates.ipynb"
    if order=="CH_to_normal":
        lons, lats = coord_trans(x-shift_x, y-shift_y,order="CH_to_normal")
        return lons, lats
    elif order =="normal_to_CH":
        x_trans, y_trans=coord_trans(x, y, order= "normal_to_CH")
        return x_trans+shift_x, y_trans+shift_y
        
    

def perform_PCA(dataloader, img_enc, device="cuda", file_name="pca_model", n_components=None, n_sample=None):
    """load with pca_model = joblib.load('pca_model.pkl')"""
    all_embeddings = []
    
    counter=0
    for embeddings, labels, _ in tqdm(dataloader, desc="Computing embeddings for PCA"):
        embeddings = embeddings.to(device)
        embeddings = img_enc(embeddings)  # get the actual encoded vectors
        all_embeddings.append(embeddings.detach().cpu().numpy())
        counter+= embeddings.shape[0]
        if n_sample is not None and counter>=n_sample:
            break

    embeddings = np.vstack(all_embeddings)
    scaler = StandardScaler()
    scaler.fit(embeddings)
    embeddings = scaler.transform(embeddings)
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    joblib.dump(pca, f"PCA_models/{file_name}.pkl")



def coord_to_PCA(coords, pos_enc,pca_model_path, comp_idx=1): 
    '''takes coordinates and return the principal value associated to it. Considers the comp_idx-th component'''
    pca_model=joblib.load(pca_model_path)
    embeddings=pos_enc(coords).cpu()
    vect_to_plot = pca_model.transform(embeddings.detach().numpy())
    return vect_to_plot[:,comp_idx]

def plot_country_values(country_name, fct_to_plot,pos_encoder,pca_model_path="PCA_All_comp_full_dataset_Normalized.pkl", grid_resolution=0.1, cmap='viridis',device="cuda",comp_idx=0,save_path=None):
    coords = create_country_grid(country_name, grid_resolution)
    values = fct_to_plot(coords.to(device), pos_encoder,pca_model_path,comp_idx=comp_idx)#np.array([coord_trans(coord) for coord in coords])
    # Get the country polygon
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    idx = list(countries.names).index(country_name)
    polygon = countries.polygons[idx]
    if country_name=="France":
        polygon=filter_France(polygon)
    boundary = gpd.GeoSeries([polygon])
    plot_values(coords,values,boundary,save_path,country_name)

def create_country_grid(country_name, grid_resolution=0.1):
    '''Returns the coordinates of points inside the country. 
        return size: (nbr_points,2) with order lat, lon for the 2nd dimension
    '''
    # Load 110m Natural Earth countries
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    # Find the index of the country
    try:
        idx = list(countries.names).index(country_name)
    except ValueError:
        raise ValueError(f"Country '{country_name}' not found in regionmask.")

    polygon = countries.polygons[idx]
    if country_name=="France":
        polygon=filter_France(polygon)
    # Create a lon/lat grid covering the bounding box
    minx, miny, maxx, maxy = polygon.bounds
    lon_grid = np.arange(minx, maxx, grid_resolution)
    lat_grid = np.arange(miny, maxy, grid_resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    coords = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])

    # Keep only points inside the polygon
    coords_inside = np.array([coord for coord in coords if polygon.contains(Point(coord))])
    print (coords_inside.shape)
    coords_inside= torch.tensor(coords_inside, dtype=torch.float32)
    return coords_inside #returns a torch tensor


def filter_France(polygon):
    parts = list(polygon.geoms)
    bbox_metro = geom.box(minx=-10, miny=40, maxx=15, maxy=55)

    metro_parts = [p for p in parts if p.intersects(bbox_metro)]
    if not metro_parts:
        raise RuntimeError("Could not isolate metropolitan France polygon.")

    return MultiPolygon(metro_parts)


def get_encoders(model): #returns the position encoder and image encoder from the model, depending on the name
    if hasattr(model, "pos_encoder"):
        pos_encoder=model.pos_encoder
    elif hasattr(model, "location_encoder"):
        pos_encoder=model.location_encoder
    else:
        raise ValueError("do_and_plot_PCA: model does not have a recognized position encoder attribute")
    if hasattr(model, "lin1") and hasattr(model, "relu") and hasattr(model, "lin2"):
        img_encoder = nn.Sequential(model.lin1,model.relu,model.lin2)
    elif hasattr(model, "img_encoder"):
        img_encoder=model.img_encoder
    elif hasattr(model, "image_encoder"):
        img_encoder=model.image_encoder
    else:
        raise ValueError("do_and_plot_PCA: model does not have a recognized image encoder attribute")
    return pos_encoder, img_encoder

def do_and_plot_PCA(model, data_path,pca_file_name,nbr_components=None, nbr_plots=3, batch_size=4064, sort_duplicates=False,dictionary=None,country_name="Switzerland",save_path_pic=None):

    pos_encoder, img_encoder=get_encoders(model)

    dataloader, _ =dataloader_emb(data_path,batch_size=batch_size, shuffle=True,train_ratio=0.8, sort_duplicates=sort_duplicates, dictionary=dictionary)
    perform_PCA(dataloader, img_enc=img_encoder, device="cuda", file_name=pca_file_name, n_components=nbr_components, n_sample=None)

    pca_model_path=f"PCA_models/{pca_file_name}.pkl"
    
    for i in range(nbr_plots):
        plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.01, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path_pic)

def plot_PCA(pca_model_path,nbr_plots,pos_encoder,country_name="Switzerland",save_path=None):
    for i in range(nbr_plots):
        plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.01, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path)


def NCEAS_covariates(lon, lat, directory="embeddings_data_and_dictionaries/data_SDM_NCEAS/Environnement", return_dict=True):
    """
    Fetch raster values for given lon/lat coordinates from all .tif files in a directory.
    
    Parameters:
        lon (array-like): Longitudes
        lat (array-like): Latitudes
        directory (str or Path): Path to folder containing .tif raster files
    
    Returns: either a dictionary or an np array, depending on the argumennt "return_dict"
        dict: Keys are raster filenames (without extension), values are arrays of sampled values
        
    """
    directory = Path(directory)
    
    # Transform coordinates
    x, y = coord_trans_shift(lon, lat, order="normal_to_CH") #error?
    points = np.column_stack([x, y])  # shape (n_points, 2)
    
    # Get all .tif files
    tif_files = sorted(directory.glob("*.tif"))
    
    if len(tif_files) == 0:
        print("No TIFF files found in directory:", directory)
        return {}
    
    # Dictionary to hold values from all rasters
    all_values = {}
    
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            # Sample raster at given points
            values = np.array([v[0] for v in src.sample(points)])
            all_values[tif_path.stem] = values  # use filename without extension as key
    
    if return_dict:
        return all_values
    else:
        # Stack values into a 2D array: shape (num_points, num_rasters)
        # Order of columns matches tif_files
        array_values = np.column_stack([all_values[tif_path.stem] for tif_path in tif_files])
        return array_values

def print_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt  # assume raw state_dict
    print("\n=== Keys in state_dict ===")
    for k in state_dict.keys():
        print(k)
    print("\n=== Shapes of tensors ===")
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            print(f"{k:40s}: {tuple(v.shape)}")
        else:
            print(f"{k:40s}: NON-TENSOR ({type(v)})")


def get_gbif_covariates(dictionary, idx_list, covariate_list=['scientificName']):
    '''looks into the dictionary to get the data corespondint to the indices.
        the covariates must be columns in the dictionary.
        retrun: (len(idx_list), len(covariate_list)) array
    '''
    # Ensure all requested covariates exist in the DataFrame
    missing_cols = [col for col in covariate_list if col not in dictionary.columns]
    if missing_cols:
        raise ValueError(f"The following covariates are missing in the DataFrame: {missing_cols}")
    # Select rows and columns, convert to numpy array
    selected_data = dictionary.loc[idx_list, covariate_list].to_numpy()
    return selected_data


def apply_pos_enc(coords, pos_encoder,pca_model_path=None,comp_idx=None):
    return pos_encoder(coords).cpu()


def map_image(doublenetwork, path_to_image, country="Switzerland", device="cuda", grid_resolution=0.1, save_path=None):
    '''
    Plots the similarity score of each coordinate with the image.
    If no save_path is given, uses plt.show()
    '''
    image=Image.open(path_to_image).convert("RGB")
    emb_img = embedds_image(image)
    map_embedding(doublenetwork, emb_img, country=country, device=device, grid_resolution=grid_resolution, save_path=save_path)
    

def embedds_image(image, device="cuda"):
    '''Fetches the model every time: only for plots or exeptinal use, do not call repeatedly!'''
    bioclip, preprocess_train, processor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    image= processor(image).unsqueeze(0).to(device)
    bioclip = bioclip.to(device)
    bioclip.eval()
    emb_img = bioclip.encode_image(image)
    return emb_img

def map_embedding(doublenetwork, embedding, country="Switzerland", device="cuda", grid_resolution=0.1, save_path=None):
    '''
    Plots the similarity score of each coordinate with the image embedding.
    If no save_path is given, uses plt.show()
    '''
    coords= create_country_grid(country, grid_resolution=grid_resolution).to(device)
    coords = torch.flip(coords, dims=[1])
    values=doublenetwork(embedding,coords)
    coords=torch.flip(coords, dims=[1])
    print("sims shape:", values.shape)
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    idx = list(countries.names).index(country)
    polygon = countries.polygons[idx]
    print(country)
    if country=="France":
        polygon=filter_France(polygon)
    boundary = gpd.GeoSeries([polygon])
    plot_values(coords,values,boundary,save_path,country)

def plot_values(coords,values,boundary,save_path,country):
    coords=to_numpy(coords)
    values=to_numpy(values)
    fig, ax = plt.subplots(figsize=(8, 10))
    boundary.plot(ax=ax, color="none", edgecolor="black")
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Value")
    ax.set_title(f"Image similarity over {country}")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shapely.geometry import Point, Polygon
import regionmask
import torch

def plot_species_density(df, species_name, country_name, grid_resolution=0.1, bandwidth=0.05,plot_points=True):
    """
    Plots the spatial density of a species over a specific country.

    Parameters:
        df (pd.DataFrame): Must contain 'scientificName', 'decimalLatitude', 'decimalLongitude'.
        species_name (str): Name of the species to plot. Plots the whole dataset if None.
        country_name (str): Name of the country.
        grid_resolution (float): Resolution of the grid in degrees.
        bandwidth (float): Bandwidth for Gaussian KDE.
    """
    # Filter for the species
    if species_name is not None:
        df_species = df[df['scientificName'] == species_name]
        if df_species.empty:
            raise ValueError(f"No records found for species '{species_name}'")
    else: df_species=df
    
    # Get the country grid (coordinates inside country polygon)
    country_coords = create_country_grid(country_name, grid_resolution=grid_resolution)
    
    # Convert torch tensor to numpy
    country_coords_np = country_coords.numpy()
    
    # Filter observations inside the country
    country_polygon = regionmask.defined_regions.natural_earth_v5_0_0.countries_10.polygons[
        list(regionmask.defined_regions.natural_earth_v5_0_0.countries_10.names).index(country_name)
    ]
    
    species_points = df_species[['decimalLongitude', 'decimalLatitude']].values
    species_points_inside = np.array([pt for pt in species_points if country_polygon.contains(Point(pt))])
    
    if len(species_points_inside) == 0:
        raise ValueError(f"No records for species '{species_name}' found inside {country_name}")
    elif len(species_points_inside) == 1:
        # Single-point fallback: constant zero density everywhere
        density = np.zeros(len(country_coords_np))
    else:
        # KDE density estimation
        kde = gaussian_kde(species_points_inside.T, bw_method=bandwidth)
        density = kde(country_coords_np.T)
    
    # # Plot
    # plt.figure(figsize=(8,6))
    # plt.scatter(country_coords_np[:,0], country_coords_np[:,1], c=density, s=50, cmap='viridis')
    # plt.colorbar(label='Density')
    # if plot_points:
    #     plt.scatter(species_points_inside[:,0], species_points_inside[:,1], c='red', s=2, alpha=0.5, label='Occurrences')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title(f"Spatial density of {species_name} in {country_name}")
    # plt.legend()
    # plt.show()
    # Create a regular grid over the map
    x_min, x_max = country_coords_np[:,0].min(), country_coords_np[:,0].max()
    y_min, y_max = country_coords_np[:,1].min(), country_coords_np[:,1].max()
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 200),
                    np.linspace(y_min, y_max, 200))
    grid_coords = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate KDE on the grid
    Z = kde(grid_coords).reshape(X.shape)

    # Plot density as a smooth surface
    plt.figure(figsize=(8,6))
    plt.imshow(Z, origin='lower', aspect='auto',
            extent=[x_min, x_max, y_min, y_max],
            cmap='viridis')
    plt.colorbar(label='Density')

    # Optionally overlay points
    if plot_points:
        plt.scatter(species_points_inside[:,0], species_points_inside[:,1],
                    c='red', s=2, alpha=0.5, label='Occurrences')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f"Spatial density of {species_name} in {country_name}")
    plt.legend()
    plt.show()

def get_species_emb(indices, dictionary, model,tokenizer, column_name="taxa_bioclip", device="cuda"):
    '''
    Takes in a np/torch array (n_batch,) of integers (indices).
    Returns a (n_batch, d_encoding) of (bioCLIP) species_embeddings for the corresponding indices 
    The column containing the name information has to be in the dictionary.

    bioclip is used with: 
    model, preprocess_train, processor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    model = model.to(device)
    model.eval()

    then ¨
    text_tokens = tokenizer([species_name])
    with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)
    '''
    np_idx=to_numpy(indices)
    species = dictionary[column_name].iloc[np_idx]
    tokens= tokenizer(species).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    return embeddings

