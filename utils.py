import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import data_extraction
from tqdm import tqdm
import h5py
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import MultiPolygon
from shapely import geometry as geom
import regionmask
import open_clip
import pandas as pd
import wandb
import os
import nn_classes
import yaml
from scipy.stats import gaussian_kde
import re
#local
import train_contrastive
import datasets






    



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





def test_similarity(data_file_name, doublenetwork, nbr_iter=1000, nbr_samples=2,device="cuda", plot_sims=True,sort_duplicates=True, dataframe_path=None, modalities=["images","coords"]):
    '''Picks random samples and gives the (average) similarity between the image emmbedding and the coordinate embedding
    dataframe is used to get the dataloader. It can be set to None, together with
    '''
    tokenizer, scaler, bioclip= train_contrastive.prepare_modality_tools(modalities)
    if dataframe_path is not None:
        dataframe=pd.read_csv(dataframe_path)
    else:
        dataframe=None
    _ , dataloader=datasets.dataloader_factory(data_file_name,batch_size=nbr_samples,shuffle=True,sort_duplicates=sort_duplicates, dataframe=dataframe) #test set
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
        mod= train_contrastive.get_modalities(modalities, emb_vectors, coord, idx, ["bcc","calc","ccc","ddeg","nutri","pday","precyy","sfroyy","slope","sradyy","swb","tavecc","topo"], dataframe, scaler, bioclip, tokenizer, device="cuda")
        logits=doublenetwork(mod[0],mod[1])

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



    


def plot_country_values(country_name, fct_to_plot,pos_encoder,pca_model_path="PCA_All_comp_full_dataset_Normalized.pkl", grid_resolution=0.1, cmap='viridis',device="cuda",comp_idx=0,save_path=None,vmin=None,vmax=None):
    coords = create_country_grid(country_name, grid_resolution)
    values = fct_to_plot(coords.to(device), pos_encoder,pca_model_path,comp_idx=comp_idx)#np.array([coord_trans(coord) for coord in coords])
    # Get the country polygon
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    idx = list(countries.names).index(country_name)
    polygon = countries.polygons[idx]
    if country_name=="France":
        polygon=filter_France(polygon)
    boundary = gpd.GeoSeries([polygon])
    plot_values(coords,values,boundary,save_path,country_name,vmin=vmin,vmax=vmax)

def create_country_grid(country_name, grid_resolution=0.1):
    '''Returns the coordinates of points inside the country. 
        returns: torch tensor of size (nbr_points,2) with order lat, lon for the 2nd dimension
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
    '''
    Takes the polygon of France (including overseas) and returns only the part corresponding to metropolitan France.
    '''
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


def get_gbif_covariates(dataframe, idx_list, covariate_list=['scientificName']):
    '''looks into the dataframe to get the data corespondint to the indices.
        the covariates must be columns in the dataframe.
        retrun: (len(idx_list), len(covariate_list)) array
    '''
    # Ensure all requested covariates exist in the DataFrame
    missing_cols = [col for col in covariate_list if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"The following covariates are missing in the DataFrame: {missing_cols}")
    # Select rows and columns, convert to numpy array
    selected_data = dataframe.loc[idx_list, covariate_list].to_numpy()
    return selected_data


def apply_pos_enc(coords, pos_encoder,pca_model_path=None,comp_idx=None):
    return pos_encoder(coords).cpu()


def map_image(doublenetwork, image, country="Switzerland", device="cuda", grid_resolution=0.1, save_path=None,vmin=None,vmax=None):
    '''
    Plots the similarity score of each coordinate with the image.
    image can be either a path or a PIL image.
    If no save_path is given, uses plt.show() directly
    '''
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError(
            "path_or_image must be a file path (str or Path) or a PIL.Image.Image"
        )
    emb_img = embedds_image(image)
    map_embedding(doublenetwork, emb_img, country=country, device=device, grid_resolution=grid_resolution, save_path=save_path,vmin=vmin,vmax=vmax)
    

def embedds_image(image, device="cuda"):
    '''Fetches the model every time: only for plots or exceptional use, do not call repeatedly!'''
    bioclip, preprocess_train, processor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    image= processor(image).unsqueeze(0).to(device)
    bioclip = bioclip.to(device)
    bioclip.eval()
    emb_img = bioclip.encode_image(image)
    return emb_img

def map_embedding(doublenetwork, embedding, country="Switzerland", device="cuda", grid_resolution=0.1, save_path=None,vmin=None,vmax=None):
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
    plot_values(coords,values,boundary,save_path,country,vmin=vmin,vmax=vmax)

def plot_values(coords,values,boundary,save_path,country,title=None,vmin=None,vmax=None):
    '''
    Plots the values at the coordinates, with the country boundary.

    coords: (n_points, 2) array of lat, lon. Can be a torch tensor or a numpy array.
    values: (n_points,) array of values to plot. Can be a torch tensor or a numpy array.
    boundary: GeoSeries containing the country boundary to plot.
    save_path: if None, uses plt.show(), otherwise saves the figure to the given path
    '''
    coords=to_numpy(coords)
    values=to_numpy(values)
    fig, ax = plt.subplots(figsize=(8, 10))
    boundary.plot(ax=ax, color="none", edgecolor="black")
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap='viridis', s=1, alpha=0.5,vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="Value")
    if title is None:
        title = f"Image similarity over {country}"
    ax.set_title(title)
    if save_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    

def to_numpy(x):
    '''
    Converts a torch tensor to a numpy array, does nothing to a numpy array.
    Used to handle both types.
    '''
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")


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
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(country_coords_np[:,0], country_coords_np[:,1], c=density, s=50, cmap='viridis')
    plt.colorbar(label='Density')
    if plot_points:
        plt.scatter(species_points_inside[:,0], species_points_inside[:,1], c='red', s=2, alpha=0.5, label='Occurrences')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f"Spatial density of {species_name} in {country_name}")
    plt.legend()
    plt.show()
    


def get_species_emb(indices, dataframe, model,tokenizer, column_name="taxa_bioclip", device="cuda"):
    '''
    Takes in a np/torch array (n_batch,) of integers (indices).
    Returns a (n_batch, d_encoding) of (bioCLIP) species_embeddings for the corresponding indices

    The column containing the name information has to be in the dataframe.
    '''
    np_idx=to_numpy(indices)
    species = dataframe[column_name].iloc[np_idx]
    tokens= tokenizer(species).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    return embeddings

def get_example(dataset_path, idx,vector_name="vectors_bioclip"):
    with h5py.File(dataset_path, "r") as f:
        vector = f[vector_name][idx]  # <-- access the i-th vector directly
    return vector


def get_run_name(base_name, cfg, sweep_keys):
    '''
    returns the base name with the parameters concatenated.
    (Used for wandb sweep, to distinguish the runs more easily.)
    '''
    if wandb.run.sweep_id is None:
        return base_name
    
    parts = [base_name]
    for key in sweep_keys:
        val = cfg[key]
        if isinstance(val, bool):
            parts.append(f"{key}_{val}")
        else:
            parts.append(str(val))
    return "_".join(parts)

def plot_image_from_index(dataframe, index, return_only=False):
    '''
    Downloads and plots the image corresponding to the given index in the data dataframe.
    Either returns the PIL image (if return_only=True) or plots it directly.
    '''
    url = dataframe.iloc[index]["identifier"]
    response = data_extraction.get(url)
    response.raise_for_status()

    image = Image.open(response.raw).convert("RGB")
    if return_only:
        return image
    plt.imshow(image)
    plt.axis("off")
    plt.show()



def configs_from_folder(checkpoint_folder,sort=True):
    '''
    Works only for the contrastive folder name
    Assumes folders are named checkpoint_i
    '''
    checkpoint_paths_model = []

    def extract_index(name):
        m = re.search(r'checkpoint_(\d+)', name)
        return int(m.group(1)) if m else float('inf')

    dirs=os.listdir(checkpoint_folder)
    if sort:
        dirs = sorted(dirs,key=extract_index)

    for d in dirs:
        subdir = os.path.join(checkpoint_folder, d)
        model_path = os.path.join(subdir, "model.pt")
        if os.path.isdir(subdir) and "checkpoint_" in d:
            if os.path.isfile(model_path):
                checkpoint_paths_model.append(model_path)

    return checkpoint_paths_model

def video_similarity(checkpoint_folder,config_path,image,save_path=None,vmin=None,vmax=None):
    '''
    goes through the checkpoints of the folder and get the similarity map for each model.
    Works only for the specific folder checkpoint format of this project.
    image can be either a path or a PIL image.
    '''
    models= configs_from_folder(checkpoint_folder) #paths
    with open(config_path) as f:
        cfg=yaml.safe_load(f)
    model= nn_classes.build_model(cfg)

    for i in range(len(models)):
        print(models[i])
        state_dict = torch.load(models[i], map_location='cuda',weights_only=True)
        model.load_state_dict(state_dict)
        if save_path is None:
            output_dir=None
        else:
            output_dir=os.path.join(save_path, f"{i}")
        map_image(model, image, country="France", device="cuda", grid_resolution=0.03, save_path=output_dir,vmin=vmin,vmax=vmax)