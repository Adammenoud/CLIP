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
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
from shapely import geometry as geom
import regionmask


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
        # Natural Earth polygons are MultiPolygon objects
        parts = list(polygon.geoms)

        # Metropolitan France lies roughly between lon -5 to 10 and lat 42 to 52
        bbox_metro = geom.box(minx=-10, miny=40, maxx=15, maxy=55)

        # Keep only polygons that intersect this box
        metro_parts = [p for p in parts if p.intersects(bbox_metro)]
        if len(metro_parts) == 0:
            raise RuntimeError("Could not isolate metropolitan France polygon.")

        polygon = geom.MultiPolygon(metro_parts)
    boundary = gpd.GeoSeries([polygon])

    fig, ax = plt.subplots(figsize=(8, 10))
    boundary.plot(ax=ax, color="none", edgecolor="black")
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=20)
    plt.colorbar(sc, ax=ax, label="Value")
    ax.set_title(f"Values over {country_name}")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def create_country_grid(country_name, grid_resolution=0.1):
    # Load 110m Natural Earth countries
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_10
    # Find the index of the country
    try:
        idx = list(countries.names).index(country_name)
    except ValueError:
        raise ValueError(f"Country '{country_name}' not found in regionmask.")

    polygon = countries.polygons[idx]
    if country_name=="France":
        # Natural Earth polygons are MultiPolygon objects
        parts = list(polygon.geoms)

        # Metropolitan France lies roughly between lon -5 to 10 and lat 42 to 52
        bbox_metro = geom.box(minx=-10, miny=40, maxx=15, maxy=55)

        # Keep only polygons that intersect this box
        metro_parts = [p for p in parts if p.intersects(bbox_metro)]

        if len(metro_parts) == 0:
            raise RuntimeError("Could not isolate metropolitan France polygon.")

        polygon = geom.MultiPolygon(metro_parts)
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

def do_and_plot_PCA(model, data_path,pca_file_name,nbr_components=None, nbr_plots=3, batch_size=4064, sort_duplicates=False,dictionary=None,country_name="Switzerland",save_path=None):

    if not hasattr(model, "img_encoder"): #compatibility issues
        img_encoder = nn.Sequential(model.lin1,model.relu,model.lin2)
    else:
        img_encoder=model.image_encoder

    dataloader, _ =dataloader_emb(data_path,batch_size=batch_size, shuffle=True,train_ratio=0.8, sort_duplicates=sort_duplicates, dictionary=dictionary)
    perform_PCA(dataloader, img_enc=img_encoder, device="cuda", file_name=pca_file_name, n_components=nbr_components, n_sample=None)

    pca_model_path=f"PCA_models/{pca_file_name}.pkl"
    for i in range(nbr_plots):
        plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=model.pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.01, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path)

def plot_PCA(pca_model_path,nbr_plots,pos_encoder,country_name="Switzerland",save_path=None):
    for i in range(nbr_plots):
        plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.01, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path)