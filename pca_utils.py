import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
#local
import datasets
import utils


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
    print("component vector:" ,pca.components_)
    joblib.dump(pca, f"PCA_models/{file_name}.pkl")



def coord_to_PCA(coords, pos_enc,pca_model_path, comp_idx=1): 
    '''takes coordinates and return the principal value associated to it. Considers the comp_idx-th component'''
    pca_model=joblib.load(pca_model_path)
    embeddings=pos_enc(coords).cpu()
    embeddings=F.normalize(embeddings, p=2, dim=1)
    vect_to_plot = pca_model.transform(embeddings.detach().numpy())
    return vect_to_plot[:,comp_idx]

def do_and_plot_PCA(model, data_path,config,pca_file_name,nbr_components=None, nbr_plots=3,country_name="Switzerland",save_path_pic=None, n_sample=None,grid_resolution=0.03):

    pos_encoder, img_encoder=utils.get_encoders(model)

    dataloader, _ , _=datasets.dataloader_factory(file_path=data_path,config=config)
    perform_PCA(dataloader, img_enc=img_encoder, device="cuda", file_name=pca_file_name, n_components=nbr_components, n_sample=n_sample)

    pca_model_path=f"PCA_models/{pca_file_name}.pkl"
    
    for i in range(nbr_plots):
        utils.plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=grid_resolution, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path_pic)



def plot_PCA(pca_model_path,nbr_plots,pos_encoder,country_name="Switzerland",save_path=None):
    for i in range(nbr_plots):
        utils.plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.03, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path)


if __name__ == "__main__":
    import nn_classes
    import yaml
    from geoclip import LocationEncoder
    base_path="Model_saves/3dimensional_embedding_space/3dimensional_embedding_space"
    config_path=base_path+"/config.yaml"
    model_path=base_path+"/best_model.pt"
    with open(config_path) as f:
        config=yaml.safe_load(f)

    loc_encoder = nn.Sequential(LocationEncoder(sigma=[2**0, 2**4, 2**8], from_pretrained=False), nn.Linear(512,3))
    model = nn_classes.DoubleNetwork_V2(loc_encoder, dim_output=3)
    model = model.to("cuda")
    statedict = torch.load("Model_saves/3dimensional_embedding_space/3dimensional_embedding_space/model.pt", map_location='cuda')
    model.load_state_dict(statedict)



    do_and_plot_PCA(model=model,
                    data_path="Embeddings_and_dataframes/plants/embeddings_inaturalist_FR_plants.h5",
                    config=config,
                    pca_file_name="3d_visualization_PCA",
                    nbr_components=1,
                    nbr_plots=1,
                    country_name="France",
                    save_path_pic="FOLDER/pca_3d_visualization_1component",
                    n_sample=None,
                    grid_resolution=0.03,
                    )