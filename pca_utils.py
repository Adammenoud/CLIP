import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import joblib
from sklearn.preprocessing import StandardScaler
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
    joblib.dump(pca, f"PCA_models/{file_name}.pkl")



def coord_to_PCA(coords, pos_enc,pca_model_path, comp_idx=1): 
    '''takes coordinates and return the principal value associated to it. Considers the comp_idx-th component'''
    pca_model=joblib.load(pca_model_path)
    embeddings=pos_enc(coords).cpu()
    vect_to_plot = pca_model.transform(embeddings.detach().numpy())
    return vect_to_plot[:,comp_idx]

def do_and_plot_PCA(model, data_path,pca_file_name,nbr_components=None, nbr_plots=3, batch_size=4064, sort_duplicates=False,dataframe=None,country_name="Switzerland",save_path_pic=None):

    pos_encoder, img_encoder=utils.get_encoders(model)

    dataloader, _ =datasets.dataloader_factory(data_path,batch_size=batch_size, shuffle=True,train_ratio=0.8, sort_duplicates=sort_duplicates, dataframe=dataframe)
    perform_PCA(dataloader, img_enc=img_encoder, device="cuda", file_name=pca_file_name, n_components=nbr_components, n_sample=None)

    pca_model_path=f"PCA_models/{pca_file_name}.pkl"
    
    for i in range(nbr_plots):
        utils.plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.01, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path_pic)

def plot_PCA(pca_model_path,nbr_plots,pos_encoder,country_name="Switzerland",save_path=None):
    for i in range(nbr_plots):
        utils.plot_country_values(country_name=country_name, fct_to_plot=coord_to_PCA , pos_encoder=pos_encoder,pca_model_path=pca_model_path, grid_resolution=0.01, cmap='viridis',device="cuda",comp_idx=i,save_path=save_path)
