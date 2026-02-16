import numpy as np
import plotly.graph_objects as go
import yaml
import pandas as pd
from geoclip import LocationEncoder
import torch.nn as nn
import torch
from tqdm import tqdm
import plotly.express as px


import nn_classes
import datasets
import utils

data_path = "Embeddings_and_dataframes/plants/embeddings_inaturalist_FR_plants.h5"
dict_path = "/home/adam/source/CLIP/Embeddings_and_dataframes/plants/dictionary_inaturalist_FR_plants"

with open("config_copy.yaml") as f:
    cfg = yaml.safe_load(f)

k=5
dataframe=pd.read_csv(dict_path)
top_k = dataframe["species"].value_counts().nlargest(k).index
dataframe = dataframe[dataframe["species"].isin(top_k)]


dataloader, test_dataloader, dataset_type =datasets.dataloader_factory(data_path,
                                                  cfg,
                                                  dataframe=dataframe,
                                                  )

loc_encoder = nn.Sequential(LocationEncoder(sigma=[2**0, 2**4, 2**8], from_pretrained=False), nn.Linear(512,3))
model = nn_classes.DoubleNetwork_V2(loc_encoder, dim_output=3)
model = model.to("cuda")
statedict = torch.load("Model_saves/3dimensional_embedding_space/3dimensional_embedding_space/model.pt", map_location='cuda')
model.load_state_dict(statedict)
model.eval()
pos_enc, img_enc = utils.get_encoders(model)
#location_encoder= model.pos_encoder

counter=0
full_embeddings = []
full_species = []

pbar = tqdm(dataloader)
for (images, _, idx) in pbar:
    images = images.to("cuda")

    embeddings = img_enc(images)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    full_embeddings.append(embeddings.detach().cpu().numpy())

    # Get species names from dataframe
    species = dataframe.iloc[idx]["species"].values
    full_species.extend(species)
    # if counter > 0:
    #     break
    # counter+=1

full_embeddings = np.vstack(full_embeddings)
full_species = np.array(full_species)

print(full_embeddings.shape)
print(full_species.shape)
x = full_embeddings[:,0]
y = full_embeddings[:,1]
z = full_embeddings[:,2]

unique_species = np.unique(full_species)

color_map = dict(zip(
    unique_species,
    px.colors.qualitative.Dark24[:len(unique_species)]
))

colors = [color_map[s] for s in full_species]

fig = go.Figure(go.Scatter3d(
    x=full_embeddings[:, 0],
    y=full_embeddings[:, 1],
    z=full_embeddings[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=colors
    ),
    text=full_species,  # hover shows species
))

fig.update_layout(
    title="3D Embedding Space Colored by Species",
    scene_aspectmode='data'
)

fig.write_html("epoch.html")
