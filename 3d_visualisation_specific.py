import numpy as np
import plotly.graph_objects as go
import yaml
import pandas as pd
from geoclip import LocationEncoder
import torch.nn as nn
import torch
from tqdm import tqdm
import plotly.express as px
from PIL import Image
import io
import base64
import torch.nn as nn
import os

import nn_classes
import datasets
import utils

data_path = "Embeddings_and_dataframes/plants/embeddings_inaturalist_FR_plants.h5"
dict_path = "/home/adam/source/CLIP/Embeddings_and_dataframes/plants/dictionary_inaturalist_FR_plants"

image_path="Data/swiss_images/all_images/17_2645281974.jpg"

save_name="near_random_points"

with open("config_3d.yaml") as f:
    cfg = yaml.safe_load(f)

k=5
dataframe=pd.read_csv(dict_path)
top_k = dataframe["species"].value_counts().nlargest(k).index
top_k = top_k.append(pd.Index(["Euphorbia peplis"]))
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

#specific image: Augustifollium
image = Image.open(image_path).convert("RGB")
emb_img = utils.embedds_image(image)
emb_img = img_enc(emb_img)
emb_img = emb_img.detach().cpu().numpy().squeeze()
emb_img = emb_img / np.linalg.norm(emb_img)
#other embedding: 


#dataloader
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

# Example: highlight several species
highlight_species = ["Euphorbia peplis"]#["Chamaenerion angustifolium"]

# Assign colors: red for highlights, gray for the rest
colors = ["red" if sp in highlight_species else "blue" for sp in full_species]

# Optional: make highlighted species slightly bigger
sizes = [5 if sp in highlight_species else 2 for sp in full_species]

fig = go.Figure(go.Scatter3d(
    x=full_embeddings[:, 0],
    y=full_embeddings[:, 1],
    z=full_embeddings[:, 2],
    mode='markers',
    marker=dict(
        size=sizes,
        color=colors
    ),
    text=full_species  # hover shows species
))

fig.update_layout(
    title=f"Highlighting {', '.join(highlight_species)}",
    scene_aspectmode='data'
)


fig.add_trace(go.Scatter3d(
    x=[emb_img[0]],
    y=[emb_img[1]],
    z=[emb_img[2]],
    mode='markers',
    marker=dict(
        size=8,      # bigger so it stands out
        color='green'
    ),
    text=["Query Image"]  # hover text
))
###### Images visualization
images_save_path=f"{save_name}_images"
os.makedirs(images_save_path,exist_ok=True)

class dot_product(nn.Module):
    def __init__(self,pos_encoder):
        super().__init__()
        self.pos_encoder=pos_encoder

    def forward(self, image_emb, coordinates): #takes a batch of images and their labels
        '''
        returns the dot product of the normalized images vectors with the vectors from the encodding of the coordinates (with the pos_encoder).
        Can be used to compute directly the similarity as a callback in map_embeddings
        '''
        pos_emb=self.pos_encoder(coordinates)
        
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        pos_emb = pos_emb / pos_emb.norm(dim=1, keepdim=True)
        # Compute cosine similarity (dot product here, since vectors are normalized)
        logits = image_emb @ pos_emb.t()
        return logits
dot_prod=dot_product(pos_enc)

points=points=np.random.normal(size=(100,3))
#--- Take noise around Augustifollium instead of uniform
points=emb_img+ 0.1*points
#---

points /= np.linalg.norm(points,axis=1, keepdims=True)
hover_imgs=[]
print("clicking maps:")
for i,p in enumerate(points):
    vector=torch.tensor(p, device="cuda",dtype=torch.float32).unsqueeze(0)
    img=utils.map_embedding(dot_prod, vector, country = "France", grid_resolution=0.03,return_pil=True, vmin=-1,vmax=1)
    path=f"{images_save_path}/p{i:04d}.png"
    img.save(path)
    hover_imgs.append(path)


fig.add_trace(
    go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode="markers",
        marker=dict(size=4, color="purple"),
        customdata=hover_imgs,
        name="Embeddings"
    )
)

html=fig.to_html(full_html=True, include_plotlyjs="cdn")

panel = """
<div id="img-box" style="
    position:fixed;
    top:10px;
    right:10px;
    width:350px;
    background:white;
    border:1px solid #aaa;
    padding:10px;
    z-index:1000;">
    <h4>Clicked point</h4>
    <img id="img-view" style="width:100%;" />
</div>

<script>
document.getElementsByClassName('plotly-graph-div')[0]
  .on('plotly_click', function(data){
      let img = data.points[0].customdata;
      document.getElementById("img-view").src = img;
  });
</script>
"""
html=html.replace("</body>",panel+"</body>")

with open(f"{save_name}_plot.html", "w") as f:
    f.write(html)

# fig.write_html(save_name)
# print(f"Saved to {save_name}")
