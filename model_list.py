#import necessary for models
from py_compile import main
from xml.parsers.expat import model
import utils
import importlib
from geoclip import LocationEncoder, GeoCLIP
import open_clip
import torch
import nn_classes
import SDM_eval
import csv
import torch.nn as nn
import classifier_training
import pandas as pd
# imorts specific to the file
import glob
import torch
import re
import matplotlib.pyplot as plt
import os



#geoclip pos encoder model (my model)
data_path="embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioCLIP_full_dataset_embeddings.h5"
dim_emb=512
pos_encoder_geoclip_pos_enc =LocationEncoder(from_pretrained=False)
model_geoclip_pos_enc= nn_classes.DoubleNetwork_V2(pos_encoder=pos_encoder_geoclip_pos_enc, dim_hidden=768, dim_output=dim_emb).to("cuda")
state_dict = torch.load("Model_saves/geoclip_pos_enc/model.pt")
model_geoclip_pos_enc.load_state_dict(state_dict)
pos_enc_geoclip_pos_enc=model_geoclip_pos_enc.pos_encoder

#difference embeddings model
data_path="embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/difference_embeddings.h5"
dim_emb=512
pos_encoder_difference =LocationEncoder(from_pretrained=False)
model_difference= nn_classes.DoubleNetwork_V2(pos_encoder=pos_encoder_difference, dim_hidden=768, dim_output=dim_emb).to("cuda")
state_dict = torch.load("Model_saves/difference_embeddings/model.pt")
model_difference.load_state_dict(state_dict)
pos_enc_difference=model_difference.pos_encoder


#trained geoclip plocation encoder
pos_enc_geoclip_paper_trained =LocationEncoder(from_pretrained=True).to("cuda")

#untrained geoclip pos encoder
pos_enc_geoclip_paper_untrained =LocationEncoder(from_pretrained=False).to("cuda")

# downloaded_species_emb_VS_coords
device="cuda"
data_path="embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioCLIP_species_embeddings.h5"
dim_emb=512
pos_enc_species_emb = LocationEncoder(from_pretrained=False).to("cuda")
model_species_emb = nn_classes.DoubleNetwork_V2(pos_enc_species_emb, dim_hidden=768, dim_output=dim_emb).to(device)
state_dict = torch.load("Model_saves/downloaded_specie_embedding_VS_coords/model.pt")
model_species_emb.load_state_dict(state_dict)
pos_enc_species_emb=model_species_emb.pos_encoder

#small_pos_enc_correct_frequencies
pos_enc_correct_frequencies= nn_classes.Fourier_MLP(original_dim=2, fourier_dim=64, hidden_dim=256, output_dim=128).to("cuda")
model_small_pos_enc_correct_frequencies= nn_classes.DoubleNetwork_V2(pos_enc_correct_frequencies, dim_hidden=768, dim_output=128).to(device)
state_dict = torch.load("Model_saves/small_pos_enc_correct_frequencies/checkpoints/checkpoint_29/model.pt")
model_small_pos_enc_correct_frequencies.load_state_dict(state_dict)
pos_enc_correct_frequencies=model_small_pos_enc_correct_frequencies.pos_encoder

#classifier
fourier_dim=32
dictionary=pd.read_csv("embeddings_data_and_dictionaries/Embeddings/Bioclip_encoder/bioclip_data_dictionary_all_taxons")
class_name="scientificName"
n_species=len(dictionary[class_name].unique())
classifier = nn.Sequential(
        nn_classes.Fourier_enc(fourier_dim),
        nn_classes.MLP(
            in_dim=fourier_dim,
            hidden=[256, 256, 256],
            out_dim=n_species,
            drop_last=True
        )
    )
state_dict = torch.load("classifier_run_checkpoints/model.pt")
classifier.load_state_dict(state_dict)
classifier = classifier.to("cuda")

#actual goeclip:
dim_hidden=768
dim_output=512
image_encoder=nn.Sequential( nn.Linear(768, dim_hidden),nn.ReLU(),nn.Linear(dim_hidden, dim_output) )
model_actual_geoclip= GeoCLIP(from_pretrained=False)
model_actual_geoclip.image_encoder=image_encoder
state_dict = torch.load("Model_saves/actual_geoclip_long_run/model.pt")
model_actual_geoclip.load_state_dict(state_dict)
pos_enc_actual_geoclip=model_actual_geoclip.location_encoder.to("cuda")

#drop high sigma
model_drop_high_sigma=nn_classes.DoubleNetwork_V2(LocationEncoder(from_pretrained=False, sigma=[2**0, 2**4]))
state_dict = torch.load("/home/adam/source/CLIP/Model_saves/geoclip_pos_enc_drop_high_sigma/model.pt")
model_drop_high_sigma.load_state_dict(state_dict)
pos_encoder_drop_high_sigma=model_drop_high_sigma.pos_encoder.to("cuda")

#actual geoclip queue size = 1
dim_hidden=768
dim_output=512
image_encoder=nn.Sequential( nn.Linear(768, dim_hidden),nn.ReLU(),nn.Linear(dim_hidden, dim_output) )
model_queue_1= GeoCLIP(from_pretrained=False,queue_size=1)
model_queue_1.image_encoder=image_encoder.to("cuda")

pos_encoders_dict = {
    "geoclip_pos_enc": pos_enc_geoclip_pos_enc,
    "difference": pos_enc_difference,
    "geoclip_paper_untrained": pos_enc_geoclip_paper_untrained,
    "geoclip_paper_trained": pos_enc_geoclip_paper_trained,
    "species_emb": pos_enc_species_emb,
    "correct_frequencies": pos_enc_correct_frequencies,
    "classifier" : classifier,
    "actual_geoclip": pos_enc_actual_geoclip,
    "drop_high_sigma": pos_encoder_drop_high_sigma,
    #"queue_1" : 
}
#---------------------------------------------------------------    
#Checking all models on PO/PA data


def write_results_to_csv( pos_encoders_dict,data_callback, save_name="results.csv",do_pca=False, n_pca_components=None):
    '''takes a dictionary and evaluate model for each pos encoder
       (not generalised to other metrics)'''
    with open(save_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["pos_encoder", "auc_cov_PR", "auc_emb_PR", "auc_both_PR", "auc_cov_MLP", "auc_emb_MLP", "auc_both_MLP"]

        writer.writerow(header)

        for name, pos_encoder in pos_encoders_dict.items():
            print(f"Evaluating {name} model")
            results= SDM_eval.train_and_eval(
                pos_encoder=pos_encoder,
                do_pca=do_pca,
                n_pca_components=n_pca_components,
                hidden_size=[256, 256],
                epochs=200,
                train_MLP=True,
                data_callback=data_callback
                
                )
            writer.writerow([
                    name,
                    results["auc_cov_PR"],
                    results["auc_emb_PR"],
                    results["auc_both_PR"],
                    results["auc_cov_MLP"],
                    results["auc_emb_MLP"],
                    results["auc_both_MLP"]
                ])


def extract_step(path):
    match = re.search(r'checkpoint_(\d+)', path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No checkpoint number found in path: {path}")


def evaluate_checkpoints(model, checkpoint_dir,
                         save_path="AUC_over_training_checkpoints.png", data_callback=SDM_eval.get_data_geoplant):
    '''evaluates all checkpoints in a directory'''
    #get the relevant paths
    checkpoint_paths = []
    for d in os.listdir(checkpoint_dir):
        subdir = os.path.join(checkpoint_dir, d)
        model_path = os.path.join(subdir, "model.pt")
        if os.path.isdir(subdir) and "checkpoint_" in d:
            if os.path.isfile(model_path):
                checkpoint_paths.append(model_path)
    #print("checkpoints:" ,checkpoint_paths)
    
    # Sort by training step
    checkpoint_paths = sorted(checkpoint_paths, key=extract_step)
    
    auc_list_MLP = []
    auc_list_PR = []

    plt.figure(figsize=(8, 5))

    for i, ckpt_path in enumerate(checkpoint_paths):
        print("Loading checkpoint:", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
        model.eval()

        output = SDM_eval.train_and_eval(model.pos_encoder, train_MLP=True, data_callback=data_callback) #, data_callback=SDM_eval.get_data_geoplant)

        # Append results
        auc_list_MLP.append(output["auc_emb_MLP"])
        auc_list_PR.append(output["auc_emb_PR"])

        # Plot after each checkpoint
        plt.clf()  # clear previous plot
        plt.plot(range(1, i + 2), auc_list_MLP, marker='o', linestyle='-', color='blue', label="AUC emb MLP")
        plt.plot(range(1, i + 2), auc_list_PR, marker='x', linestyle='--', color='red', label="AUC emb PR")
        plt.xlabel("Checkpoint Step")
        plt.ylabel("AUC")
        plt.title("AUC over Training Checkpoints")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)  # overwrites each time

    plt.close()
    
    return auc_list_MLP, auc_list_PR

if __name__ == "__main__":
    print("evaluation")
    write_results_to_csv(pos_encoders_dict, data_callback=SDM_eval.get_data_geoplant,save_name="results/AUC/results_geoplant.csv",do_pca=False)


# evaluate_checkpoints(model_geoclip_pos_enc,
#                     #model_small_pos_enc_correct_frequencies,
#                     #checkpoint_dir="Model_saves/small_pos_enc_correct_frequencies/checkpoints",
#                     checkpoint_dir="/home/adam/source/CLIP/Model_saves/geoclip_pos_enc_10epochs/checkpoints",
#                         save_path="AUC_over_first_epochs_geoclip_pos_enc.png",
#                         data_callback=SDM_eval.get_data_geoplant)

    
