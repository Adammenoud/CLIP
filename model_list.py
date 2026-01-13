#import necessary for models
import utils
import importlib
from geoclip import LocationEncoder
import open_clip
import torch
import nn_classes
import SDM_eval
import csv
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

#actual geoclip?

pos_encoders_dict = {
    "pos_enc_geoclip_pos_enc": pos_enc_geoclip_pos_enc,
    "pos_enc_difference": pos_enc_difference,
    "pos_enc_geoclip_paper_untrained": pos_enc_geoclip_paper_untrained,
    "pos_enc_geoclip_paper_trained": pos_enc_geoclip_paper_trained,
    "pos_enc_species_emb": pos_enc_species_emb,
    "pos_enc_correct_frequencies": pos_enc_correct_frequencies
}
#---------------------------------------------------------------    
#Checking all models on PO/PA data


def write_results_to_csv( pos_encoders_dict,data_callback, save_name="results.csv"):
    '''takes a dictionary and evaluate model for each pos encoder
       (not yet generalised)'''
    with open(save_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Write header
        header = ["pos_encoder", "auc_cov_PR", "auc_emb_PR", "auc_both_PR", "auc_cov_MLP", "auc_emb_MLP", "auc_both_MLP"]

        writer.writerow(header)

        for name, pos_encoder in pos_encoders_dict.items():
            print(f"Evaluating {name} model")
            results= SDM_eval.train_and_eval(
                pos_encoder=pos_encoder,
                do_pca=False,
                n_pca_components=None,
                hidden_size=[256, 256],
                epochs=200,
                train_MLP=True,
                data_callback=data_callback)
            writer.writerow([
                    name,
                    results["auc_cov_PR"],
                    results["auc_emb_PR"],
                    results["auc_both_PR"],
                    results["auc_cov_MLP"],
                    results["auc_emb_MLP"],
                    results["auc_both_MLP"]
                ])


print("Evaluating checkpoints for geoclip_pos_enc model")
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
    
    # Lists to store results
    auc_list_MLP = []
    auc_list_PR = []

    plt.figure(figsize=(8, 5))  # prepare figure for runtime updates

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

#write_results_to_csv(pos_encoders_dict, data_callback=SDM_eval.get_data_geoplant) #NCEAS data by default
evaluate_checkpoints(model_geoclip_pos_enc,
                    #model_small_pos_enc_correct_frequencies,
                    #checkpoint_dir="Model_saves/small_pos_enc_correct_frequencies/checkpoints",
                    checkpoint_dir="Model_saves/geoclip_pos_enc/Checkpoints",
                     save_path="AUC_over_training_checkpoints_geoclip_enc_NCEAS.png",
                     data_callback=SDM_eval.get_data_NCEAS)
