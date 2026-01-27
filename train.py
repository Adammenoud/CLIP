from torchvision import models
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import json
from datetime import datetime
from torchinfo import summary
import numpy as np
from geoclip import LocationEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import open_clip
import warnings
import wandb

import utils
import nn_classes
import data_extraction

def train(doublenetwork,
          epochs,
          dataloader,
          batch_size,
          lr=1e-4,
          device="cuda",
          save_name=None,
          saving_frequency=1,
          nbr_checkppoints=None, 
          test_dataloader=None,
          test_frequency=1,
          nbr_tests=10,
          modalities=["images","coords"], #either "images","coords","NCEAS" or "species". Has to be compatible with the model's forward
          dictionary=None,  #if want to use a different dictionary in the "species" case
          covariate_names= ["bcc","calc","ccc","ddeg","nutri","pday","precyy","sfroyy","slope","sradyy","swb","tavecc","topo"],
          detach_k_top=None
          ):
    '''
    Performs nbr_tests batchs of validation at the end of every test_frequency epochs.
    It most hold that nbr_chepoints < epochs

    "modalities" can be either "images","coords","NCEAS" or "species". The order is the order that will be used in the model's forward method.
    They are implemented using the "prepare_modality_tools" and "get_modalities" functions.
    
    "dictionary" is required only when working with the "species" modality.
    "covariate_names" is used only for the NCEAS covariates.
    '''
    #### initialization
    doublenetwork = doublenetwork.to(device)
    doublenetwork.train()
    writer =  SummaryWriter(log_dir=os.path.join("runs", f"{save_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))
    optimizer=torch.optim.AdamW(doublenetwork.parameters(),lr=lr)
    l = len(dataloader)
    current_checkpoint=1
    save_name="Model_saves/"+save_name #
    if test_dataloader is not None:#warning if test_dataloader too small
        n_available_batches=len(test_dataloader)
        if n_available_batches < nbr_tests:
            warnings.warn(
                f"Requested nbr_tests={nbr_tests}, "
                f"but test_dataloader only provides {n_available_batches} batches. "
                f"Running only {n_available_batches} tests.",
                RuntimeWarning,)
    # adjustments for other modalities than images, coords
    tokenizer, scaler, bioclip= prepare_modality_tools(modalities, covariate_names,device)
    #### training loop
    for ep in range(epochs):
        pbar = tqdm(dataloader)
        for i, (images, coords,idx) in enumerate(pbar):
            images = images.to(device)
            coords = coords.to(device)
                
            modality_list= get_modalities(modalities, images, coords, idx, covariate_names, dictionary, scaler, bioclip, tokenizer, device=device)

            #also change idx on test loop
            logits=doublenetwork(modality_list[0],modality_list[1])  #logits of cosine similarities
            loss=nn_classes.CE_loss(logits,device=device, detach_k_top=detach_k_top)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(CE=loss.item())
            writer.add_scalar("Cross-entropy", loss.item(), global_step=ep * l + i)
            metrics = get_metrics(logits)
            wandb.log(
                {
                    "Cross-entropy training": loss.item(),
                    "logit_scale": doublenetwork.logit_scale.item(),
                    **{f"train/{k}": v for k, v in metrics.items()},
                },
                step=ep * l + i
            )
        
    ################################# validation logs and checkpoints
        if ep % saving_frequency == 0 and (save_name is not None):
            os.makedirs(save_name, exist_ok=True)
            torch.save(doublenetwork.state_dict(), os.path.join(save_name, f"model.pt")) #overwrites the same file, so to avoid getting floded by saves
            torch.save(optimizer.state_dict(), os.path.join(save_name, f"optim.pt"))
            hparams = {                             #json
                    "save_name" : save_name,
                    "saving_frequency" : saving_frequency,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "current epoch (if stopped)" : ep,
                    "saving_frequency":saving_frequency,
                    "nbr_checkppoints":nbr_checkppoints, 
                    "test_frequency":test_frequency,
                    "nbr_tests": nbr_tests}
            config_path = os.path.join(save_name, "hyperparameters.json")
            with open(config_path, "w") as f:
                json.dump(hparams, f, indent=4)
            # summary_str = str(summary(doublenetwork, input_size=[images.shape, labels.shape],verbose=0 )) #txt
            # with open(os.path.join(save_name, "model_summary.txt"), "w", encoding="utf-8") as f:
            #     f.write(summary_str)
        if save_name is not None and nbr_checkppoints is not None and ep % (epochs//nbr_checkppoints)==0 and ep!= 0:
            checkpoint_name = f"{save_name}/checkpoints/checkpoint_{current_checkpoint}"
            os.makedirs(checkpoint_name, exist_ok=True)
            torch.save(doublenetwork.state_dict(), os.path.join(checkpoint_name, f"model.pt")) #overwrites the same file, so to avoid getting floded by saves
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_name, f"optim.pt"))
            current_checkpoint +=1
        if test_dataloader is not None and ep % test_frequency == 0:
            doublenetwork.eval()
            with torch.no_grad():
                total_loss = 0.0
                n_batches = 0
                metrics_accumulator = []

                for test_images, test_coords, test_idx in test_dataloader:
                    if n_batches >= nbr_tests:
                        break
                    test_images = test_images.to(device)
                    test_coords = test_coords.to(device)
                    test_modality_list= get_modalities(modalities, test_images, test_coords, test_idx, covariate_names, dictionary, scaler, bioclip, tokenizer, device=device)

                    test_logits = doublenetwork(test_modality_list[0], test_modality_list[1]) 
                    test_loss = nn_classes.CE_loss(test_logits, device=device)
                    total_loss += test_loss.item()

                    metrics_accumulator.append(get_metrics(test_logits))
                    n_batches += 1

                avg_loss = total_loss / max(1, n_batches)# the test dataloader my be too small sometimes
                writer.add_scalar("CE on test set", avg_loss, global_step=ep)

                avg_metrics = {}
                for k in metrics_accumulator[0]:
                        avg_metrics[k] = float(np.mean([m[k] for m in metrics_accumulator if k in m]))
                wandb.log(
                    {
                        "Cross-entropy validation": avg_loss,
                        **{f"validation/{k}": v for k, v in avg_metrics.items()},
                    },
                    step=ep * l + i
                )

        doublenetwork.train()
    #######################

    doublenetwork.eval()
    writer.close()
    wandb.finish()
    return doublenetwork





def get_modalities(modality_names, images, coords, idx, covariate_names, dictionary, scaler, bioclip, tokenizer, device="cuda"):
    '''
    Coords order: [latitude, longitude]
    '''
    #sanity checks:
    if len(modality_names) != 2:
        raise ValueError(f"Expected exactly 2 modalities, got {len(modality_names)}: {modality_names}")
    allowed_modalities = ["images", "coords", "NCEAS", "species"]
    invalid_mods = [m for m in modality_names if m not in allowed_modalities]
    if invalid_mods:
        raise ValueError(f"Invalid modality names: {invalid_mods}. Allowed: {allowed_modalities}")
    #getting the right modalities
    results = {}
    if "images" in modality_names:
        results["images"] = images  # already passed in, no computation
    
    if "coords" in modality_names:
        results["coords"] = coords
    if "NCEAS" in modality_names: # gets covariates instead of coordinates
        coords=coords.cpu().detach().numpy()
        NCEAS_covariates=utils.NCEAS_covariates(coords[:,1],coords[:,0], return_dict=True) #this function take lon, lat
        NCEAS_covariates = np.column_stack([NCEAS_covariates[cov] for cov in covariate_names])  #keeps the right order for scaling
        NCEAS_covariates=scaler.transform(NCEAS_covariates)
        results["NCEAS"] = torch.tensor(NCEAS_covariates).to(device)
    if "species" in modality_names:
        idx=idx.squeeze()
        results["species"]=utils.get_species_emb(idx,dictionary,bioclip,tokenizer)

    return [results[mod] for mod in modality_names]


def prepare_modality_tools(modalities, covariate_names, device="cuda"):
    tokenizer = None
    scaler = None
    bioclip = None
    if "NCEAS" in modalities: #scaling done over po data (simpler in terms of implementation, not ideal)
        po_data_path="embeddings_data_and_dictionaries/data_SDM_NCEAS/SWItrain_po.csv"
        po_data=pd.read_csv(po_data_path)
        po_covariates=po_data.loc[:,covariate_names ]
        X_cov=po_covariates.to_numpy()
        scaler = StandardScaler()
        scaler.fit(X_cov)
    if "species" in modalities:
        bioclip, preprocess_train, processor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
        bioclip = bioclip.to(device)
        bioclip.eval()
    return tokenizer, scaler, bioclip

def mean_rank(logits):
    targets = torch.arange(logits.size(0), device=logits.device)
    sorted_idx = logits.argsort(dim=1, descending=True)
    ranks = (sorted_idx == targets.unsqueeze(1)).nonzero()[:,1] + 1
    return ranks.float().mean()

def recall_at_k(logits, k=1):
    # logits: [B, B]
    targets = torch.arange(logits.size(0), device=logits.device)
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).float().mean()


def get_metrics(logits):
    """
    logits: [B, B] similarity matrix
    returns: dict[str, float]
    """
    B = logits.size(0)

    metrics = {
        "diag_sim": logits.diag().mean().item(),
        "off_diag_sim": (
            (logits.sum() - logits.diag().sum()) /
            (logits.numel() - B)
        ).item(),
        "mean_rank": mean_rank(logits).item(),
        "recall_at_1": recall_at_k(logits, 1).item(),
        "recall_at_10": recall_at_k(logits, min(10, B)).item(),
    }

    if B >= 50:
        metrics["recall_at_50"] = recall_at_k(logits, 50).item()
    if B >= 100:
        metrics["recall_at_100"] = recall_at_k(logits, 100).item()

    return metrics