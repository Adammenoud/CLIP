import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import PoissonRegressor
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import yaml
from nn_classes import MLP
import os
import warnings
#local
import NCEAS
import nn_classes
    
def train_one_MLP(model, X, y, epochs=200, batch_size=250, lr=1e-4):
    model=model.to("cuda")
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,num_workers=4)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss().to("cuda")
    #loss_fn = deepmaxent_loss().to("cuda")

    pbar=range(epochs)
    for _ in tqdm(pbar):
        for xb, yb in dl:
            xb = xb.to("cuda")
            yb = yb.to("cuda")
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    model=model.cpu()
    return model
    
def fit_multi_GLM(X,y):
    PR_list=[]
    for i in tqdm(range(y.shape[1])):
        #PR
        PR=PoissonRegressor()
        PR.fit(X,y[:,i])
        PR_list.append(PR)
    return PR_list


def get_embeddings(coords, pos_encoder, device="cuda"):
    '''encodes with the pos_encoder. 
    The model should expect the order of coords to be (lat, lon)
    '''
    coords = torch.tensor(coords, dtype=torch.float32).to(device) 
    with torch.no_grad():
        emb = pos_encoder(coords).cpu().numpy()
    return emb

def get_data_NCEAS(po_data_path="Data/data_SDM_NCEAS/SWItrain_po.csv", 
                   pa_csv_path="Data/data_SDM_NCEAS/SWItest_pa.csv", 
                   env_csv_path="Data/data_SDM_NCEAS/SWItest_env.csv",
                   species_columns=[
                                    'swi01','swi02','swi03','swi04','swi05','swi06','swi07','swi08','swi09','swi10',
                                    'swi11','swi12','swi13','swi14','swi15','swi16','swi17','swi18','swi19','swi20',
                                    'swi21','swi22','swi23','swi24','swi25','swi26','swi27','swi28','swi29','swi30'],
                   covariates = ["bcc","calc","ccc","ddeg","nutri","pday","precyy","sfroyy","slope","sradyy","swb","tavecc","topo"]
                   ):

    '''
    Return the folowing np arrays: 
    return shape:
        X_cov, (n_po_samples, n_covariates)
        y,  (n_po_samples, n_species)
        X_test_cov, (n_pa_samples, n_covariates)
        y_true (n_pa_samples, n_species)
        coords (n_po_samples,2)

        coordinate order is lat, lon !! (what models take as input)
    '''
    
    po_data=pd.read_csv(po_data_path)
    pa_data=pd.read_csv(pa_csv_path)
    env=pd.read_csv(env_csv_path)

    y_test = pa_data.loc[:, species_columns]
    X_test_cov = env.loc[:,covariates]


    y = pd.get_dummies(po_data["spid"])
    po_data = pd.concat([po_data.drop(columns=["spid"]), y], axis=1) #For the full one-hot-encoded dataframe

    po_covariates=po_data.loc[:,covariates]

    X_cov=po_covariates.to_numpy()
    y=y.to_numpy()
    X_test_cov=X_test_cov.to_numpy()
    y_test=y_test.to_numpy()
    # Coords so we can get embeddings later
    lons_po, lats_po = NCEAS.coord_trans_shift(po_data["x"].values, po_data["y"].values,order="CH_to_normal")
    coords_po = np.column_stack((lats_po, lons_po))
    lons_pa, lats_pa = NCEAS.coord_trans_shift(pa_data["x"].values, pa_data["y"].values,order="CH_to_normal")   
    coords_pa = np.column_stack((lats_pa, lons_pa))

    return X_cov, y,coords_po, X_test_cov, y_test , coords_pa

def get_data_geoplant(
    po_csv_path="Data/filtered_geoplant/geoplant_po_france_withcovs.csv",
    pa_csv_path="Data/filtered_geoplant/geoplant_pa_france_withcovs.csv",
        ):
    """
        The covariates come form AlphaEarth, and the data form Geoplant (see /filtered_geoplant/filter_geoplant.py). 
        It would be also be interesting to have the 
        covariates from the geoplant rasters.
    Returns:
        X_cov        (n_po, n_covariates)
        y            (n_po, n_species)
        lonlat_po    (n_po, 2)

        X_test_cov   (n_pa, n_covariates)
        y_true       (n_pa, n_species)
        coords       (n_pa, 2)
    coordinate order is lat, lon !! (what models take as input)
    """
    po = pd.read_csv(po_csv_path)
    pa = pd.read_csv(pa_csv_path)
    covariate_columns = [c for c in po.columns if c.startswith("A")]
    species_columns = [c for c in po.columns if c.startswith("sp_")]
    lon_col = "lon"
    lat_col = "lat"

    # --- PO ---
    X_cov = po.loc[:, covariate_columns].to_numpy()
    y = po.loc[:, species_columns].to_numpy()
    coords_po = po.loc[:, [lat_col, lon_col]].to_numpy() #lat lon order

    # --- PA ---
    X_test_cov = pa.loc[:, covariate_columns].to_numpy()
    y_true = pa.loc[:, species_columns].to_numpy()
    coords_pa = pa.loc[:, [lat_col, lon_col]].to_numpy()

    return X_cov, y, coords_po, X_test_cov, y_true, coords_pa


def get_data_geoplant_corrected(
                X_po_path="Data/geoplant_corrected/X_po_france_covs.csv",
                y_po_path="Data/geoplant_corrected/Y_po_france.csv",

                X_pa_path="Data/geoplant_corrected/X_pa_france_covs.csv",
                y_pa_path="Data/geoplant_corrected/Y_pa_france.csv",
                #X : lont lat + Covs
                #y : species
                
        ):
    X_po=pd.read_csv(X_po_path)
    y_po=pd.read_csv(y_po_path)

    X_pa=pd.read_csv(X_pa_path)
    y_pa=pd.read_csv(y_pa_path)

    covariate_columns = [c for c in X_po.columns if c.startswith("A")]
    lon_col = "lon"
    lat_col = "lat"

    #Po
    X_cov=X_po.loc[:, covariate_columns].to_numpy()
    y=y_po.to_numpy()
    coords_po = X_po.loc[:, [lat_col, lon_col]].to_numpy()

    #Pa
    X_test_cov=X_pa.loc[:, covariate_columns].to_numpy()
    y_true=y_pa.to_numpy()
    coords_pa = X_pa.loc[:, [lat_col, lon_col]].to_numpy()

    return X_cov, y, coords_po, X_test_cov, y_true, coords_pa

def train_models(
        pos_encoder,
        do_pca=True,
        n_pca_components=None,
        hidden_size=[256, 256],
        epochs=200,
        train_MLP=True,
        data_callback=get_data_geoplant_corrected, #must be deterministic (also called in eval)
        PR_to_train = ["emb", "cov", "both"],
        MLP_to_train = ["emb", "cov", "both"]
):
    X_cov, y, coords_po, X_test_cov, y_true, coords_pa = data_callback()
    n_species = y.shape[1]
    n_cov= X_cov.shape[1]

    print("getting embeddings")
    full_list=PR_to_train+MLP_to_train
    if "emb" in full_list or "both" in full_list:
        X_emb=get_embeddings(coords_po, pos_encoder)
    else:
        X_emb=None
    #Normalize
    if X_emb is not None:
        scaler_emb = StandardScaler()
        scaler_emb.fit(X_emb)
        X_emb = scaler_emb.transform(X_emb)
    else:
        scaler_emb = None
    scaler_cov = StandardScaler()
    scaler_cov.fit(X_cov)
    X_cov = scaler_cov.transform(X_cov)
    #PCA (optional)
    if do_pca:
        pca = PCA(n_components=n_pca_components)
        X_emb = pca.fit_transform(X_emb)
    print("shape Xemb", X_emb.shape)
    print("shape Xcov", X_cov.shape)
    print("shape y",y.shape)

    # Fit PR
    print("fitting models")
    PR_cov=None
    PR_emb=None
    PR_both=None
    if "emb" in PR_to_train:
        PR_emb=fit_multi_GLM(X_emb,y)
    if "cov" in PR_to_train:   
        PR_cov =fit_multi_GLM(X_cov,y)
    if "both" in PR_to_train:
        PR_both=fit_multi_GLM(np.concatenate([X_emb,X_cov],axis=1),y)

    # Fit MLP
    MLP_emb=None
    MLP_cov=None
    MLP_both=None
    if "emb" in MLP_to_train:
        MLP_emb=MLP(in_dim=X_emb.shape[1], hidden=hidden_size, out_dim=n_species)
        MLP_emb = train_one_MLP(MLP_emb, X_emb, y,epochs=epochs)
    if "cov" in MLP_to_train:   
        MLP_cov=MLP(in_dim=n_cov, hidden=hidden_size, out_dim=n_species)
        MLP_cov = train_one_MLP(MLP_cov, X_cov, y,epochs=epochs)
    if "both" in MLP_to_train:
        MLP_both=MLP(in_dim=X_emb.shape[1]+n_cov, hidden=hidden_size, out_dim=n_species)
        MLP_both = train_one_MLP(MLP_both, np.concatenate([X_emb,X_cov],axis=1), y,epochs=epochs)

    #return dictionary
    trained_models = {
    "PR_emb": PR_emb,
    "PR_cov": PR_cov,
    "PR_both": PR_both,
    "MLP_emb": MLP_emb,
    "MLP_cov": MLP_cov,
    "MLP_both": MLP_both,
    "scaler_cov": scaler_cov,
    "scaler_emb": scaler_emb,
    "pca": pca if do_pca else None
    }
    return trained_models

def evaluate_models(
    pos_encoder, 
    trained_models,
    data_callback=get_data_geoplant_corrected
):
    """
    Flexible evaluation using the trained_models dictionary.
    Only evaluates models that exist in trained_models.
    """

    # Unpack data
    X_cov, y, coords_po, X_test_cov, y_true, coords_pa = data_callback()

    # Unpack trained objects
    PR_emb = trained_models.get("PR_emb")
    PR_cov = trained_models.get("PR_cov")
    PR_both = trained_models.get("PR_both")
    MLP_emb = trained_models.get("MLP_emb")
    MLP_cov = trained_models.get("MLP_cov")
    MLP_both = trained_models.get("MLP_both")
    scaler_cov = trained_models.get("scaler_cov")
    scaler_emb = trained_models.get("scaler_emb")
    pca_model = trained_models.get("pca")

    # Get embeddings for test points if needed
    X_test_emb = None
    if PR_emb is not None or PR_both is not None or MLP_emb is not None or MLP_both is not None:
        X_test_emb = get_embeddings(coords_pa, pos_encoder)

    # Normalize / PCA
    if scaler_cov is not None:
        X_test_cov = scaler_cov.transform(X_test_cov)
    if X_test_emb is not None:
        X_test_emb = scaler_emb.transform(X_test_emb)
        if pca_model is not None:
            X_test_emb = pca_model.transform(X_test_emb)

    output_shape = y_true.shape

    # Prepare predictions
    y_pred_cov = np.zeros(output_shape) if PR_cov is not None else None
    y_pred_emb = np.zeros(output_shape) if PR_emb is not None else None
    y_pred_both = np.zeros(output_shape) if PR_both is not None else None
    y_pred_cov_MLP = np.zeros(output_shape) if MLP_cov is not None else None
    y_pred_emb_MLP = np.zeros(output_shape) if MLP_emb is not None else None
    y_pred_both_MLP = np.zeros(output_shape) if MLP_both is not None else None

    # GLM predictions
    for i in range(output_shape[1]):
        if PR_cov is not None:
            y_pred_cov[:, i] = PR_cov[i].predict(X_test_cov)
        if PR_emb is not None:
            y_pred_emb[:, i] = PR_emb[i].predict(X_test_emb)
        if PR_both is not None:
            y_pred_both[:, i] = PR_both[i].predict(np.concatenate([X_test_emb, X_test_cov], axis=1))

    # MLP predictions
    if MLP_cov is not None:
        y_pred_cov_MLP = MLP_cov(torch.from_numpy(X_test_cov).float()).detach().numpy().squeeze()
    if MLP_emb is not None:
        y_pred_emb_MLP = MLP_emb(torch.from_numpy(X_test_emb).float()).detach().numpy().squeeze()
    if MLP_both is not None:
        y_pred_both_MLP = MLP_both(torch.from_numpy(np.concatenate([X_test_emb, X_test_cov], axis=1)).float()).detach().numpy().squeeze()

    # Compute AUCs
    results = {}
    if y_pred_cov is not None:
        results["auc_cov_PR"] = roc_auc_score(y_true, y_pred_cov)
        print("AUC cov PR:", results["auc_cov_PR"])
    if y_pred_emb is not None:
        results["auc_emb_PR"] = roc_auc_score(y_true, y_pred_emb)
        print("AUC emb PR:", results["auc_emb_PR"])
    if y_pred_both is not None:
        results["auc_both_PR"] = roc_auc_score(y_true, y_pred_both)
        print("AUC both PR:", results["auc_both_PR"])
    if y_pred_cov_MLP is not None:
        results["auc_cov_MLP"] = roc_auc_score(y_true, y_pred_cov_MLP)
        print("AUC cov MLP:", results["auc_cov_MLP"])
    if y_pred_emb_MLP is not None:
        results["auc_emb_MLP"] = roc_auc_score(y_true, y_pred_emb_MLP)
        print("AUC emb MLP:", results["auc_emb_MLP"])
    if y_pred_both_MLP is not None:
        results["auc_both_MLP"] = roc_auc_score(y_true, y_pred_both_MLP)
        print("AUC both MLP:", results["auc_both_MLP"])

    return results


def train_and_eval(
    pos_encoder,
    do_pca=True,
    n_pca_components=None,
    hidden_size=[256, 256],
    epochs=200,
    data_callback=get_data_geoplant_corrected,
    PR_to_train=["emb", "cov", "both"],
    MLP_to_train=["emb", "cov", "both"]
):
    # Train
    trained_models = train_models(
        pos_encoder=pos_encoder,
        do_pca=do_pca,
        n_pca_components=n_pca_components,
        hidden_size=hidden_size,
        epochs=epochs,
        data_callback=data_callback,
        PR_to_train=PR_to_train,
        MLP_to_train=MLP_to_train
    )
    # Evaluate
    results = evaluate_models(
        pos_encoder=pos_encoder,
        trained_models=trained_models,
        data_callback=data_callback
    )

    return results

# Why not only SDM_eval_from_folder ? -------------------------------------

def apply_callback_config(base_path, params_to_read, callback,**kwargs):
    """¨
    base_path (str): Path to the folder containing config.yaml and model.pt (both must be in the same folder)
    params_to_read (list): List of parameter keys to extract from config.yaml, e.g., ["drop_high_freq", "dataset"]
    callback (function): Function that takes a PyTorch model and returns something (most likely 'train_and_eval')
    **kwargs : any other arguments needed by the callback.
    """

    config_path = os.path.join(base_path, 'config.yaml')
    model_path = os.path.join(base_path, 'best_model.pt')
    #warnings
    if not os.path.isfile(config_path):
        warnings.warn(f"config.yaml not found at: {config_path}")
        return
    if not os.path.isfile(model_path):
        warnings.warn(f"model.pt not found at: {model_path}")
        model_path = os.path.join(base_path, 'best_model.ckpt') #try lightning checkpoint
        if not os.path.isfile(model_path):
            warnings.warn(f"model checkpoint not found at: {model_path}")
        return
    
    #load config, model
    with open(config_path) as f:
        cfg=yaml.safe_load(f)
    model=nn_classes.build_model(cfg)
    state_dict = torch.load(model_path, map_location='cuda',weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]  #if it is a lightning checkpoint instead, gets the statedict
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()} #gets rid of the model. added by lightning
    state_dict.pop('class_lookup', None) # Remove 'class_lookup' if it exists, since we build the model without it (we do not use the last layer)
    model.load_state_dict(state_dict)


    # Function to get nested parameters
    def get_nested(config_dict, key_path):
        keys = key_path.split('.')
        val = config_dict
        for k in keys:
            val = val[k]
        return val

    extracted_params = {param: get_nested(cfg, param) for param in params_to_read}

    if cfg["model_name"]=="contrastive": #get the right encoder
        model=model.pos_encoder
    else:
        model=model[0] #get the LocationEncoder (syntax from the Sequential definition)

    result = callback(model, **kwargs)

    # Save
    output_data = {
    'extracted_params': extracted_params,
    'result': result
    }
    return output_data

def SDM_eval_from_folder(base_path, params_to_read=["drop_high_freq", "dataset","vectors_name","model_name","use_species",],  
                         PR_to_train = ["emb"], 
                         MLP_to_train = ["emb"],
                         **kwargs):
    
    param_and_results=apply_callback_config(base_path,
                                  params_to_read,
                                  train_and_eval, 
                                  PR_to_train=PR_to_train,
                                  MLP_to_train=MLP_to_train,
                                  **kwargs
                                  )
    return param_and_results

##-----------------------



def run_SDM_on_superfolder(
    super_folder,
    output_csv,
    **sdm_kwargs
):
    '''
    takes a super folder and test with SDM each model saved inside.
    must be like :superfolder-model_1-model.pt
                                     -config.yaml
                             -model_2 ..
    '''
    all_results = []



    folders = [
        os.path.join(super_folder, d)
        for d in os.listdir(super_folder)
        if os.path.isdir(os.path.join(super_folder, d))
    ]

    for folder in folders:
        print(f"\n=== Running SDM on: {folder} ===")
        try:
            output_data = SDM_eval_from_folder(
                base_path=folder,
                **sdm_kwargs
            )

            extracted_params = output_data.get("extracted_params", {})
            result = output_data.get("result", {})

            row = {
                "run_name": os.path.basename(folder), #keep the last one
                **extracted_params,
                **result,
                }
            all_results.append(row)

        except Exception as e:
            print(f"Failed on {folder}: {e}")

    df = pd.DataFrame(all_results)

    # append instead of overwrite
    try:
        old_df = pd.read_csv(output_csv)
        final_df = pd.concat([old_df, df], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # Either file doesn't exist or is empty/invalid
        final_df = df

    final_df.to_csv(output_csv, index=False)
    print(f"Results appended to {output_csv}")
    return final_df

if __name__ == "__main__":
    run_SDM_on_superfolder("Model_saves/sweep_classifier","results.csv")
    run_SDM_on_superfolder("Model_saves/sweep_classifier_emb","results.csv")
    run_SDM_on_superfolder("Model_saves/sweep_contrastive_images","results.csv")
    run_SDM_on_superfolder("Model_saves/sweep_contrastive_species","results.csv")
    run_SDM_on_superfolder("Model_saves/sweep_mixed_embeddings","results_mixed.csv",
                           params_to_read=["drop_high_freq", "dataset","vectors_name","model_name",
                                           "use_species","run_name_clean","mixed_data_method","mixed_embeddings"])

