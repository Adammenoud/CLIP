import utils
import nn_classes
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
    
def train_one_MLP(model, X, y, epochs=200, batch_size=250, lr=1e-4):
    model=model.to("cuda")
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.BCEWithLogitsLoss().to("cuda")
    #loss_fn = deepmaxent_loss().to("cuda")

    pbar=range(epochs)
    for _ in tqdm(pbar):
        for xb, yb in dl:
            xb = xb.to("cuda")
            yb = yb.to("cuda")
            pred = model(xb)
            loss = loss_fn(pred, yb).to("cuda")
            optim.zero_grad()
            loss.backward()
            optim.step()
    model=model.cpu()
    return model


class deepmaxent_loss(nn.Module):
    def __init__(self):
        super(deepmaxent_loss, self).__init__()
    def forward(self, input, target):
        loss = -((target)*(input.log_softmax(0))).mean(0).mean()
        return loss
    
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
    lons_po, lats_po = utils.coord_trans_shift(po_data["x"].values, po_data["y"].values,order="CH_to_normal")
    coords_po = np.column_stack((lats_po, lons_po))
    lons_pa, lats_pa = utils.coord_trans_shift(pa_data["x"].values, pa_data["y"].values,order="CH_to_normal")   
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
        data_callback=get_data_geoplant_corrected #must be deterministic (also called in eval)
):
    X_cov, y, coords_po, X_test_cov, y_true, coords_pa = data_callback()
    n_species = y.shape[1]
    n_cov= X_cov.shape[1]

    print("getting embeddings")


    X_emb=get_embeddings(coords_po, pos_encoder)
    #Normalize
    scaler_emb = StandardScaler()
    scaler_emb.fit(X_emb)
    X_emb = scaler_emb.transform(X_emb)
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
    # Fit GLM
    print("fitting models")


    PR_emb=fit_multi_GLM(X_emb,y)
    PR_cov =fit_multi_GLM(X_cov,y)
    PR_both=fit_multi_GLM(np.concatenate([X_emb,X_cov],axis=1),y)
    if train_MLP:
        MLP_emb=MLP(in_dim=X_emb.shape[1], hidden=hidden_size, out_dim=n_species)
        MLP_emb = train_one_MLP(MLP_emb, X_emb, y,epochs=epochs)

        MLP_cov=MLP(in_dim=n_cov, hidden=hidden_size, out_dim=n_species)
        MLP_cov = train_one_MLP(MLP_cov, X_cov, y,epochs=epochs)

        MLP_both=MLP(in_dim=X_emb.shape[1]+n_cov, hidden=hidden_size, out_dim=n_species)
        MLP_both = train_one_MLP(MLP_both, np.concatenate([X_emb,X_cov],axis=1), y,epochs=epochs)
    else:
        MLP_emb=None
        MLP_cov=None
        MLP_both=None

    return PR_emb, PR_cov, PR_both, MLP_emb, MLP_cov, MLP_both, scaler_cov, scaler_emb, pca if do_pca else None

def evaluate_models(
    pos_encoder, 
    scaler_cov, 
    scaler_emb, 
    PR_cov, #from fit_multi_GLM
    PR_emb, #from fit_multi_GLM
    PR_both, #from fit_multi_GLM
    MLP_cov, #from train_one_MLP : shape (n_samples, n_species)
    MLP_emb,  #from train_one_MLP
    MLP_both,  #from train_one_MLP
    pca_model=None, #to lower dim of X_test_emb.
    train_MLP=True,
    data_callback=get_data_geoplant_corrected    
):
    """
    Evaluation on PA data
    """

    X_cov, y, coords_po, X_test_cov, y_true, coords_pa = data_callback()


    # pa_data=pd.read_csv(pa_csv_path)
    # env=pd.read_csv(env_csv_path)
    # y_true = pa_data.loc[:, species_columns]
    # X_test_cov = env.loc[:,covariates]
    # X_test_cov=X_test_cov.to_numpy()
    # y_true=y_true.to_numpy()

    X_test_emb=get_embeddings(coords_pa,pos_encoder)

    X_test_cov = scaler_cov.transform(X_test_cov)#norm
    X_test_emb = scaler_emb.transform(X_test_emb)
    if pca_model is not None:
        X_test_emb= pca_model.transform(X_test_emb)#pca

    output_shape=y_true.shape
    y_pred_cov=np.zeros(output_shape)
    y_pred_emb=np.zeros(output_shape)
    y_pred_both=np.zeros(output_shape)
    y_pred_cov_MLP=np.zeros(output_shape)
    y_pred_emb_MLP=np.zeros(output_shape)
    y_pred_both_MLP=np.zeros(output_shape)
    for i in range(output_shape[1]):
        y_pred_cov[:,i]=PR_cov[i].predict(X_test_cov)
        y_pred_emb[:,i]=PR_emb[i].predict(X_test_emb)
        y_pred_both[:,i]=PR_both[i].predict(np.concatenate([X_test_emb,X_test_cov],axis=1))
    if train_MLP:
        y_pred_cov_MLP=MLP_cov(torch.from_numpy(X_test_cov).float()).detach().numpy().squeeze()
        y_pred_emb_MLP=MLP_emb(torch.from_numpy(X_test_emb).float()).detach().numpy().squeeze()
        y_pred_both_MLP=MLP_both(torch.from_numpy(np.concatenate([X_test_emb,X_test_cov],axis=1)).float()).detach().numpy().squeeze()

    print(y_true.shape, y_pred_cov.shape)
    auc_cov_PR = roc_auc_score(y_true, y_pred_cov)
    print("AUC cov PR:", auc_cov_PR)
    auc_emb_PR = roc_auc_score(y_true, y_pred_emb)
    print("AUC emb PR:", auc_emb_PR)
    auc_both_PR = roc_auc_score(y_true, y_pred_both)
    print("AUC both PR:", auc_both_PR)

    auc_cov_MLP = roc_auc_score(y_true, y_pred_cov_MLP)
    print("AUC cov MLP:", auc_cov_MLP)
    auc_emb_MLP = roc_auc_score(y_true, y_pred_emb_MLP)
    print("AUC emb MLP:", auc_emb_MLP)
    auc_both_MLP = roc_auc_score(y_true, y_pred_both_MLP)
    print("AUC both MLP:", auc_both_MLP)
    return {
    "auc_cov_PR": auc_cov_PR,
    "auc_emb_PR": auc_emb_PR,
    "auc_both_PR": auc_both_PR,
    "auc_cov_MLP": auc_cov_MLP,
    "auc_emb_MLP": auc_emb_MLP,
    "auc_both_MLP": auc_both_MLP
        }


def train_and_eval(
                pos_encoder,
                do_pca=False,
                n_pca_components=None,
                hidden_size=[256, 256],
                epochs=200,
                train_MLP=True,
                data_callback=get_data_geoplant
                ):
    PR_emb, PR_cov, PR_both, MLP_emb, MLP_cov, MLP_both, scaler_cov, scaler_emb, pca = train_models(
                pos_encoder=pos_encoder,
                do_pca=do_pca,
                n_pca_components=n_pca_components,
                hidden_size=hidden_size,
                epochs=epochs,
                train_MLP=train_MLP,
                data_callback=data_callback
                )
            #eval
    results = evaluate_models(
                pos_encoder, 
                scaler_cov, 
                scaler_emb, 
                PR_cov, #from fit_multi_GLM
                PR_emb, #from fit_multi_GLM
                PR_both, #from fit_multi_GLM
                MLP_cov, #from train_one_MLP : shape (n_samples, n_species)
                MLP_emb,  #from train_one_MLP
                MLP_both,
                pca_model=pca, #to lower dim of X_test_emb.
                train_MLP=train_MLP,
                data_callback=data_callback
                ) 
    return results



def sklearn_classification(X_cov=None, y=None, X_test_cov=None, y_true=None, data_callback=None):
    '''
    10-fold CV for the k parameter, then outputs the AUC for k-NN classification on test set.
    Handles multi-output binary classification (e.g., 62 responses at once).
    '''
    if data_callback is not None:
        X_cov, y, coords_po, X_test_cov, y_true, coords_pa = data_callback()  


    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    param_grid = {"knn__estimator__n_neighbors": range(1, 20)}
    
    CV_object = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=10, 
        scoring='roc_auc_ovr',  # will work for multi-output
        n_jobs=-1
    )
    
    CV_object.fit(X_cov, y)
    best_k = CV_object.best_params_["knn__estimator__n_neighbors"]
    
    y_pred_proba = np.array([estimator.predict_proba(X_test_cov)[:, 1] 
                                for estimator in CV_object.best_estimator_.named_steps['knn'].estimators_]).T
    
    aucs = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        aucs.append(auc)
    mean_auc = np.mean(aucs)
    print(f"Mean AUC across {y_true.shape[1]} responses: {mean_auc:.4f}, best k={best_k}")
    return mean_auc, best_k


    

if __name__ == "__main__":
#Load model
    device="cuda"
    data_path="embeddings_data_and_dictionaries/bioCLIP_full_dataset_embeddings.h5"
    dim_fourier_encoding=64 #multiple of 4!!
    dim_hidden=256
    dim_emb=128 #this one is actually shared with img embeddings
    max_freq=dim_fourier_encoding // (2 * 2)
    scales = torch.arange(4, max_freq+4, dtype=torch.float32).to("cuda")
    print("scales:",scales)
    pos_encoder= nn_classes.Fourier_MLP(original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=dim_emb)
    #pos_encoder=utils.RFF_MLPs( original_dim=2, fourier_dim=dim_fourier_encoding, hidden_dim=dim_hidden, output_dim=512,M=8,sigma_min=1,sigma_max=256).to(device)
    model= nn_classes.DoubleNetwork_V2(pos_encoder,dim_hidden=768,dim_output=dim_emb).to(device)
    model.load_state_dict(torch.load("Models/high_frequency_encoding/model.pt", weights_only=True))
    pos_encoder=model.pos_encoder
    
#Train
    PR_emb, PR_cov, MLP_emb, MLP_cov, scaler_cov, scaler_emb, pca = train_models(
        pos_encoder=pos_encoder,
        do_pca=False,
        n_pca_components=None,
    )
#Eval
    evaluate_models(
        pos_encoder, 
        scaler_cov, 
        scaler_emb, 
        PR_cov, #from fit_multi_GLM
        PR_emb, #from fit_multi_GLM
        MLP_cov, #from train_one_MLP : shape (n_samples, n_species)
        MLP_emb,  #from train_one_MLP
        pca_model=None, #to lower dim of X_test_emb.
        ) 

def apply_model_with_config(base_path, params_to_read, callback, output_file="."):
    """
    base_path (str): Path to the folder containing config.yaml and model.pt
    params_to_read (list): List of parameter keys to extract from config.yaml, e.g., ['training.batch_size']
    callback (function): Function that takes a PyTorch model and returns something
    output_file (str): Path to save the result of callback
    """
    config_path = os.path.join(base_path, 'config.yaml')
    model_path = os.path.join(base_path, 'model.pt')

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Function to get nested parameters
    def get_nested(config_dict, key_path):
        keys = key_path.split('.')
        val = config_dict
        for k in keys:
            val = val[k]
        return val

    extracted_params = {param: get_nested(config, param) for param in params_to_read}
    model = torch.load(model_path, map_location='cpu')

    result = callback(model)

    # Save
    output_data = {
    'extracted_params': extracted_params,
    'result': result
    }
    torch.save(output_data, output_file)
    print(f"Saved output to {output_file}")