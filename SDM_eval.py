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


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256, 256], out_dim=30):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))  # linear output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x=self.net(x)
        return x
    
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


def get_embeddings(df, pos_encoder, device="cuda"):
    '''Takes a df with columns "x" and "y" in Swiss coordinates and returns the positional embeddings from the pos_encoder'''

    shift_x, shift_y= (1011627.4909483634, -100326.1477937577) #See "coordinates.ipynb"
    lons, lats = utils.coord_trans(df["x"].values-shift_x, df["y"].values-shift_y,order="CH_to_normal")
    coords = torch.tensor(
        np.column_stack([lats, lons]),
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        emb = pos_encoder(coords).cpu().numpy()

    return emb

def fourier_enc(x, scales=None,fourier_dim=64):
    pass
#Class Fourier_enc from nn_models
def train_models(
        pos_encoder,
        do_pca=True,
        n_pca_components=None,
        hidden_size=[256, 256],
        covariates = ["bcc","calc","ccc","ddeg","nutri","pday","precyy","sfroyy","slope","sradyy","swb","tavecc","topo"],
        po_data_path="embeddings_data_and_dictionaries/data_SDM_NCEAS/SWItrain_po.csv",
        epochs=200

):
    

    po_data=pd.read_csv(po_data_path)

    y = pd.get_dummies(po_data["spid"])
    po_data = pd.concat([po_data.drop(columns=["spid"]), y], axis=1) #For the full one-hot-encoded dataframe

    po_covariates=po_data.loc[:, covariates]

    X_cov=po_covariates.to_numpy()
    y=y.to_numpy()

    print("getting embeddings")

    X_emb=get_embeddings(po_data, pos_encoder)
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

    MLP_emb=MLP(in_dim=X_emb.shape[1], hidden=hidden_size, out_dim=30)
    MLP_emb = train_one_MLP(MLP_emb, X_emb, y,epochs=epochs)

    MLP_cov=MLP(in_dim=len(covariates), hidden=hidden_size, out_dim=30)
    MLP_cov = train_one_MLP(MLP_cov, X_cov, y,epochs=epochs)

    MLP_both=MLP(in_dim=X_emb.shape[1]+len(covariates), hidden=hidden_size, out_dim=30)
    MLP_both = train_one_MLP(MLP_both, np.concatenate([X_emb,X_cov],axis=1), y,epochs=epochs)

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
    covariates = ["bcc","calc","ccc","ddeg","nutri","pday","precyy","sfroyy","slope","sradyy","swb","tavecc","topo"],
    pa_csv_path="embeddings_data_and_dictionaries/data_SDM_NCEAS/SWItest_pa.csv",
    env_csv_path="embeddings_data_and_dictionaries/data_SDM_NCEAS/SWItest_env.csv",
    species_columns=[
    'swi01','swi02','swi03','swi04','swi05','swi06','swi07','swi08','swi09','swi10',
    'swi11','swi12','swi13','swi14','swi15','swi16','swi17','swi18','swi19','swi20',
    'swi21','swi22','swi23','swi24','swi25','swi26','swi27','swi28','swi29','swi30']
):
    """
    Evaluation on PA data
    """
    pa_data=pd.read_csv(pa_csv_path)
    env=pd.read_csv(env_csv_path)


    y_true = pa_data.loc[:, species_columns]
    X_test_cov = env.loc[:,covariates]
    X_test_cov=X_test_cov.to_numpy()
    y_true=y_true.to_numpy()

    X_test_emb=get_embeddings(env,pos_encoder) ##############################

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
    "auc_cov_MLP": auc_cov_MLP,
    "auc_emb_MLP": auc_emb_MLP
        }








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

