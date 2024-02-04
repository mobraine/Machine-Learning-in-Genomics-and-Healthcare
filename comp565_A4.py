# STUDENT NAME: Zilong Wang
# STUDENT ID: 260823366
import pickle
import numpy as np
import pandas as pd

import scanpy as sc
import anndata
import random

import torch
from etm import ETM
from torch import optim
from torch.nn import functional as F

import os
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from seaborn import heatmap, lineplot, clustermap

random.seed(10)

# mouse pancreas single-cell dataset
# read in data and cell type labels
with open('data/MP.pickle', 'rb') as f:
    df = pickle.load(f)

with open('data/MP_genes.pickle', 'rb') as f:
    genes = pickle.load(f)

df.set_index('Unnamed: 0', inplace=True)  # set first column (cell ID as the index column)
sample_id = pickle.load(open('data/cell_IDs.pkl', 'rb'))
df = df.loc[list(sample_id), :]

X = df[genes].values  # extract the N x M cells-by-genes matrix

sample_info = pd.read_csv('data/sample_info.csv')

mp_anndata = anndata.AnnData(X=X)

mp_anndata.obs['Celltype'] = sample_info['assigned_cluster'].values

N = X.shape[0]  # number of single-cell samples
K = 16  # number of topics
M = X.shape[1]  # number of genes


def evaluate_ari(cell_embed, adata):
    """
        This function is used to evaluate ARI using the lower-dimensional embedding
        cell_embed of the single-cell data
        :param cell_embed: a NxK single-cell embedding generated from NMF or scETM
        :param adata: single-cell AnnData data object (default to to mp_anndata)
        :return: ARI score of the clustering results produced by Louvain
    """
    adata.obsm['cell_embed'] = cell_embed
    sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs['louvain'])
    return ari


######## Q1 NMF sum of squared error ########
W_init = np.random.random((M, K))
H_init = np.random.random((K, N))


# Complete this function
def nmf_sse(X, W, H, adata=mp_anndata, niter=100):
    """
        NMF with sum of squared error loss as the objective
        :param X: M x N input matrix
        :param W: M x K basis matrix
        :param H: K x N coefficient matrix
        :param adata: annotated X matrix with cluster labels for evaluating ARI (default to mouse pancreas data)
        :param niter: number of iterations to run
        :return:
            1. updated W and H that minimize sum of squared error ||X - WH||^2_F s.t. W,H>=0
            2. niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
    """
    perf = np.ndarray(shape=(niter, 3), dtype='float')

    # WRITE YOUR CODE HERE
    import tqdm
    for i in tqdm.tqdm(range(niter)):
        H = np.multiply(H, np.divide(np.matmul(W.T, X), np.matmul(np.matmul(W.T, W), H)))
        W = np.multiply(W, np.divide(np.matmul(X, H.T), np.matmul(np.matmul(W, H), H.T)))
        mean_sse = np.trace(np.matmul(X.T, X) - np.matmul(X.T, np.matmul(W,H)) - np.matmul(H.T, np.matmul(W.T,X)) + np.matmul(np.matmul(H.T, W.T), np.matmul(W,H))) / (N*M)
        ari = evaluate_ari(H.T, adata)
        perf[i][0] = i
        perf[i][1] = mean_sse
        perf[i][2] = ari
    return W, H, perf


W_nmf_sse, H_nmf_sse, nmf_sse_perf = nmf_sse(X.T, W_init, H_init, niter=100)


######## Q2: write a function to monitor ARI and objective function ########
def monitor_perf(perf, objective, path=""):
    """
    :param perf: niter-by-3 ndarray with iteration index, objective function, and ARI as the 3 columns
    :param objective: 'SSE', 'Poisson', or 'NELBO'
    :param path: path to save the figure if not display to the screen
    :behaviour: display or save a 2-by-1 plot showing the progress of optimizing objective and ARI as
        a function of iterations
    """

    # WRITE YOUR CODE HERE
    x_axis = perf[:,0]
    y_axis_objective = perf[:,1]
    y_axis_ari = perf[:,2]

    fig, axs = plt.subplots(2)
    axs[0].plot(x_axis,y_axis_objective)
    axs[0].set(xlabel='iter', ylabel=objective)
    axs[1].plot(x_axis, y_axis_ari)
    axs[1].set(xlabel='iter', ylabel='ARI')
    plt.savefig(path)


monitor_perf(nmf_sse_perf, "MSE", 'figures/nmf_sse.eps')


######## Q3 NMF Poisson likelihood ########
# NMF with Poisson likelihood
# Complete this function
def nmf_psn(X, W, H, adata=mp_anndata, niter=100):
    """
        NMF with log Poisson likelihood as the objective
        :param X.T: M x N input matrix
        :param W: M x K basis matrix
        :param H: K x N coefficient matrix
        :param niter: number of iterations to run
        :return:
            1. updated W and H that minimize sum of squared error ||X - WH||^2_F s.t. W,H>=0
            2. niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
    """
    perf = np.ndarray(shape=(niter, 3), dtype='float')

    # WRITE YOUR CODE HERE
    import tqdm
    for i in tqdm.tqdm(range(niter)):
        H = np.where(H > 0, H, 1e-16)
        W = np.where(W > 0, W, 1e-16)
        H = np.multiply(H, np.divide(np.matmul(W.T, np.divide(X, np.matmul(W,H))), np.matmul(W.T, np.ones((M,N)))))
        W = np.multiply(W, np.divide(np.matmul(np.divide(X, np.matmul(W,H)), H.T), np.matmul(np.ones((M,N)), H.T)))
        mean_sse = np.trace(np.multiply(X, np.log(np.matmul(W,H))) - np.matmul(W,H)) / (N*M)
        ari = evaluate_ari(H.T, adata)
        perf[i][0] = i
        perf[i][1] = mean_sse
        perf[i][2] = ari
    return W, H, perf


W_nmf_psn, H_nmf_psn, nmf_psn_perf = nmf_psn(X.T, W_init, H_init, niter=100)

monitor_perf(nmf_psn_perf, "Poisson", 'figures/nmf_psn.eps')

# compare NMF-SSE and NMF-Poisson
fig, ax = plt.subplots()
nmf_sse_perf_df = pd.DataFrame(data=nmf_sse_perf, columns=['Iter', "SSE", 'ARI'])
nmf_psn_perf_df = pd.DataFrame(data=nmf_psn_perf, columns=['Iter', "Poisson", 'ARI'])
ax.plot(nmf_sse_perf_df["Iter"], nmf_sse_perf_df["ARI"], color='blue', label='NMF-SSE')
ax.plot(nmf_psn_perf_df["Iter"], nmf_psn_perf_df["ARI"], color='red', label='NMF-PSN')
ax.legend()
plt.xlabel("Iteration");
plt.ylabel("ARI")
plt.savefig("figures/nmf_sse_vs_psn.eps")

######## Q4-Q8 VAE single-cell embedded topic model ########
X_tensor = torch.from_numpy(np.array(X, dtype="float32"))
sums = X_tensor.sum(1).unsqueeze(1)
X_tensor_normalized = X_tensor / sums

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ETM(num_topics=K,
            vocab_size=len(genes),
            t_hidden_size=256,
            rho_size=256,
            theta_act='relu',
            embeddings=None,
            train_embeddings=True,
            enc_drop=0.5).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.2e-6)


# train the VAE for one epoch
def train_scETM_helper(model, X_tensor, X_tensor_normalized):
    # initialize the model and loss
    model.train()
    optimizer.zero_grad()
    model.zero_grad()

    # forward and backward pass
    nll, kl_theta = model(X_tensor, X_tensor_normalized)
    loss = nll + kl_theta
    loss.backward()  # backprop gradients w.r.t. negative ELBO

    # clip gradients to 2.0 if it gets too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # update model to minimize negative ELBO
    optimizer.step()

    return torch.sum(loss).item()


# get sample encoding theta from the trained encoder network
def get_theta(model, input_x):
    model.eval()
    with torch.no_grad():
        q_theta = model.q_theta(input_x)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta


######## Q4 complete this function ########
def train_scETM(model, X_tensor, X_tensor_normalized, adata=mp_anndata, niter=1000):
    """
        :param model: the scETM model object
        :param X_tensor: NxM raw read count matrix X
        :param X_tensor_normalized: NxM normalized read count matrix X
        :param adata: annotated single-cell data object with ground-truth cell type information for evaluation
        :param niter: maximum number of epochs
        :return:
            1. model: trained scETM model object
            2. perf: niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
    """
    perf = np.ndarray(shape=(niter, 3), dtype='float')

    # WRITE YOUR CODE HERE
    import tqdm
    for i in tqdm.tqdm(range(niter)):
        nelbo = train_scETM_helper(model, X_tensor, X_tensor_normalized)
        theta = get_theta(model, X_tensor_normalized)
        ari = evaluate_ari(theta, adata)
        perf[i][0] = i
        perf[i][1] = nelbo
        perf[i][2] = ari

    return model, perf


model, scetm_perf = train_scETM(model, X_tensor, X_tensor_normalized)

monitor_perf(scetm_perf, "NELBO", 'figures/scETM_train.eps')

######## Q5 Compare NMF-Poisson and scETM ########

# WRITE YOUR CODE HERE
W_nmf_psn_1000, H_nmf_psn_1000, nmf_psn_perf_1000 = nmf_psn(X.T, W_init, H_init, niter=1000)
monitor_perf(nmf_psn_perf_1000, "Poisson_1000_iter", 'figures/nmf_psn_1000iter.eps')

fig, ax = plt.subplots()
scetm_perf_df = pd.DataFrame(data=scetm_perf, columns=['Iter', "NELBO", 'ARI'])
nmf_psn_perf_1000_df = pd.DataFrame(data=nmf_psn_perf_1000, columns=['Iter', "Poisson", 'ARI'])
ax.plot(scetm_perf_df["Iter"], scetm_perf_df["ARI"], color='blue', label='scETM')
ax.plot(nmf_psn_perf_1000_df["Iter"], nmf_psn_perf_1000_df["ARI"], color='red', label='NMF-PSN')
ax.legend()
plt.xlabel("Iteration");
plt.ylabel("ARI")
plt.savefig("figures/nmf_scETM_vs_psn1000.eps")

######## Q6 plot t-SNE for NMF-Poisson and scETM ########

# WRITE YOUR CODE HERE
theta = get_theta(model, X_tensor_normalized)
mp_anndata_theta = anndata.AnnData(X=np.array(theta))
mp_anndata_theta.obs['Celltype'] = sample_info['assigned_cluster'].values
scETM_tsne = sc.tl.tsne(mp_anndata_theta, n_pcs=2)
sc.pl.tsne(mp_anndata_theta, color = 'Celltype')#, color_map = 'magma', palette = 'Set2' )

mp_anndata_H_nmf_psn_1000 = anndata.AnnData(X=H_nmf_psn_1000.T)
mp_anndata_H_nmf_psn_1000.obs['Celltype'] = sample_info['assigned_cluster'].values
nmf_psn_1000_tsne = sc.tl.tsne(mp_anndata_H_nmf_psn_1000, n_pcs=2)
sc.pl.tsne(mp_anndata_H_nmf_psn_1000, color = 'Celltype')#, color_map = 'magma', palette = 'Set2')

######## Q7 plot cells by topics ########

# WRITE YOUR CODE HERE
from seaborn import color_palette
from matplotlib.patches import Patch

theta_df = pd.DataFrame(theta, index=sample_info['assigned_cluster'].values)
row_labels = np.unique(sample_info['assigned_cluster'])
label_colors = {label: color for label,color in zip(row_labels, color_palette("hls", len(row_labels)))}
row_colors = pd.DataFrame(row_labels, columns=['label'])['label'].map(label_colors)
clustermap(theta_df, cmap='Reds', row_colors=row_colors)
handles = [Patch(facecolor=label_colors[name]) for name in label_colors]
plt.legend(handles, label_colors, title='Species',
           bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
plt.savefig('figures/cells_heatmap_scetm.png')


######## Q8 plot genes-by-topics ########

# WRITE YOUR CODE HERE
beta = model.get_beta()
beta = torch.transpose(beta, 0, 1)
beta = beta.detach().numpy()
beta_df = pd.DataFrame(beta, index=genes)
plt.figure(figsize=(6,4))
heatmap(beta_df.head(5), vmax=0.2, square=False)
plt.savefig('figures/topics_heatmap_scetm.png')



