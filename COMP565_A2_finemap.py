#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:37:47 2023

@author: Zilong Wang 260823366
"""
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import os
import math
from itertools import combinations
import matplotlib as plt

BASE_FOLDER = '/Users/Mobrain1/Desktop/COMP 565/A2'
M=100
N=498
ld = pd.read_csv(os.path.join(BASE_FOLDER,'LD.csv.gz'))
ld.set_index("Unnamed: 0", inplace = True)
zscore = pd.read_csv(os.path.join(BASE_FOLDER,'zscore.csv.gz'))
zscore.set_index("Unnamed: 0", inplace = True)

#%% Q1
# # 

# ld_CC_123 = ld[['rs10104559', 'rs1365732', 'rs12676370']].loc[['rs10104559', 'rs1365732', 'rs12676370']]# [1 1 1]
# zscore_C_123 = zscore.loc[['rs10104559', 'rs1365732', 'rs12676370']]

# # 2 causal
# ld_CC_12 = ld[['rs10104559', 'rs1365732']].loc[['rs10104559', 'rs1365732']] # [1 1 0]
# zscore_C_12 = zscore.loc[['rs10104559', 'rs1365732']]

# ld_CC_13 = ld[['rs10104559', 'rs12676370']].loc[['rs10104559', 'rs12676370']] # [1 0 1]
# zscore_C_13 = zscore.loc[['rs10104559', 'rs12676370']]

# ld_CC_23 = ld[['rs1365732', 'rs12676370']].loc[['rs1365732', 'rs12676370']] # [0 1 1]
# zscore_C_23 = zscore.loc[['rs1365732', 'rs12676370']]

# # 1 causal
# ld_CC_1 = ld[['rs10104559']].loc[['rs10104559']] # [1 0 0]
# zscore_C_1 = zscore.loc[['rs10104559']]

# ld_CC_2 = ld[['rs1365732']].loc[['rs1365732']] # [0 1 0]
# zscore_C_2 = zscore.loc[['rs1365732']]

# ld_CC_3 = ld[['rs12676370']].loc[['rs12676370']] # [0 0 1]
# zscore_C_3 = zscore.loc[['rs12676370']]

def efficient_BF (ld_CC, z_C, k):
    I_k = np.identity(k)
    ld_CC_star = np.add(ld_CC, np.matmul(ld_CC, np.matmul(2.49 * I_k, ld_CC))) # where 2.49 is Ns^2
    try: 
        numerator = multivariate_normal.pdf(np.array(z_C).T, mean=None, cov = ld_CC_star)
        denominator = multivariate_normal.pdf(np.array(z_C).T, mean=None, cov = ld_CC)
    
        efficient_BF = numerator / denominator
    
        return efficient_BF
    except: # disgard those configurations that result in non-finite multivariate density
        return None
    
# BF_123 = efficient_BF(ld_CC_123, zscore_C_123, 3)
# BF_12 = efficient_BF(ld_CC_12, zscore_C_12, 2)
# BF_13 = efficient_BF(ld_CC_13, zscore_C_13, 2)
# BF_23 = efficient_BF(ld_CC_23, zscore_C_23, 2)
# BF_1 = efficient_BF(ld_CC_1, zscore_C_1, 1)
# BF_2 = efficient_BF(ld_CC_2, zscore_C_2, 1)
# BF_3 = efficient_BF(ld_CC_3, zscore_C_3, 1)

#%% Q2
def prior (k):
    return math.pow(1 / M, k) * math.pow((M-1) / M, M-k)

#%% Q3
# def all_config_posterior ():
import tqdm
import warnings
warnings.filterwarnings("ignore")

all_SNPs = list(zscore.index) # 100 SNPs
sum_all_config_scores = 0 # the denominator of posterior
each_config_score = [] # The list of all possible configurations' scores, where score = BF_config * prior_k, size (156523,)
each_config = []  # The list of binary represented configurations, where 1 stands for the SNP being causal and 0 for non-causal, the equivalent array's size is (156523, 100)

for k in range (1, 4): # 1, 2 or 3 causal SNPs
    configs = list(combinations(all_SNPs, k)) # all the possible configurations given 1, 2 or 3 causal SNPs
    prior_k = prior(k) # the prior when there are k causal SNPs
    for config in tqdm.tqdm(configs): # for each possible causal configuration
        ld_CC = ld[list(config)].loc[list(config)] # get R_CC
        zscore_C = zscore.loc[list(config)] # get z^hat_CC
        BF = efficient_BF(ld_CC, zscore_C, k) # get this configuration's BF
        if BF != None: # only take into account the valid configurations
            each_config.append([1 if snp in config else 0 for snp in all_SNPs]) # this configuration's binary representation
            score = BF * prior_k 
            each_config_score.append(score)
            sum_all_config_scores += score

all_config_posterior = np.divide(each_config_score, sum_all_config_scores) # obtaining the posterior for each valid configuration
plt.pyplot.scatter(x=range(all_config_posterior.shape[0]), y=np.sort(all_config_posterior))  # plotting sorted posteriors

#%% Q4
each_config = np.array(each_config) # shape (156523, 100)
each_config_score = np.array(each_config_score) # shape (156523,)
pip_all_SNPs = [] # list of PIP for each SNP
for j in range(100): # for each SNP_j
    inds_SNP_j_causal = np.where(each_config[:, j]==1) # find the indices of configurations where SNP_j is causal
    sum_all_configs_scores_SNP_j_causal = each_config_score[inds_SNP_j_causal].sum() # numerator of PIP for SNP_j, sum of all the configurations'score where SNP_j is causal
    pip_SNP_j = sum_all_configs_scores_SNP_j_causal / sum_all_config_scores # PIP of SNP_j
    pip_all_SNPs.append(pip_SNP_j)

plt.pyplot.scatter(x=range(100), y=pip_all_SNPs) # visualize my inferred PIP, highly similar to the given PIP.
df_pip_all_SNPs = pd.DataFrame(pip_all_SNPs, index=all_SNPs) # store my inferred PIP into pandas dataframe
df_pip_all_SNPs.to_csv(os.path.join(BASE_FOLDER,'COMP565 A2 SNP pip.csv.gz'), compression='infer')
#%% validation
SNP_pip = pd.read_csv(os.path.join(BASE_FOLDER,'SNP_pip.csv.gz'))
plt.pyplot.scatter(x=range(100), y = SNP_pip['x'])
