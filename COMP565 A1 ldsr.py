#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 17:20:28 2023

@author: Mobrain1
"""
import pandas as pd
import numpy as np
import os

BASE_FOLDER = '/Users/Mobrain1/Desktop/COMP 565/A1/data'
M=4268
N=1000
beta_marg = pd.read_csv(os.path.join(BASE_FOLDER, 'beta_marginal.csv.gz'))
ld = pd.read_csv(os.path.join(BASE_FOLDER, 'LD.csv.gz'))

beta_marg_sqr = beta_marg['V1'].pow(2)
ldsc = ld.loc[:, ld.columns != 'Unnamed: 0'].pow(2).sum(axis=0)
ldsc_sqr = ldsc.pow(2)

# LD score regression based on OLS, adapted from answer to A1, Q1
numerator = 0
for j in range (M):
    numerator += ldsc.iloc[j] * (beta_marg_sqr.iloc[j] - 1 / N) # for each SNP, sum 

denominator = 0
for j in range (M):
    denominator += ldsc_sqr.iloc[j] / M
    
h_2 = numerator / denominator # let h_2 be heritability

# estimated heritability h_2 is equal to 0.19017122782624662

# h_2 = np.dot(np.dot(ldsc_sqr.values.T, ldsc_sqr.values),ldsc_sqr.values.T, N*beta_marg_sqr)
