import pandas as pd
import os
import numpy as np
from scipy.stats import zscore
from neurokit2 import ppg_clean, eda_clean
from biosppy import bvp

def preprocess_gsr(gsr_d, fs=4):
    eda_cleaning = eda_clean(gsr_d['GSR'].values, sampling_rate=fs)
    gsr_d['GSR_clean'] = pd.Series(eda_cleaning)
    gsr_d['GSR_clean'] = gsr_d['GSR_clean'].rolling(window=fs, min_periods=1 ).mean()  # Smoothing
    gsr_d['GSR_clean'] = zscore(gsr_d['GSR_clean'])
def preprocess_bvp(bvp_d, fs=64):
    bvp_cleaning = ppg_clean(bvp_d['BVP'].values, sampling_rate=fs)
    bvp_d['BVP_clean'] = pd.Series(bvp_cleaning)
    bvp_d['BVP_clean'] = np.clip(bvp_d['BVP_clean'].values, np.percentile(bvp_d['BVP_clean'].values, 1), np.percentile(bvp_d['BVP_clean'].values, 99))
    bvp_d['BVP_clean'] = bvp_d['BVP_clean'].rolling(window=fs , min_periods=1).mean()  # Smoothing
    bvp_d['BVP_clean'] = zscore(bvp_d['BVP_clean'])
def process_folder(folder):

    bvp_path = os.path.join(folder, "BVP.csv")
    gsr_path = os.path.join(folder, "GSR.csv")

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)


    # Process signals
    preprocess_gsr(gsr)
    preprocess_bvp(bvp)

    # Overwrite the original BVP.csv with cleaned data
    bvp.to_csv(bvp_path, index=False)
    gsr.to_csv(gsr_path, index=False)

parent_folder = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023"

for subfolder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        process_folder(subfolder_path)

