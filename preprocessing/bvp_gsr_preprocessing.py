import pandas as pd
import os
import numpy as np
from scipy.stats import zscore
import neurokit2 as nk2


def preprocess_gsr(gsr_d, fs=4):
    signals, info = nk2.eda_process(gsr_d['GSR'], sampling_rate=fs)
    gsr_d['GSR_tonic'] = zscore(pd.Series(signals['EDA_Tonic']))
    gsr_d['GSR_phasic'] = zscore(pd.Series(signals['EDA_Phasic']))
    gsr_d['GSR_clean'] = zscore(pd.Series(signals['EDA_Clean']))
    gsr_d['GSR_avg'] = gsr_d['GSR_clean'].rolling(window=fs, min_periods=1 ).mean()  # Smoothing
    gsr_d['GSR_std'] = gsr_d['GSR_clean'].rolling(window=fs, min_periods=1 ).std()

def preprocess_bvp(bvp_d, fs=64):
    signals, info = nk2.ppg_process(bvp_d['BVP'], sampling_rate=fs)
    bvp_d['BVP_rate'] = zscore(pd.Series(signals['PPG_Rate']))
    bvp_d['BVP_clean'] = zscore(pd.Series(signals['PPG_Clean']))
    bvp_d['BVP_clean'] = np.clip(bvp_d['BVP_clean'].values, np.percentile(bvp_d['BVP_clean'].values, 1), np.percentile(bvp_d['BVP_clean'].values, 99))
    bvp_d['BVP_avg'] = bvp_d['BVP_clean'].rolling(window=fs , min_periods=1).mean()  # Smoothing
    bvp_d['BVP_std'] = bvp_d['BVP_clean'].rolling(window=fs, min_periods=1).std()
def process_folder(folder):

    bvp_path = os.path.join(folder, "BVP.csv")
    gsr_path = os.path.join(folder, "GSR.csv")

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)

    bvp = bvp[['NTPTime', 'BVP']]
    # Process signals
    gsr = gsr[['NTPTime', 'GSR']]
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


