import pandas as pd

import neurokit2 as nk

import os

parent_folder = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023"
for subfolder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        gsr_path = os.path.join(subfolder_path, "GSR.csv")
        bvp_path = os.path.join(subfolder_path, "BVP.csv")
        gsr = pd.read_csv(gsr_path)
        bvp = pd.read_csv(bvp_path)
        print(pd.isnull(gsr['GSR_clean']).sum())
        print(pd.isnull(bvp['BVP_clean']).sum())
        # varianz stdabweichung
