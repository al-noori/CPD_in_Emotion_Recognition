import os
import pandas as pd
import shutil
from scipy.stats import zscore
import stat
parent_folder = r"C:\Users\A. Lowejatan Noori\Desktop\test\anonymized-23-10-2023"

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clean_and_filter_folders():
    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)

        if os.path.isdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, 'ground_truth.csv')
            # remove unnecessary files
            for filename in ['ACC.csv', 'IBI.csv', 'ST.csv']:
                file_to_remove = os.path.join(subfolder_path, filename)
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)

            df = pd.read_csv(csv_path)

            # consider only neutral scenarios with tiago++
            non_neutral = df['condition'].dropna().str.lower().ne('neutral')
            elenoide = df['robot'].dropna().str.lower() == 'elenoide'

            if non_neutral.any() or elenoide.any():
                shutil.rmtree(subfolder_path, onerror=handle_remove_readonly)



# Run everything
clean_and_filter_folders()
