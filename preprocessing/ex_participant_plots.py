import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
base_path = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\data\88acdccfe1ab13225e2cb86a3fe13ba4c63d4ce9f3a38f219381c124a2ff6edc"
gsr = pd.read_csv(os.path.join(base_path, "GSR.csv"))
bvp = pd.read_csv(os.path.join(base_path, "BVP.csv"))

# Settings
start_time_gsr = gsr.loc[gsr['TAG'] == 'HRI_start', 'NTPTime'].iloc[0]
start_time_bvp = bvp.loc[bvp['TAG'] == 'HRI_start', 'NTPTime'].iloc[0]
end_t = bvp.loc[bvp['TAG'] == 'second_baseline', 'NTPTime'].iloc[1]

# Plot setup
fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
signals = [('GSR', gsr, 'GSR_clean'), ('BVP', bvp, 'BVP_clean')]
axs[0].axvline(x=start_time_gsr, color='orange', linestyle='--')
axs[0].text(start_time_gsr, axs[0].get_ylim()[1] * 1.1, 'HRI Start', rotation=90, va='bottom', color='orange', fontsize=8)

axs[1].axvline(x=start_time_bvp, color='orange', linestyle='--')
axs[1].text(start_time_bvp, axs[1].get_ylim()[1] * 1.1, 'HRI Start', rotation=90, va='bottom', color='orange', fontsize=8)

# Full and downsampled plots
for i, (label, df, col) in enumerate(signals):
    axs[i].plot(df['shortNTPTime'], df[col])
    axs[i].set_title(label)
    axs[i].grid(True)
# Event annotations on full-resolution plotsfor i in range(2):

for _, row in gsr.dropna(subset=['emotion_HRI']).iterrows():
    axs[0].axvline(x=row['shortNTPTime'], color='red', linestyle='--')
    axs[0].text(row['shortNTPTime'], axs[0].get_ylim()[1]*1.1, row['emotion_HRI'], rotation=90, va='bottom', color='red', fontsize=8)

for _, row in gsr.dropna(subset=['emotion_HRI']).iterrows():
    axs[1].axvline(x=row['shortNTPTime'], color='red', linestyle='--')
    axs[1].text(row['shortNTPTime'], axs[1].get_ylim()[1] * 1.1, row['emotion_HRI'], rotation=90, va='bottom', color='red',
                fontsize=8)

plt.xlabel('shortNTPTime')
plt.tight_layout()
plt.show()
