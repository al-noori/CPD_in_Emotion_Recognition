from pathlib import Path
from best_psi import best_psi
from best_threshold import best_threshold
from point_score import point_score
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must come before importing pyplot


import matplotlib.pyplot as plt
import path
import pandas as pd
import os
from ruptures import Pelt
from scipy.signal import find_peaks

def mcpd(Y, win_size=40, alpha=1):
    """
    Multiple Change Point Detection (MCPD) via model
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    indices = find_peaks(pscore, height=threshold)[0] #Non-Maximum Suppression
    interval = np.where(pscore > threshold)[0]
    #np.where(pscore > threshold)[0]
    return indices, interval, pscore, threshold

def group_cp_regions(timestamps, indices, min_gap=1):
    """Group nearby change points into (start, end) time regions"""
    if len(indices) == 0:
        return []
    indices = sorted(indices)
    regions = []
    start_idx = indices[0]
    last_time = timestamps[start_idx]
    for idx in indices[1:]:
        current_time = timestamps[idx]
        if current_time - last_time > min_gap:
            regions.append((timestamps[start_idx], last_time))
            start_idx = idx
        last_time = current_time
    regions.append((timestamps[start_idx], last_time))
    return regions

# Load data
base_path = Path(path.DATA_PATH,'88acdccfe1ab13225e2cb86a3fe13ba4c63d4ce9f3a38f219381c124a2ff6edc')
print("Base path:", base_path)
print(path.DATA_PATH, path.PLOTS_PATH)
gsr_pf = pd.read_csv(os.path.join(base_path, "GSR.csv"))
bvp_pf = pd.read_csv(os.path.join(base_path, "BVP.csv"))

# Convert timestamps from ms to seconds
X = gsr_pf.loc[gsr_pf['shortNTPTime'].notna(), 'shortNTPTime'].values
X2 = bvp_pf.loc[bvp_pf['shortNTPTime'].notna(), 'shortNTPTime'].values
BEGIN = X[0]  # Start time for normalization
X = (X - X[0]) / 1000  # Normalize to start from 0
X2 = (X2 - X2[0]) / 1000 # Normalize to start from 0

Y = gsr_pf[['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']].values[gsr_pf['shortNTPTime'].notna()]
Y2 = bvp_pf[['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']].values[bvp_pf['shortNTPTime'].notna()]
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))

gsr_valid = gsr_pf.dropna(subset=['shortNTPTime'])
bvp_valid = bvp_pf.dropna(subset=['shortNTPTime'])
ax[0].plot(X, gsr_valid['GSR_clean'].values, color='blue')
ax[2].plot(X2, bvp_valid['BVP_clean'].values, color='blue')
print("Y shape:", Y.shape)
print("Y2 shape:", Y2.shape)
print("Y reshape " , Y.reshape(-1, 1).shape)
print("Y2 reshape " , Y2.reshape(-1, 1).shape)

cp_indices_max, cp_indices, pscore, threshold = mcpd(Y, win_size=20, alpha=2)
cp_indices_2_max, cp_indices_2, pscore2, threshold2 = mcpd(Y2, win_size=200, alpha=2)
'''
cp_indices_max = np.array([50, 150, 300])
cp_indices = np.array([45, 46, 47, 145, 146, 295, 296])
pscore = np.random.rand(len(Y))  # Fake point scores
threshold = 0.6

# Stub for BVP
cp_indices_2_max = np.array([100, 400, 600])
cp_indices_2 = np.array([95, 96, 97, 395, 396, 595, 596])
pscore2 = np.random.rand(len(Y2))  # Fake point scores
threshold2 = 0.5
print(X.shape, X2.shape)
'''
# GSR plot
for _, row in gsr_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[0].axvline(x=((row['shortNTPTime'] - BEGIN) / 1000), color='black', linewidth=1, linestyle='-')
for idx in cp_indices_max:
    ax[0].axvline(X[idx], color='red', linestyle='-', linewidth=1)


ax[0].set_ylabel('GSR')



# GSR point score
ax[1].plot(X, pscore, color='blue')
ax[1].axhline(y=threshold, color='black', linewidth=1)

# Position threshold text on right side
xlim = ax[1].get_xlim()
ax[1].text(xlim[1], threshold, 'Threshold', color='black', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right')

ax[1].set_ylabel('Score')
# BVP plot
for _, row in bvp_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[2].axvline(x=((row['shortNTPTime'] - BEGIN) / 1000) , color='black', linewidth=1, linestyle='-')

for idx in cp_indices_2_max:
    ax[2].axvline(X2[idx], color='red', linestyle='-', linewidth=1)
ax[2].set_ylabel('BVP')

# BVP point score
ax[3].plot(X2, pscore2, color='blue')
ax[3].axhline(y=threshold2, color='black', linewidth=1)

# Position threshold text on right side
xlim2 = ax[3].get_xlim()
ylim2 = ax[3].get_ylim()
ax[3].text(0.995*xlim2[1], threshold2, 'Threshold', color='black', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right')

ax[3].set_ylabel('Score')

# X-axis label centered below bottom subplot
ax[3].set_xlabel('Time (seconds)')
# Remove legends if any


ax[1].set_yticks([0, 0.5, 1])
ax[3].set_yticks([0, 0.5, 1])

plt.tight_layout()
plt.show()
plt.savefig( path.PLOTS_PATH / "mcpd_plot.png", dpi=300, bbox_inches='tight')