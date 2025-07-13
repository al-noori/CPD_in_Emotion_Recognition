from best_psi import best_psi
from best_threshold import best_threshold
from point_score import point_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ruptures import Pelt

def pelt_cpd(Y, model="rbf", pen=10):
    """
    Change point detection using PELT
    """
    algo = Pelt(model=model).fit(Y)
    cp_indices = algo.predict(pen=pen)
    # Remove the last point as ruptures returns length as last cp
    return np.array(cp_indices[:-1])

def mcpd(Y, win_size=40, alpha=1):
    """Multiple Change Point Detection (MCPD) via iCID"""
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    return np.where(pscore > threshold)[0], pscore, threshold

def group_cp_regions(timestamps, indices, min_gap=2.0):
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
base_path = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023\b1d5f67d3ee6e58b85238a74e11cbb7a2b1881b831731ae2eb2ed1792e121638"
gsr_pf = pd.read_csv(os.path.join(base_path, "GSR.csv"))
bvp_pf = pd.read_csv(os.path.join(base_path, "BVP.csv"))

# Convert timestamps from ms to seconds
X = gsr_pf.loc[gsr_pf['shortNTPTime'].notna(), 'shortNTPTime'].values / 1000
X2 = bvp_pf.loc[bvp_pf['shortNTPTime'].notna(), 'shortNTPTime'].values / 1000
Y = gsr_pf['GSR_clean'].values[gsr_pf['shortNTPTime'].notna()]
Y2 = bvp_pf['BVP_clean'].values[bvp_pf['shortNTPTime'].notna()]

cp_indices, pscore, threshold = mcpd(Y.reshape(-1,1), win_size=10, alpha=1)
cp_indices2, pscore2, threshold2 = mcpd(Y2.reshape(-1,1), win_size=300, alpha=1)

#--- BVP only-----
# Plot BVP signal
'''
fig, ax = plt.subplots(2,1,figsize=(12, 5))
ax[0].plot(X2, Y2, color='orange', label='BVP Signal')

# Highlight change point regions
for start, end in group_cp_regions(X2, cp_indices2, min_gap=5.0):
    ax[0].axvspan(start, end, color='red', alpha=0.2)

# Overlay emotion annotations
for _, row in bvp_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[0].axvline(x=row['shortNTPTime'] / 1000, color='red', linestyle='--')

ax[0].set_ylabel('BVP Signal')
ax[0].set_xlabel('Time (seconds)')

ax[1].plot(X2, pscore2, color='blue', label='Score')
ax[1].axhline(y=threshold2, color='black', linewidth=0.8)
ax[1].set_ylabel('Score')
ax[1].set_xlabel('Time (seconds)')
'''
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 9))

# GSR plot
ax[0].plot(X, Y, color='blue')
for start, end in group_cp_regions(X, cp_indices, min_gap=2.0):
    ax[0].axvspan(start, end, color='red', alpha=0.2)
for _, row in gsr_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[0].axvline(x=row['shortNTPTime'] / 1000, color='red', linestyle='--')
ax[0].set_ylabel('GSR Signal', rotation=90, labelpad=0)
ax[0].yaxis.set_label_coords(-0.04, 0.5)
ax[0].set_xlabel('Time (seconds)')
ax[0].xaxis.set_label_coords(0.5, -0.04)

pelt_indices = pelt_cpd(Y.reshape(-1, 1), model="rbf", pen=10)
# Highlight PELT change points
for idx in pelt_indices:
    ax[0].axvline(x=X[idx], color='green', linestyle='--', linewidth=0.8, label='PELT CP' if idx == pelt_indices[0] else "")



# GSR point score
ax[1].plot(pscore, color='green')
ax[1].axhline(y=threshold, color='black', linewidth=0.8)

# Position threshold text on right side
xlim = ax[1].get_xlim()
ylim = ax[1].get_ylim()
ax[1].text(0.95*xlim[1], threshold, 'Threshold', color='black', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right')

ax[1].set_ylabel('Score', rotation=90, labelpad=0)
ax[1].yaxis.set_label_coords(-0.04, 0.5)

# BVP plot
ax[2].plot(X2, Y2, color='orange')
for start, end in group_cp_regions(X2, cp_indices2, min_gap=5.0):
    ax[2].axvspan(start, end, color='red', alpha=0.2)
for _, row in bvp_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[2].axvline(x=row['shortNTPTime'] / 1000, color='red', linestyle='--')
ax[2].set_ylabel('BVP Signal', rotation=90, labelpad=0)
ax[2].yaxis.set_label_coords(-0.04, 0.5)
ax[2].set_xlabel('Time (seconds)')
ax[2].xaxis.set_label_coords(0.5, -0.04)

# BVP point score
ax[3].plot(pscore2, color='purple')
ax[3].axhline(y=threshold2, color='black', linewidth=0.8)

# Position threshold text on right side
xlim2 = ax[3].get_xlim()
ylim2 = ax[3].get_ylim()
ax[3].text(0.95*xlim2[1], threshold2, 'Threshold', color='black', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right')

ax[3].set_ylabel('Score', rotation=90, labelpad=0)
ax[3].yaxis.set_label_coords(-0.04, 0.5)

# X-axis label centered below bottom subplot
ax[3].set_xlabel('Time (seconds)')
ax[3].xaxis.set_label_coords(0.5, -0.7)
# Remove legends if any


for a in ax:
    if a.legend_:
        a.legend_.remove()
plt.tight_layout()
plt.show()
