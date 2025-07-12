from best_psi import best_psi
from best_threshold import best_threshold
from point_score import point_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def mcpd(Y, win_size = 40, alpha = 1):
    """
    Multiple Change Point Detection (MCPD) via iCID
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    return np.where(pscore > threshold)[0], pscore, threshold


# Example usage of the mcpd function

base_path = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023\2462788a541f3772221bd44769cfe111823ca51e135d4e7b60874a73325984eb"
gsr_pf = pd.read_csv(os.path.join(base_path, "GSR.csv"))
bvp_pf = pd.read_csv(os.path.join(base_path, "BVP.csv"))

X = gsr_pf.loc[gsr_pf['shortNTPTime'].notna(), 'shortNTPTime'].values / 1000
X2 = bvp_pf.loc[bvp_pf['shortNTPTime'].notna(), 'shortNTPTime'].values / 1000

Y = gsr_pf['GSR_clean'].values[gsr_pf['shortNTPTime'].notna()]
cp_indices, pscore, threshold = mcpd(Y.reshape(-1,1), win_size=60, alpha=1)

Y2 = bvp_pf['BVP_clean'].values[bvp_pf['shortNTPTime'].notna()]
cp_indices2, pscore2, threshold2 = mcpd(Y2.reshape(-1,1),win_size=300, alpha=2)
# Plotting the results


fig, ax = plt.subplots(nrows = 4, ncols= 1, figsize=(12, 6))
ax[0].plot(X, Y, label='GSR', color='blue')
for cp in cp_indices:
    ax[0].axvline(x=X[cp], color='red', linestyle='--', alpha=0.1)
ax[0].legend()

ax[2].plot(X2, Y2, label='BVP', color='orange')
for cp in cp_indices2:
    ax[2].axvline(x=X2[cp], color='red', linestyle='--', alpha=0.1)
ax[2].legend()

for _, row in gsr_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[0].axvline(x=(row['shortNTPTime'] / 1000), color='purple', linestyle='-')
for _, row in bvp_pf.dropna(subset=['emotion_HRI']).iterrows():
    ax[2].axvline(x=(row['shortNTPTime'] / 1000), color='purple', linestyle='-')



ax[1].plot(pscore, label='GSR Point Score', color='green')
ax[1].axhline(y=threshold, color='red', linestyle='--', label='Threshold')
ax[1].legend()

ax[3].plot(pscore2, label='BVP Point Score', color='purple')
ax[3].axhline(y=threshold2, color='red', linestyle='--', label='Threshold')
ax[3].legend()


print(cp_indices)
plt.legend()
plt.tight_layout()
plt.show()

