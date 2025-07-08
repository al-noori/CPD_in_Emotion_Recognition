from best_psi import best_psi
from best_threshold import best_threshold
from point_score import point_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def mcpd(Y, win_size = 50, alpha = 1):
    """
    Multiple Change Point Detection (MCPD) via iCID
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    print(pscore)
    return np.where(pscore > threshold)[0], pscore, threshold


# Example usage of the mcpd function

base_path = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023\b1d5f67d3ee6e58b85238a74e11cbb7a2b1881b831731ae2eb2ed1792e121638"
gsr_pf = pd.read_csv(os.path.join(base_path, "GSR.csv"))
bvp_pf = pd.read_csv(os.path.join(base_path, "BVP.csv"))
Y = gsr_pf['GSR_clean'].values[gsr_pf['shortNTPTime'].notna()]
Y= Y[::1]# Ensure we only use valid GSR data
cp_indices, pscore, threshold = mcpd(Y.reshape(-1,1))

Y2 = bvp_pf['BVP_clean'].values[bvp_pf['shortNTPTime'].notna()]
Y2 = Y2[::1]# Ensure we only use valid BVP data
cp_indices2, pscore2, threshold2 = mcpd(Y2.reshape(-1,1))
# Plotting the results


fig, ax = plt.subplots(nrows = 4, ncols= 1, figsize=(12, 6))
ax[0].plot(Y, label='GSR', color='blue')
for cp in cp_indices:
    ax[0].axvline(x=cp, color='red', linestyle='--', alpha=0.5)
ax[0].legend()

ax[2].plot(Y2, label='BVP', color='orange')
for cp in cp_indices2:
    ax[2].axvline(x=cp, color='red', linestyle='--', alpha=0.5)
ax[2].legend()

ax[1].plot(pscore, label='GSR Point Score', color='green')
ax[1].axhline(y=threshold, color='red', linestyle='--', label='Threshold')
ax[1].legend()

ax[3].plot(pscore2, label='BVP Point Score', color='purple')
ax[3].axhline(y=threshold2, color='red', linestyle='--', label='Threshold')
ax[3].legend()


print(cp_indices)
plt.legend()
plt.show()

