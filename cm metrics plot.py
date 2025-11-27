from physf import *
from dataf import *
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    "font.family": "serif",
    "font.weight": "bold",
    "font.size": 28,
    "font.serif": "Computer Modern",
    "legend.fontsize": 20
})

# GLOBAL

gb = np.load('global/global stim b.npy') # baseline predictive accuracy
gp = np.load('global/global stim p.npy') # model predictive accuracy
gbtp = np.load('global/global stim b_tp.npy') # baseline true positives
gptp = np.load('global/global stim p_tp.npy') # model true positives
gbtn = np.load('global/global stim b_tn.npy') # baseline true negatives
gptn = np.load('global/global stim p_tn.npy') # model true negatives

# LOCAL

lb = np.load('local/local stim b.npy') # baseline predictive accuracy
lp = np.load('local/local stim p.npy') # model predictive accuracy
lbtp = np.load('local/local stim b_tp.npy') # baseline true positives
lptp = np.load('local/local stim p_tp.npy') # model true positives
lbtn = np.load('local/local stim b_tn.npy') # baseline true negatives
lptn = np.load('local/local stim p_tn.npy') # model true negatives

gbm, gbci = mean_and_ci(gb, 90)
gpm, gpci = mean_and_ci(gp, 90)
gbtpm, gbtpci = mean_and_ci(gbtp, 90)
gptpm, gptpci = mean_and_ci(gptp, 90)
gbtnm, gbtnci = mean_and_ci(gbtn, 90)
gptnm, gptnci = mean_and_ci(gptn, 90)

lbm, lbci = mean_and_ci(lb, 90)
lpm, lpci = mean_and_ci(lp, 90)
lbtpm, lbtpci = mean_and_ci(lbtp, 90)
lptpm, lptpci = mean_and_ci(lptp, 90)
lbtnm, lbtnci = mean_and_ci(lbtn, 90)
lptnm, lptnci = mean_and_ci(lptn, 90)

ss = np.arange(3, 7)
ts = np.arange(-200, 300, 100) # in ms


f, axs = plt.subplots(2, 3, sharex=True, sharey='col', figsize=(24, 10))

cmap = {0: "#D81B60", 1: "#FFC107", 2: '#004D40', 3: "#1E88E5"} # colorblind-friendly palette
offsets = [-24, -8, 8, 24]

calls = list(cmap.keys())

for i in calls:
    axs[0, 0].scatter(ts+offsets[i], gpm[:, calls[i]], color=cmap[i], s=60, marker='o', label=f'{i+3} neurons') # 'o' for global stim, 'x' for local stim. specify in caption
    axs[0, 0].scatter(ts+offsets[i], gbm[:, calls[i]], color=cmap[i], alpha=0.3, s=60, marker='o')

    axs[1, 0].scatter(ts+offsets[i], lpm[:, calls[i]], color=cmap[i], s=60, marker='x', label=f'{i+3} neurons')
    axs[1, 0].scatter(ts+offsets[i], lbm[:, calls[i]], color=cmap[i], alpha=0.3, s=60, marker='x')

    axs[0, 0].errorbar(ts+offsets[i], gpm[:, calls[i]], yerr=gpci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8)
    axs[0, 0].errorbar(ts+offsets[i], gbm[:, calls[i]], yerr=gbci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8, alpha=0.3)

    axs[1, 0].errorbar(ts+offsets[i], lpm[:, calls[i]], yerr=lpci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8)
    axs[1, 0].errorbar(ts+offsets[i], lbm[:, calls[i]], yerr=lbci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8, alpha=0.3)

    axs[0, 1].scatter(ts+offsets[i], gptpm[:, calls[i]], color=cmap[i], s=60, marker='o', label=f'{i+3} neurons') # 'o' for global stim, 'x' for local stim. specify in caption
    axs[0, 1].scatter(ts+offsets[i], gbtpm[:, calls[i]], color=cmap[i], alpha=0.3, s=60, marker='o')

    axs[1, 1].scatter(ts+offsets[i], lptpm[:, calls[i]], color=cmap[i], s=60, marker='x', label=f'{i+3} neurons')
    axs[1, 1].scatter(ts+offsets[i], lbtpm[:, calls[i]], color=cmap[i], alpha=0.4, s=60, marker='x')

    axs[0, 1].errorbar(ts+offsets[i], gptpm[:, calls[i]], yerr=gptpci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8)
    axs[0, 1].errorbar(ts+offsets[i], gbtpm[:, calls[i]], yerr=gbtpci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8, alpha=0.3)

    axs[1, 1].errorbar(ts+offsets[i], lptpm[:, calls[i]], yerr=lptpci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8)
    axs[1, 1].errorbar(ts+offsets[i], lbtpm[:, calls[i]], yerr=lbtpci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8, alpha=0.3)

    axs[0, 2].scatter(ts+offsets[i], gptnm[:, calls[i]], color=cmap[i], s=60, marker='o', label=f'{i+3} neurons') # 'o' for global stim, 'x' for local stim. specify in caption
    axs[0, 2].scatter(ts+offsets[i], gbtnm[:, calls[i]], color=cmap[i], alpha=0.4, s=60, marker='o')

    axs[1, 2].scatter(ts+offsets[i], lptnm[:, calls[i]], color=cmap[i], s=60, marker='x', label=f'{i+3} neurons')
    axs[1, 2].scatter(ts+offsets[i], lbtnm[:, calls[i]], color=cmap[i], alpha=0.4, s=60, marker='x')

    axs[0, 2].errorbar(ts+offsets[i], gptnm[:, calls[i]], yerr=gptnci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8)
    axs[0, 2].errorbar(ts+offsets[i], gbtnm[:, calls[i]], yerr=gbtnci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8, alpha=0.3)

    axs[1, 2].errorbar(ts+offsets[i], lptnm[:, calls[i]], yerr=lptnci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8)
    axs[1, 2].errorbar(ts+offsets[i], lbtnm[:, calls[i]], yerr=lbtnci[:, :, calls[i]], fmt='none', ecolor=cmap[i], elinewidth=2, capsize=8, alpha=0.3)
    

axs[0, 0].set_xticks(ts)
axs[0, 0].set_ylim(0.975, 1)
axs[0, 0].set_yticks(np.arange(0.98, 1, 0.01))
axs[0, 0].set_ylabel('\\Huge{\\textbf{Predictive accuracy}}')
axs[0, 0].legend()

axs[0, 1].set_ylim(-0.05, 1)
axs[0, 1].set_yticks(np.arange(0, 1, 0.2))
axs[0, 1].set_ylabel('\\Huge{\\textbf{True positive rate}}')
axs[0, 1].legend()

axs[0, 2].set_ylim(0.984, 1.001)
axs[0, 2].set_yticks(np.arange(0.984, 1, 0.004))
axs[0, 2].set_ylabel('\\Huge{\\textbf{True negative rate}}')
axs[0, 2].legend(loc=4)

axs[1, 0].set_xticks(ts)
axs[1, 0].set_xlabel('\\Huge{\\textbf{Timeshift (ms)}}')
axs[1, 0].set_ylabel('\\Huge{\\textbf{Predictive accuracy}}')
axs[1, 0].legend()

axs[1, 1].set_xlabel('\\Huge{\\textbf{Timeshift (ms)}}')
axs[1, 1].set_ylabel('\\Huge{\\textbf{True positive rate}}')
axs[1, 1].legend()

axs[1, 2].set_xlabel('\\Huge{\\textbf{Timeshift (ms)}}')
axs[1, 2].set_ylabel('\\Huge{\\textbf{True negative rate}}')
axs[1, 2].legend(loc=4)

plt.tight_layout()
# plt.savefig('figs/cm metrics 2.png', dpi=400)
# plt.show()