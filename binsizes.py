from physf import *
from dataf import *
import matplotlib.ticker

plt.rcParams.update({
    'legend.fontsize': 16,
    'font.size': 22
})

# def metrics_by_binsize(df: pd.DataFrame):

#     b = 0  # baseline predictive accuracy
#     m = 0  # model predictive accuracy
#     bn = 1  # baseline true negative rate
#     mn = 0  # model true negative rate
#     bp = 0  # baseline true positive rate
#     mp = 0  # model true positive rate

#     corr_mat = construct_corr_matrix(df)

#     candidates, cand_corrs = filter_neurons(corr_mat, 60, SSIZE)

#     print(f'NEURONS CHOSEN:\t{candidates}.')
#     print(f'CORRELATIONS:\t{cand_corrs}.\n')

#     _, stim, net = select_neuron_subset(df, candidates)  # time bins are rows

#     j = maxent(stim, net, SSIZE)

#     b, m, mn, mp = cm_metrics(stim, net, j)

#     return b, m, bn, mn, bp, mp

# x_type = input(f'Global (\'global\') or local (\'local\') stim? ')

# TS = -1 if x_type == 'global' else 0  # timeshift 0 ms
# SSIZE = 6  # subsystem size of 6

# xp_list = [9, 12, 14, 15, 16, 17, 18, 19]
# bsizes = [0.025, 0.05, 0.1, 0.2] # in seconds

# b, p, b_tn, p_tn, b_tp, p_tp = [], [], [], [], [], []

# for xp in xp_list:

#     bb, mb, bnb, mnb, bpb, mpb = [], [], [], [], [], [] # metrics with b at the end to denote binsizes

#     for bin in bsizes:

#         print(f'ANALYZING EXPERIMENT {xp} WITH BIN SIZE {1000 * bin} MS.')
            
#         if x_type == 'local':
#             PATH = f'real data/exp{xp}_local.mat'
#             Ts, Cs = load_data(PATH)
#             _, full_array = neuron_system(
#                 Ts / bin, Cs, int(72000 / bin))
#             df = split_array(full_array, int(TS * (0.1 / bin)), bin)
        
#         elif x_type == 'global':
#             PATH = f'real data/experiment_{xp}_20h_stim.mat'
#             df = extract_opto(PATH, int(TS * (0.1 / bin)), bin)

#         else:
#             raise ValueError(
#                 'The stimulation type needs to be global or local.')
        
#         bl, ml, bln, mln, blp, mlp = metrics_by_binsize(df)

#         print('\n')
#         print(bl,ml,bln,mln,blp,mlp)
#         print('\n')

#         bb.append(bl)
#         mb.append(ml)
#         bnb.append(bln)
#         mnb.append(mln)
#         bpb.append(blp)
#         mpb.append(mlp)
    
#     b.append(bb)
#     p.append(mb)
#     b_tn.append(bnb)
#     p_tn.append(mnb)
#     b_tp.append(bpb)
#     p_tp.append(mpb)

# print(f'Baseline:\n{b}\n\n')
# print(f'Model:\n{p}\n\n')
# print(f'Baseline true negatives:\n{b_tn}\n\n')
# print(f'Model true negatives:\n{p_tn}\n\n')
# print(f'Baseline true positives:\n{b_tp}\n\n')
# print(f'Model true positives:\n{p_tp}\n\n')

# arr, anames = [b, p, b_tn, p_tn, b_tp, p_tp], ['b', 'p', 'b_tn','p_tn', 'b_tp', 'p_tp']

# for a, name in zip(arr, anames):
#     a = np.array(a)
#     np.save(f'{x_type}/{x_type} bsizes {name}.npy', a)

bg = np.load(f'global/global bsizes b.npy')
pg = np.load(f'global/global bsizes p.npy')
b_tn = np.load(f'global/global bsizes b_tn.npy')
pg_tn = np.load(f'global/global bsizes p_tn.npy')
b_tp = np.load(f'global/global bsizes b_tp.npy')
pg_tp = np.load(f'global/global bsizes p_tp.npy')

bl = np.load(f'local/local bsizes b.npy')
pl = np.load(f'local/local bsizes p.npy')
pl_tn = np.load(f'local/local bsizes p_tn.npy')
pl_tp = np.load(f'local/local bsizes p_tp.npy')

impg = (pg - bg) / (1 - bg) # improvement over baseline
impl = (pl - bl) / (1 - bl)

impg, impgci = mean_and_ci(impg, 90)
impl, implci = mean_and_ci(impl, 90)
pg_tn, pg_tnci = mean_and_ci(pg_tn, 90)
pl_tn, pl_tnci = mean_and_ci(pl_tn, 90)
pg_tp, pg_tpci = mean_and_ci(pg_tp, 90)
pl_tp, pl_tpci = mean_and_ci(pl_tp, 90)
b_tn, _ = mean_and_ci(b_tn, 90)
b_tp, _ = mean_and_ci(b_tp, 90)

binsizes = np.array([25, 50, 100, 200]) # in ms

f, axs = plt.subplots(1, 3, sharex=True, figsize=(15, 5))

axs[0].scatter(binsizes-2, impg, color='#004D40', s=50, marker='o')
axs[0].scatter(binsizes+2, impl, color='#1E88E5', s=50, marker='^')
axs[0].errorbar(binsizes-2, impg, yerr=impgci, fmt='none', elinewidth=2, ecolor='#004D40', capsize=5, alpha=0.3)
axs[0].errorbar(binsizes+2, impl, yerr=implci, fmt='none', elinewidth=2, ecolor='#1E88E5', capsize=5, alpha=0.3)
axs[0].set_xticks(binsizes)
axs[0].set_ylim(-0.1, 1)
axs[0].set_yticks(np.arange(0, 1.1, 0.2))
axs[0].set_xlabel('\\textbf{Time bin size (ms)}')
axs[0].set_ylabel('\\textbf{Improvement in}\n\\textbf{predictive accuracy}')

axs[1].scatter(binsizes, b_tn, color='#D81B60', s=50, marker='s', alpha=0.5, label='Baseline')
axs[1].scatter(binsizes-2, pg_tn, color='#004D40', s=50, marker='o', label='Optogenetic')
axs[1].scatter(binsizes+2, pl_tn, color='#1E88E5', s=50, marker='^', label='Electrical')
axs[1].errorbar(binsizes-2, pg_tn, yerr=pg_tnci, fmt='none', elinewidth=2, ecolor='#004D40', capsize=5, alpha=0.3)
axs[1].errorbar(binsizes+2, pl_tn, yerr=pl_tnci, fmt='none', elinewidth=2, ecolor='#1E88E5', capsize=5, alpha=0.3)
axs[1].set_xlabel('\\textbf{Time bin size (ms)}')
axs[1].set_ylabel('\\textbf{True negative rate}')

axs[2].scatter(binsizes, b_tp, color='#D81B60', s=50, marker='s', alpha=0.5)
axs[2].scatter(binsizes-2, pg_tp, color='#004D40', s=50, marker='o')
axs[2].scatter(binsizes+2, pl_tp, color='#1E88E5', s=50, marker='^')
axs[2].errorbar(binsizes-2, pg_tp, yerr=pg_tpci, fmt='none', elinewidth=2, ecolor='#004D40', capsize=5, alpha=0.3)
axs[2].errorbar(binsizes+2, pl_tp, yerr=pl_tpci, fmt='none', elinewidth=2, ecolor='#1E88E5', capsize=5, alpha=0.3)
axs[2].set_ylim(-0.05, 1)
axs[2].set_xlabel('\\textbf{Time bin size (ms)}')
axs[2].set_ylabel('\\textbf{True positive rate}')

axs[1].legend()

plt.tight_layout()
plt.savefig('figs/binsizes.png', dpi=500)
# plt.show()