from dataf import *
import scipy.stats as stats

pl = np.load('local/local stim p.npy')
bl = np.load('local/local stim b.npy')
pg = np.load('global/global stim p.npy')
bg = np.load('global/global stim b.npy')

samples_l = pl[:, 1, -1] # choose 2 for second index if 0ms timeshift, choose 1 for second index if -100 ms timeshift.
samples_g = pg[:, 1, -1]
base_g = bg[:, 1, -1]
base_l = bl[:, 1, -1]

shap_l = stats.shapiro(samples_l)
shap_g = stats.shapiro(samples_g)

print(f'Shapiro test on local p-value:\t{shap_l.pvalue:.4f}\nShapiro test on global p-value:\t{shap_g.pvalue:.4f}')

testing = stats.ttest_ind(samples_g, samples_l)

print(testing)
