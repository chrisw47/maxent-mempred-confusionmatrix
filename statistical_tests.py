from dataf import *
import scipy.stats as stats

pl = np.load('local/local stim p.npy')
pg = np.load('global/global stim p.npy')

samples_l = pl[:, 2, -1]
samples_g = pg[:, 2, -1]

shap_l = stats.shapiro(samples_l)
shap_g = stats.shapiro(samples_g)

print(f'Shapiro test on local:\t{shap_l.pvalue:.4f}\nShapiro test on global:\t{shap_g.pvalue:.4f}')

ml, stdl = np.mean(samples_l), np.std(samples_l)
mg, stdg = np.mean(samples_g), np.std(samples_g)

testing = stats.ttest_ind(samples_l, samples_g)

print(testing)
