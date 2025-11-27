from dataf import *
import scipy.stats as stats

pl = np.load('local/local stim p.npy')
pg = np.load('global/global stim p.npy')

samples_l = pl[:, 2, -1]
samples_g = pg[:, 2, -1]

print(f"Local samples:\t{samples_l}\nGlobal samples:\t{samples_g}\n\n")

shap_l = stats.shapiro(samples_l)
shap_g = stats.shapiro(samples_g)

print(f'Shapiro test on local:\t{shap_l}\nShapiro test on global:\t{shap_g}')

