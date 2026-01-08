from dataf import *
import scipy.stats as stats

pl = np.load('local/local stim p.npy')
bl = np.load('local/local stim b.npy')
pg = np.load('global/global stim p.npy')
bg = np.load('global/global stim b.npy')

# choose 2 for second index if 0ms timeshift, choose 1 for second index if -100 ms timeshift.
samples_l = pl[:, 2, -1]
samples_g = pg[:, 2, -1]
base_g = bg[:, 2, -1]
base_l = bl[:, 2, -1]

# Print predictive accuracy diffs at 0ms timeshift, 6 neurons
print(np.mean(samples_l) - np.mean(samples_g))

# Shapiro-Wilk normality for both global and focal samples
shap_l = stats.shapiro(samples_l)
shap_g = stats.shapiro(samples_g)

print(
    f'Shapiro test on local p-value:\t{shap_l.pvalue:.4f}\nShapiro test on global p-value:\t{shap_g.pvalue:.4f}')


# t-test to determine whether means are statistically distinguishable
testing = stats.ttest_ind(samples_g, samples_l)

print(testing)
