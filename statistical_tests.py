from dataf import *
import scipy.stats as stats

def hodges_lehmann_2sample(group_a, group_b):
    # Calculate every possible difference (a - b)
    diffs = np.subtract.outer(group_a, group_b).flatten()
    # The HL estimator is the median of these differences
    return np.median(diffs)

pl = np.load('local/local stim p revised.npy')
bl = np.load('local/local stim b revised.npy')
pg = np.load('global/global stim p revised.npy')
bg = np.load('global/global stim b revised.npy')

# choose 2 for second index if 0ms timeshift, choose 1 for second index if -100 ms timeshift.
samples_l = pl[:, 2, -1]
samples_g = pg[:, 2, -1]
base_g = bg[:, 2, -1]
base_l = bl[:, 2, -1]

print(samples_g)

# quit()

# Shapiro-Wilk normality for both global and focal samples
shap_l = stats.shapiro(samples_l)
shap_g = stats.shapiro(samples_g)

print(
    f'Shapiro test on local p-value:\t{shap_l.pvalue:.4f}\nShapiro test on global p-value:\t{shap_g.pvalue:.4f}')

# mann-whitney for non-parametric distributions
_, pval = stats.mannwhitneyu(samples_l, samples_g, alternative='two-sided')
_, pval_base = stats.mannwhitneyu(samples_g, base_g, alternative='two-sided')

# hodges-lehmann estimator for loc params
diff = hodges_lehmann_2sample(samples_l, samples_g)

print(f'Mann-Whitney test for samples:\t\t{pval}\nHodges-Lehmann difference:\t\t{diff}')

print(f'Mann-Whitney test for samples vs local:\t\t{pval_base}')

