from dataf import *
import scipy.stats as stats

def hodges_lehmann_2sample(group_a, group_b):
    # Calculate every possible difference (a - b)
    diffs = np.subtract.outer(group_a, group_b).flatten()
    # The HL estimator is the median of these differences
    return np.median(diffs)

pl = np.load('local/local stim p revised.npy') # First index: experiment num, second index: timeshift, third index: subset size
bl = np.load('local/local stim b revised.npy')
pg = np.load('global/global stim p revised.npy')
bg = np.load('global/global stim b revised.npy')
tpl = np.load('local/local stim p_tp revised.npy')
tpg = np.load('global/global stim p_tp revised.npy')
tnl = np.load('local/local stim p_tn revised.npy')
tng = np.load('global/global stim p_tn revised.npy')

# print model performance over baseline
best_local = pl[:, 2, -1]
best_local_tp = tpl[:, 2, -1]
best_local_tn = tnl[:, 2, -1]
best_global = pg[:, 1, -1]
best_global_tp = tpg[:, 2, -1]
best_global_tn = tng[:, 2, -1]

print(f'Mean best local stim predictive accuracy:\t{np.mean(best_local)}')
print(f'Mean best global stim predictive accuracy:\t{np.mean(best_global)}')
print(f'Mean best local stim predictive true positive:\t{np.mean(best_local_tp)}')
print(f'Mean best global stim predictive true positive:\t{np.mean(best_global_tp)}')
print(f'Mean best local stim predictive true negative:\t{np.mean(best_local_tn)}')
print(f'Mean best global stim predictive true negative:\t{np.mean(best_global_tn)}')


# choose 2 for second index if 0ms timeshift, choose 1 for second index if -100 ms timeshift.
samples_l_0 = pl[:, 2, -1]
samples_g_0 = pg[:, 2, -1]
base_g_0 = bg[:, 2, -1]
base_l_0 = bl[:, 2, -1]
samples_l_n100 = pl[:, 1, -1]
samples_g_n100 = pg[:, 1, -1]
base_g_n100 = bg[:, 1, -1]
base_l_n100 = bl[:, 1, -1]

# Shapiro-Wilk normality for both global and focal samples at 0 ms
shap_l0 = stats.shapiro(samples_l_0)
shap_g0 = stats.shapiro(samples_g_0)

# Shapiro-Wilk normality for both global and focal samples at -100 ms
shap_l1 = stats.shapiro(samples_l_n100)
shap_g1 = stats.shapiro(samples_g_n100)

print(
    f'Shapiro test on local p-value at 0ms timeshift:\t\t{shap_l0.pvalue:.4f}\nShapiro test on global p-value at 0ms timeshift:\t{shap_g0.pvalue:.10f}')

print(
    f'Shapiro test on local p-value at -100ms timeshift:\t{shap_l1.pvalue:.6f}\nShapiro test on global p-value at -100ms timeshift:\t{shap_g1.pvalue:.4f}')

# mann-whitney for non-parametric distributions 0ms
_, pval0 = stats.mannwhitneyu(samples_l_0, samples_g_0, alternative='two-sided')
_, pval_base0 = stats.mannwhitneyu(samples_g_0, base_g_0, alternative='two-sided')

# mann-whitney for non-parametric distributions -100ms
_, pval1 = stats.mannwhitneyu(samples_g_n100, samples_l_n100, alternative='two-sided')
_, pval_base1 = stats.ttest_ind(samples_g_n100, base_g_n100, equal_var=False)

# hodges-lehmann estimator for loc params 0ms
diff0 = hodges_lehmann_2sample(samples_l_0, samples_g_0)

# hodges-lehmann estimator for loc params 0ms
diff1 = hodges_lehmann_2sample(samples_l_n100, samples_g_n100)

print(f'Mann-Whitney test for samples 0ms:\t\t{pval0}\nHodges-Lehmann difference 0ms:\t\t\t{diff0}')

print(f'Mann-Whitney test for samples vs base 0ms global:\t{pval_base0}')

print(f'Mann-Whitney test for samples -100ms:\t\t{pval1}\nHodges-Lehmann difference -100ms:\t\t{diff1}')

print(f'Mann-Whitney test for samples vs base -100ms global:\t{pval_base1}')

