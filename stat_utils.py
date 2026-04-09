import numpy as np

import scipy.stats as stats



def emp_pvalue(t_null, t_obs):
    """
    Empirical one-sided p-value from a Monte Carlo null sample.
    Large values of t_obs are assumed to be more anomalous.
    """
    t_null = np.asarray(t_null)
    count = np.count_nonzero(t_null >= t_obs)
    p = (count + 1) / (len(t_null) + 1)
    return p

def emp_zscore(t_null, t_obs):
    """
    Empirical one-sided Z score from a Monte Carlo null sample.
    """
    p = emp_pvalue(t_null, t_obs)
    z = stats.norm.isf(p)
    return z

def Z_score(t_null, t_data):
    """
    Compute Z scores corresponding to the median, 0.16, and 0.84 quantiles
    of t_data, using the empirical null distribution t_null.

    Returns
    -------
    z_med : float
        Z score for the median of t_data
    z_low : float
        Z score for the 0.16 quantile of t_data
    z_high : float
        Z score for the 0.84 quantile of t_data
    """

    print("Empirical Z score")
    print()

    t_data = np.asarray(t_data)

    t_med = np.quantile(t_data, 0.50)
    t_low = np.quantile(t_data, 0.16)
    t_high = np.quantile(t_data, 0.84)

    z_med = emp_zscore(t_null, t_med)
    z_low = emp_zscore(t_null, t_low)
    z_high = emp_zscore(t_null, t_high)

    print(f"median test statistic = {t_med:.6g}")
    print(f"p value = {emp_pvalue(t_null, t_med):.3e}")
    print(f"Z score = {z_med:.3f}")
    print(f"Z = {z_med:.2f} (+{z_high - z_med:.2f}/-{z_med - z_low:.2f})")
    print()

    return z_med, z_low, z_high

def chi2_pvalue(t_obs, dof):
    """
    One-sided p-value from a chi2 distribution with given dof.
    """

    return stats.chi2(df=dof).sf(t_obs)

def chi2_zscore(t_obs, dof):
    """
    One-sided Z score from a chi2 distribution with given dof.
    Numerically stable also for very small p-values.
    """
    p = chi2_pvalue(t_obs, dof)
    z = stats.norm.isf(p)
    return z

def Z_score_chi2(t_null, t_data):
    """
    Compute Z scores corresponding to the 0.50, 0.16, and 0.84 quantiles
    of t_data, using a chi2 distribution whose effective dof is estimated
    from the mean of t_null.
    """

    print("χ² Z score")
    print()

    dof = np.mean(t_null)

    print(f"chi2 dof = {dof}")

    t_data = np.asarray(t_data)

    t_med  = np.quantile(t_data, 0.50)
    t_low  = np.quantile(t_data, 0.16)
    t_high = np.quantile(t_data, 0.84)

    z_med  = chi2_zscore(t_med, dof)
    z_low  = chi2_zscore(t_low, dof)
    z_high = chi2_zscore(t_high, dof)

    p_med = chi2_pvalue(t_med, dof)

    print(f"p value = {p_med:.3e}")
    print(f"Z score = {z_med:.3f}")
    print(f"Z = {z_med:.2f} (+{z_high - z_med:.2f}/-{z_med - z_low:.2f})")
    print()

    return z_med, z_low, z_high