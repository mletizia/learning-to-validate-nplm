import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from stat_utils import emp_zscore, Z_score, Z_score_chi2


def err_bar(hist, n_samples, density=True):
    bins_counts = hist[0]
    bins_limits = hist[1]

    x = 0.5 * (bins_limits[1:] + bins_limits[:-1])
    bins_width = bins_limits[1:] - bins_limits[:-1]

    if density:
        err = np.sqrt(bins_counts / (n_samples * bins_width))
    else:
        err = np.sqrt(bins_counts)

    return x, err


def plot_ref_data(ref, data, name=None, out_path=None, title=None,
                  density=True, bins=10,
                  c_ref='#abd9e9', e_ref='#2c7bb6',
                  c_sig='#fdae61', e_sig='#d7191c',
                  plot_chi2=True):
    """
    Plot reference vs data t distribution.
    """

    plt.figure(figsize=(8, 5))
    plt.style.use('classic')

    ref = np.asarray(ref)
    data = np.asarray(data)

    # set uniform bins across all data points
    bins = np.histogram(np.hstack((ref, data)), bins=bins)[1]

    # reference
    hist_ref = plt.hist(ref, bins=bins, color=c_ref, edgecolor=e_ref,
                        density=density, label='Reference')
    x_err, err = err_bar(hist_ref, ref.shape[0], density=density)
    plt.errorbar(x_err, hist_ref[0], yerr=err, color=e_ref,
                 marker='o', ms=6, ls='', lw=1, alpha=0.7)

    # data
    hist_data = plt.hist(data, bins=bins, color=c_sig, edgecolor=e_sig,
                         alpha=0.7, density=density, label='Data')
    x_err, err = err_bar(hist_data, data.shape[0], density=density)
    plt.errorbar(x_err, hist_data[0], yerr=err, color=e_sig,
                 marker='o', ms=6, ls='', lw=1, alpha=0.7)

    plt.ylim(bottom=0)

    # summary numbers
    md_tref = np.median(ref)
    md_tdata = np.median(data)
    max_zemp = emp_zscore(ref, np.max(ref))

    # reuse your helper functions
    zemp_50, zemp_16, zemp_84 = Z_score(ref, data)

    if plot_chi2:
        zchi2_50, zchi2_16, zchi2_84 = Z_score_chi2(ref, data)
        dof = np.mean(ref)

    if plot_chi2:
        res = (
            f"md t_ref = {md_tref:.2f}\n"
            f"md t_data = {md_tdata:.2f}\n"
            f"max Z_emp = {max_zemp:.2f}\n"
            f"Z_emp = {zemp_50:.2f} (+{zemp_84 - zemp_50:.2f}/-{zemp_50 - zemp_16:.2f})\n"
            f"Z_chi2 = {zchi2_50:.2f} (+{zchi2_84 - zchi2_50:.2f}/-{zchi2_50 - zchi2_16:.2f})"
        )
    else:
        res = (
            f"md t_ref = {md_tref:.2f}\n"
            f"md t_data = {md_tdata:.2f}\n"
            f"max Z_emp = {max_zemp:.2f}\n"
            f"Z_emp = {zemp_50:.2f} (+{zemp_84 - zemp_50:.2f}/-{zemp_50 - zemp_16:.2f})"
        )

    # plot chi2 and set xlim
    if plot_chi2:
        chi2_range = stats.chi2.ppf(q=[1e-5, 0.999], df=dof)
        x = np.arange(chi2_range[0], chi2_range[1], 0.05)
        chisq = stats.chi2.pdf(x, df=dof)
        plt.plot(x, chisq, color='#d7191c', lw=2,
                 label=rf'$\chi^2({dof:.2f})$')
        xlim = (min(chi2_range[0], np.min(ref) - 1),
                max(chi2_range[1], np.max(data) + 1))
    else:
        xlim = (np.min(ref) - 1, np.max(data) + 1)

    plt.xlim(xlim)

    if title:
        plt.title(title, fontsize=20)

    plt.ylabel('P(t)', fontsize=20)
    plt.xlabel('t', fontsize=20)

    ax = plt.gca()

    plt.legend(loc="upper right", frameon=True, fontsize=10)

    ax.text(0.75, 0.55, res, color='black', fontsize=10,
            bbox=dict(facecolor='none', edgecolor='black',
                      boxstyle='round,pad=.5'),
            transform=ax.transAxes)

    plt.tight_layout()

    if out_path:
        plt.savefig(f"{out_path}/refdata_{name}.pdf", bbox_inches='tight')

    plt.show()
    plt.close()


def plot_ref_with_fitted_chi2_ks(
    ref,
    bins=7,
    density=True,
    title=None,
    c_ref='#abd9e9',
    e_ref='#2c7bb6',
    c_fit='#2c7bb6',
):
    """
    Plot the reference distribution together with a chi2 fit whose
    effective dof is estimated from the sample mean, and report the
    KS-test p-value for compatibility with that chi2 model.
    """

    ref = np.asarray(ref)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "axes.labelsize": 18,
        "font.size": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white')

    # bins
    bins = np.histogram(ref, bins=bins)[1]

    # histogram
    hist = ax.hist(
        ref,
        bins=bins,
        density=density,
        color=c_ref,
        edgecolor=e_ref,
        alpha=0.6,
        linewidth=1.2,
        label='Reference'
    )

    # error bars (reusing your shared function)
    x_err, err = err_bar(hist, ref.shape[0], density=density)
    ax.errorbar(
        x_err, hist[0], yerr=err,
        color=e_ref, marker='o', ms=5, ls='', lw=1, alpha=0.8
    )

    # dof from empirical mean (consistent with your chi2 Z code)
    dof_hat = np.mean(ref)

    # x-range from histogram
    xmin, xmax = bins[0], bins[-1]

    # add a small margin (5%)
    dx = 0.1 * (xmax - xmin)
    xmin_plot = max(0, xmin - dx)   # chi2 support starts at 0
    xmax_plot = xmax + dx

    # chi2 curve
    x = np.linspace(xmin_plot, xmax_plot, 600)
    pdf = stats.chi2.pdf(x, df=dof_hat)
    ax.plot(
        x, pdf, color=c_fit, lw=2.5,
        label=rf'$\chi^2(\mathrm{{dof}}={dof_hat:.1f})$'
    )

    # set limits
    ax.set_xlim(xmin_plot, xmax_plot)

    # KS goodness-of-fit
    stat, pvalue = stats.kstest(ref, 'chi2', args=(dof_hat,))

    ax.text(
        0.73, 0.70, f"KS p-value = {pvalue:.3f}",
        transform=ax.transAxes,
        bbox=dict(facecolor='none', edgecolor='black',
                  boxstyle='round,pad=.4'),
        fontsize=12
    )

    if title:
        ax.set_title(title)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Density" if density else "Counts")
    ax.set_ylim(bottom=0)
    ax.tick_params(direction='in', top=True, right=True, length=5)

    leg = ax.legend(loc="upper right", frameon=True, fancybox=False)
    leg.get_frame().set_edgecolor("black")

    plt.tight_layout()
    plt.show()