import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

x = np.array([5e3, 10e3, 20e3, 50e3, 100e3, 200e3])
y = np.array([0.40, 0.93, 1.70, 4.07, 7.12, 11.81])

yerr_upper = np.array([1.17, 0.78, 0.63, 0.55, 0.92, 1.13])
yerr_lower = np.array([0.44, 0.80, 0.72, 1.24, 1.38, 1.06])

y_2 = np.array([0.13, 0.86, 1.34, 3.53, 6.48, 11.14])

yerr_upper_2 = np.array([0.73, 0.75, 0.98, 1.12, 1.04, 0.96])
yerr_lower_2 = np.array([0.96, 0.85, 1.01, 0.8, 1.26, 0.89])

# Constant offset for separation
dx = 500
x_left  = x - dx
x_right = x + dx

fig, ax = plt.subplots(figsize=(6, 4))

# =======================
# NF (left, C1)
# =======================

ax.errorbar(
    x_left[:3], y_2[:3],
    yerr=[yerr_lower_2[:3], yerr_upper_2[:3]],
    fmt='o', markersize=7,
    color='C1',
    ecolor='C1', elinewidth=1.5, capsize=3
)

ax.errorbar(
    x_left[3:], y_2[3:],
    yerr=[yerr_lower_2[3:], yerr_upper_2[3:]],
    fmt='*', markersize=10,
    color='C1',
    ecolor='C1', elinewidth=1.5, capsize=3
)

ax.plot(x, y_2, linestyle='--', color='C1', alpha=0.7, label='Gen. model as ref')


# =======================
# MoG (right, C0)
# =======================

ax.errorbar(
    x_right[:3], y[:3],
    yerr=[yerr_lower[:3], yerr_upper[:3]],
    fmt='o', markersize=7,
    color='C0',
    ecolor='C0', elinewidth=1.5, capsize=3
)

ax.errorbar(
    x_right[3:], y[3:],
    yerr=[yerr_lower[3:], yerr_upper[3:]],
    fmt='*', markersize=10, 
    color='C0',
    ecolor='C0', elinewidth=1.5, capsize=3
)

ax.plot(x, y, linestyle='--', color='C0', alpha=0.7, label='Ground truth as ref')


# =======================
# Custom legend (markers)
# =======================

marker_legend = [
    Line2D([0], [0], marker='o', color='black', linestyle='None',
           markersize=7, label='empirical'),
    Line2D([0], [0], marker='*', color='black', linestyle='None',
           markersize=10, label='extrapolated')
]

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + marker_legend,
          labels + ['empirical', f'$\chi^2$'],
          prop={'size': 10})


# =======================
# Styling
# =======================

# Horizontal sigma lines
for sig, ypos in zip([1, 3, 5], [1, 3, 5]):
    ax.axhline(y=ypos, color='k', linestyle='--', alpha=0.7)
    ax.text(x.max()*1.08, ypos + 0.6, f"{sig}$\\sigma$", va='center')

ax.set_xlabel(r"$N_D$", fontsize=12)
ax.set_ylabel(r"$Z$-score", fontsize=12)

ax.set_ylim(-2, 13)
ax.ticklabel_format(style='sci', axis='x', scilimits=(3, 3))

plt.tight_layout()
plt.savefig('Flowsim_Zscore.pdf')
plt.close()