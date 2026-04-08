import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

x = np.array([5e3, 10e3, 20e3, 40e3])

y = np.array([1.59, 2.16, 3.97, 6.08]) # MoG

yerr_upper = np.array([0.73, 0.41, 1.42, 1.16])
yerr_lower = np.array([1.26, 0.83, 0.74, 0.86])

y_2 = np.array([0.31, 1.15, 1.75, 4.25]) # NF

yerr_upper_2 = np.array([0.75, 0.90, 0.58, 0.85])
yerr_lower_2 = np.array([1.00, 1.46, 0.62, 0.82])

# Constant offset for separation
dx = 200
x_left  = x - dx
x_right = x + dx

fig, ax = plt.subplots(figsize=(6, 4))

# =======================
# NF (left, C1)
# =======================

ax.errorbar(
    x_left[:2], y_2[:2],
    yerr=[yerr_lower_2[:2], yerr_upper_2[:2]],
    fmt='o', markersize=7,
    color='C1',
    ecolor='C1', elinewidth=1.5, capsize=3
)

ax.errorbar(
    x_left[2:], y_2[2:],
    yerr=[yerr_lower_2[2:], yerr_upper_2[2:]],
    fmt='*', markersize=10,
    color='C1',
    ecolor='C1', elinewidth=1.5, capsize=3
)

ax.plot(x, y_2, linestyle='--', color='C1', alpha=0.7, label='NF as ref')


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

ax.plot(x, y, linestyle='--', color='C0', alpha=0.7, label='MoG as ref')


# =======================
# Custom legend (markers)
# =======================

marker_legend = [
    Line2D([0], [0], marker='o', color='black', linestyle='None',
           markersize=7, label='empirical'),
    Line2D([0], [0], marker='*', color='black', linestyle='None',
           markersize=10, label=f'$\chi^2$')
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
plt.savefig('MoG_Zscore.pdf')
plt.close()