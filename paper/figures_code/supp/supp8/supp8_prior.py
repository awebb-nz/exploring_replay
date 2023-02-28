from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import os

a, b = 7, 2
rv   = beta(a, b)
x    = np.linspace(0, 1, 100)

plt.figure(figsize=(4, 3), dpi=100, constrained_layout=True)
ax = plt.axes()

ax.plot(x, rv.pdf(x), linewidth=10, c='white')
ax.axvline(a/(a+b), linewidth=7, linestyle='dotted', c='r')
ax.set_ylim(0-0.2, np.max(rv.pdf(x))+0.2)

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='both', which='both', colors='white', labelsize=45)
ax.set_xlabel(r'$p$(closed)$=1$', fontsize=50)
ax.xaxis.label.set_color('white')

plt.savefig(os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig2/prior_closed.svg'), transparent=True)
plt.savefig(os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/fig2/prior_closed.png'))