from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

params = [[13, 12], [2, 2], [15, 12], [4, 2], [14, 13], [2, 4]]
x      = np.linspace(0, 1, 100)

for idx, p in enumerate(params):

    a  = p[0]
    b  = p[1]

    rv = beta(a, b)

    plt.figure(figsize=(4, 3), dpi=100, constrained_layout=True)
    ax = plt.axes()
    ax.plot(x, rv.pdf(x), linewidth=10, c='black')
    ax.axvline(a/(a+b), linewidth=7, linestyle='dotted', c='r')
    ax.set_ylim(0-0.2, np.max(rv.pdf(x))+0.2)

    ax.tick_params(axis='both', which='both', colors='black', labelsize=45)
    ax.set_xlabel(r'$p$(reward)', fontsize=50)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.savefig(os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp1/prior_%u_%u.svg'%(p[0], p[1])), transparent=True)
    plt.savefig(os.path.join('/home/georgy/Documents/Dayan_lab/PhD/bandits/paper/figures/supp/supp1/prior_%u_%u.png'%(p[0], p[1])))
    plt.close()
