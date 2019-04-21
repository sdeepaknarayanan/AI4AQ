
# coding: utf-8

# In[58]:


import matplotlib.pyplot as plt
import numpy as np


# In[59]:


import pandas as pd
df = pd.read_csv('../../Datasets/Updated_Delhi_Scaled.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
station_list = list(df['Station'].unique())




active_l = []
rand_ = []
for i,station in enumerate(station_list):
    active = pd.read_csv('Data/AL/'+station+'.csv')
    active = active.loc[:, ~active.columns.str.contains('^Unnamed')]

    rd = pd.read_csv('Data/Random/'+station+'.csv')
    rd = rd.loc[:, ~rd.columns.str.contains('^Unnamed')]

    if i==10 or i==14:
        active_l.append(np.array(active))
        rand_.append(np.array(rd))

import matplotlib
from math import sqrt
SPINE_COLOR = 'gray'
def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': [r'\usepackage{gensymb}'],
              'axes.labelsize': 8, # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8, # was 10
              'legend.fontsize': 8, # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax



latexify()
# plt.subplot(n_rows=2,n_cols=1)
# plt.title('Station 11')
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
# ax1.title('saf')
f.text(0.5,0.02,'Number of Days Elapsed',ha='center',fontsize=8)
f.text(0.02,0.5,'Mean Absolute Error',va='center',rotation='vertical',fontsize=8)
l11,= ax1.plot([i for i in range(32)],active_l[0], marker='*',label='QBC Sampling')
l12, =ax1.plot([i for i in range(32)],rand_[0],marker='.',label='Random Sampling')
l21,=ax2.plot([i for i in range(32)],active_l[1],marker='*',label='QBC Sampling')
l22,=ax2.plot([i for i in range(32)],rand_[1], marker='.',label='Random Sampling')
for j in range(1,31,5):
    ax1.axvline(j,ls='--',color='k',lw=0.5)
    ax2.axvline(j,ls='--',color='k',lw=0.5)
plt.suptitle('S11 and S15')
handles, labels = ax1.get_legend_handles_labels()
f.legend(handles, labels,frameon=True)

