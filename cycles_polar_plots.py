import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from pylab import savefig

cycles_paths = os.path.join(os.getcwd(),'cycles_plots')

full_df = pd.read_csv('swap_corrected_templates_soft_dtw_clusters7_gamma1.csv')

for counter,column in enumerate(full_df.columns,start=1):
    full_df = full_df.rename(columns={column:"PC "+str(counter)})

if not os.path.exists(cycles_paths): os.makedirs(cycles_paths)
plt.figure(0)
correlations = full_df[full_df.columns].corr(method='pearson')

mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
heat_ax = sns.heatmap(correlations, cmap='YlGnBu', annot=True, mask=mask, linewidths=0.5)

figure = heat_ax.get_figure()    
figure.savefig(os.path.join(cycles_paths,'cycles_corr.png'), dpi=800)

t = np.arange(0,len(full_df),1)
t = list(t)

cycles = []
for column in full_df.columns:
    cycles.append(list(full_df[column]))

font = {#'family':'',
'color':'black',
'weight':'normal',
'size': 14,
}

# Creates four polar axes, and accesses them through the returned array

fig, axes = plt.subplots(2, 4,figsize=(192/10, 108/10), dpi=500)

heat_ax = sns.heatmap(correlations, cmap='YlGnBu', annot=True, mask=mask, linewidths=0.5,ax=axes[1,3])
for i in range(4):
    axes[0, i].set_ylim(0, 1)
    axes[0, i].grid()
    axes[0, i].set_xlim(0, len(cycles[i]))
    axes[0, i].set_title('Pattern '+str(i+1))
    if i==0:
        axes[0, i].set_ylabel(r'$\mathbf{\beta}$ / $\mathbf{\beta_{max}}$',fontdict=font)

    axes[0, i].set_xlabel(r'$\mathbf{Time}$ (sec)',fontdict=font)
    axes[0, i].plot(t, cycles[i], color='blue')

for i in range(3):
    axes[1, i].set_ylim(0, 1)
    axes[1, i].grid()
    axes[1, i].set_xlim(0, len(cycles[i+4]))
    axes[1, i].set_title('Pattern '+str(i+5))
    if i==0:
        axes[1, i].set_ylabel(r'$\mathbf{\beta}$ / $\mathbf{\beta_{max}}$',fontdict=font)

    axes[1, i].set_xlabel(r'$\mathbf{Time}$ (sec)',fontdict=font)
    axes[1, i].plot(t, cycles[i+4], color='blue')

plt.subplots_adjust(left=0.05)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(bottom=0.05)
plt.subplots_adjust(hspace=0.30)
plt.savefig('new_cyclees.png',dpi=500)
plt.show()

        