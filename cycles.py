import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from pylab import savefig

cycles_paths = os.path.join(os.getcwd(),'cycles_plots')

full_df = pd.read_csv('swap_corrected_templates_soft_dtw_clusters7_gamma1.csv')
full_df = full_df.head(1600)
for counter,column in enumerate(full_df.columns,start=1):
    full_df = full_df.rename(columns={column:"PC "+str(counter)})

if not os.path.exists(cycles_paths): os.makedirs(cycles_paths)
plt.figure(0)
correlations = full_df[full_df.columns].corr(method='pearson')
heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)

figure = heat_ax.get_figure()    
figure.savefig(os.path.join(cycles_paths,'cycles_corr.png'), dpi=400)

t = np.arange(0,len(full_df),1)
t = list(t)

cycles = []
for column in full_df.columns:
    cycles.append(list(full_df[column]))

font = {'family':'serif',
'color':'darkred',
'weight':'normal',
'size': 14,
}

plt.style.use('ggplot')
for counter,cycle in enumerate(cycles,start=1):
    fig1 = plt.figure(counter)
    cycle_name='Cycle '+str(counter)
    plt.plot(t, cycle, color=(0/255,100/255,200/255), linestyle='-', label=cycle_name)
    plt.legend(loc='best')
    plt.title(r'Pitch Angle $\beta$ Profile', fontdict=font)
    plt.xlabel(r'$\mathbf{Time}$ (sec)',fontdict=font)
    plt.ylabel(r'$\mathbf{\beta}$ / $\mathbf{\beta_{max}}$',fontdict=font)
    plt.legend()
    plt.ylim(0,1)
    plt.xlim(0,len(t))
    plt.grid(color='darkred')
    plt.savefig(os.path.join(cycles_paths,cycle_name+'.png'))

#plt.show()



        