import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import savefig
from include.windows import onlineTrendPrediction
import include.network.net_constants as netco

cycles_paths = os.path.join(os.getcwd(),'cycles_plots')

full_df = pd.read_csv('swap_corrected_templates_soft_dtw_clusters7_gamma1.csv')
full_df = full_df.head(641)

sim_df = pd.read_csv('simulation_0.csv')
sim_df = sim_df.drop('N_ERROR_SQR',axis=1)
for counter,column in enumerate(full_df.columns,start=1):
    full_df = full_df.rename(columns={column:"PC_"+str(counter)})
if not os.path.exists(cycles_paths): os.makedirs(cycles_paths)

plt.ylim(0,1)
window_settings = [9,1]
features = netco.TREND_FEATURES
test_df = full_df[['PC_1']]
targets=[]
low = 0
high = 0
for index, row in test_df.iterrows():
    if index<window_settings[0]:
        targets.append(row[0])
    else:
        low+=1
        print(50*'-')
        del targets[0]
        targets.append(row[0])
        window_df = pd.Series(targets)
        go = onlineTrendPrediction(window_df)
        required_df = sim_df.iloc[high]
        required_df['LABEL'] = go['LABEL'][0]
        print(required_df)
    high+=1

'''
#fit_df = trendWindow(test_df,features,window_settings,os.getcwd())
sim_df.plot()
#plt.ylim(0,1)

plt.show()
'''
      