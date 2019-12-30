#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:34 2019

@author: stergios
"""
import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from p_tqdm import p_map
from numpy.random import seed

from .windows import trendWindow, patternWindow
import include.network.net_constants as netco

seed(1)

def trendProcess(data_path,model_path,features_list,window_settings):
    print(50*"#")
    print("~$> Initializing Window Making Processing for Speed Trend Prediction")
    print(50*"#")
    print("~$> Loading the dataset from " +data_path)
    full_df = pd.read_csv(data_path+'/swap_corrected_templates_soft_dtw_clusters7_gamma1.csv')
    print("~$> Loaded the dataset")
    print("~$> Collected data for",full_df.shape[0],"seconds.")
    begin = time.time()
    args = [full_df[cycle] for cycle in full_df]
    fit_df = pd.DataFrame(columns=features_list)
    print("~$> Total Windows Progression")
    results = p_map(trendWindow,args,model_path,tuple(features_list),tuple(window_settings))

    for result in results:
        fit_df = fit_df.append(result,sort=False,ignore_index=True)

    print(50*"-")    
    print("~$> Plotting Pearson Correlation Matrix")
    finish = time.time()
    correlations = fit_df[fit_df.columns].corr(method='pearson')
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)
    fit_df.to_csv(os.path.join(model_path,netco.TRAIN+'.csv'),index=False)
    plt.show(block=False)
    print(50*"-")
    print("~$> Time for data process was",round(finish-begin,2),"seconds.")
    print(50*"-")
    

def patternProcess(data_path,model_path,features_list,window_settings):
    print(50*"#")
    print("~$> Initializing Window Making Processing for Combinator Pattern Prediction")
    print(50*"#")
    print("~$> Loading the dataset from " + data_path)
    full_df = pd.read_csv(data_path+'/swap_corrected_templates_soft_dtw_clusters7_gamma1.csv')
    print("~$> Loaded the dataset")
    print("~$> Collected data for",full_df.shape[0],"seconds.")
    begin = time.time()
    args = [full_df[cycle] for cycle in full_df]
    fit_df = pd.DataFrame(columns=features_list)
    print("~$> Total Windows Progression")
    results = p_map(patternWindow,args,model_path,tuple(features_list),tuple(window_settings))

    for result in results:
        fit_df = fit_df.append(result,sort=False,ignore_index=True)
    print(50*"-")    
    print("~$> Plotting Pearson Correlation Matrix")

    correlations = fit_df[fit_df.columns].corr(method='pearson')
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)
    #plt.show()
    if not os.path.exists(model_path): os.makedirs(model_path)
    fit_df.to_csv(os.path.join(model_path,netco.TRAIN+'.csv'),index=False)

    finish = time.time()
    print(50*"-")
    print("~$> Time for data process was",round(finish-begin,2),"seconds.")
    print(50*"-")
