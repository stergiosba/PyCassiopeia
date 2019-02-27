#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""
import pandas as pd
import seaborn as sns
import numpy as np

def windowMaker(_data_df,_fit_df,__measurement,__window_features):
    print(50*"-")
    print("~$> Initializing Window Making Processing")
    print(50*"-")
    print("~$> Window size",__window_features[0],"seconds.")
    print("~$> Window step",__window_features[1],"seconds.")
    print(50*"-")
    #print("~$> You have chosen to segmentize the data per",segment_size,"seconds.")
    ave_df = pd.DataFrame()
    window_count = 0
    window_size = __window_features[0]
    window_step = __window_features[1]
    win_s = 0
    win_f = window_size
    window_count = 0
    for window in range(_data_df.index.min(),_data_df.index.max(),window_step):
        win_s+=window_step
        win_f+=window_step
        window_count+=1
    win_max = window_count
    
    win_s = 0
    win_f = window_size
    window_count = 0
    for window in range(_data_df.index.min(),_data_df.index.max(),window_step):
        window_df = _data_df[win_s:win_f]
        window_df = window_df.reset_index(drop=True)
        print(window_df)
        acc_list = []
        print(acc_list)
        for rev in window_df.index:
            if rev==0:
                pass
            else:
                acc = window_df[__measurement][rev]-window_df[__measurement][rev-1]
                acc_list.append(acc)
            ave_df = ave_df.append(acc_list)
            if acc < -0.00005:
                label =0 #Decel
            elif acc > 0.00005:
                label =1 #Accel
            else:
                label =2#S
            print(acc_list)
            #del acc_list
            _fit_df = _fit_df.append({
                'N_MAX': window_df[__measurement].max(),
                'N_MIN': window_df[__measurement].min(),
                'N_AVE': window_df[__measurement].mean(),
                'N_IN' : window_df[__measurement][window_df.index.min()],
                'N_OUT': window_df[__measurement][window_df.index.max()],
                'A_AVE': ave_df[0].mean(),
                'LABEL': label
                },ignore_index=True)
        win_s+=window_step
        print(_fit_df)
        win_f+=window_step
        window_count+=1
        print("Window",window_count,'/',win_max)
        
    win_max = window_count
    print("Total Windows Number is",window_count)
    print(50*"-")
    print("~$> Plotting Pearson Correlation Matrix")
    correlations = _fit_df[_fit_df.columns].corr(method='pearson')
    sns.heatmap(correlations, cmap="YlGnBu", annot = True)
    print(50*"-")
    fit_data_size = _fit_df.shape[0]
    _fit_df = _fit_df.sample(frac=1).reset_index(drop=True)
    fit_labels = _fit_df["LABEL"].values
    _fit_df=_fit_df.drop(["LABEL"],axis=1)
    _fit_df = _fit_df.values
    return (_fit_df,fit_labels)
    #fit_df = fit_df.transpose()
