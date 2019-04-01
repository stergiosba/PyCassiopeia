#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""
import pandas as pd
import seaborn as sns
import os

def windowMaker(_data_df,_fit_df,__measurements,__window_settings,_csv_flag,__model_path):
    csv_save_name = 'train_data.csv'
    #exit setting
    
    #settings_path = os.path.exists(__model_path+"/"+csv_save_name)
    print(50*"-")
    print("~$> Initializing Window Making Processing")
    print(50*"-")
    print("~$> Window size",__window_settings[0],"seconds.")
    print("~$> Window step",__window_settings[1],"seconds.")
    print(50*"-")
    #print("~$> You have chosen to segmentize the data per",segment_size,"seconds.")
    ave_df = pd.DataFrame()
    window_count = 0
    window_size = __window_settings[0]
    window_step = __window_settings[1]
    win_s = 0
    win_f = window_size
    window_count = 0
    for window in range(_data_df.index.min(),_data_df.index.max(),window_step):
        win_s+=window_step
        win_f+=window_step
        window_count+=1
    win_max = window_count
    
    ave_acc_df = pd.DataFrame()
    win_s = 0
    win_f = window_size
    window_count = 0
    for window in range(_data_df.index.min(),_data_df.index.max(),window_step):
        window_df = _data_df[win_s:win_f]
        window_df = window_df.reset_index(drop=True)
        acc_list = []
        for rev in window_df.index:
            if rev==0:
                pass
            else:
                acc = window_df[__measurements[0]][rev]-window_df[__measurements[0]][rev-1]
                acc_list.append(acc)
        ave_win_acc = sum(acc_list)/len(acc_list)
        #ave_acc_df = ave_acc_df.append(acc_list)
        if window_count == 0:
            label = 3 #Starting with Steady
            prev_ave_win_acc = ave_win_acc
        else:
            if (ave_win_acc-prev_ave_win_acc)<-0.05:
                label = 1 #Decel
            elif (ave_win_acc-prev_ave_win_acc)>0.05:
                label = 2 #Accel
            else:
                label = 3 #Steady
        #del acc_list
        _fit_df = _fit_df.append({
            'LABEL': label,
            'N_MAX': window_df[__measurements[0]].max(),
            'N_MIN': window_df[__measurements[0]].min(),
            'N_AVE': window_df[__measurements[0]].mean(),
            'N_IN' : window_df[__measurements[0]][window_df.index.min()],
            'N_OUT': window_df[__measurements[0]][window_df.index.max()],
            'A_AVE': ave_win_acc
            },ignore_index=True)
        win_s+=window_step
        win_f+=window_step
        prev_ave_win_acc = ave_win_acc
        window_count+=1
        print("~$> Window",window_count,'/',win_max)
        
    win_max = window_count
    print("~$> Total Windows Number is",window_count)
    print(50*"-")
    
    print("~$> Plotting Pearson Correlation Matrix")
    correlations = _fit_df[_fit_df.columns].corr(method='pearson')
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)
    os.makedirs(__model_path)
    if _csv_flag == True:
        _fit_df.to_csv(__model_path+"/"+csv_save_name)

    return _fit_df
