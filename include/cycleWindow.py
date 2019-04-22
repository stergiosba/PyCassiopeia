#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""
import os
import time
from tqdm import tqdm
from .utils import windowUnits
import pandas as pd
import seaborn as sns

def cycleWindow(_data_df,_fit_df,_window_settings,__model_path,_csv_flag = True):
    begin = time.time()
    print(_data_df)
    
    csv_save_name = 'train_data.csv'
    #exit setting
    w_size = _window_settings[0]
    w_step = _window_settings[1]
    #settings_path = os.path.exists(__model_path+"/"+csv_save_name)
    print(50*"-")
    print("~$> Initializing Window Making Processing for Engine Cycles")
    print(50*"-")
    print("~$> Window size",w_size,"seconds.")
    print("~$> Window step",w_step,"seconds.")
    print(50*"-")
    
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(_data_df),w_size,w_step)

    print("~$> Total Windows Progression")
    for cycle in _data_df:
        cycle_df = _data_df[cycle]
        w_start = 0
        w_end = w_size
        with tqdm(total = w_count,desc = "~$> ",unit="win") as pbar:
            ACC_LIST = []
            DEC_LIST = []
            for window in range(cycle_df.index.min(),cycle_df.index.max(),w_step):
                window_df = cycle_df[w_start:w_end]
                if len(window_df)!=w_size:
                    continue
                window_df = window_df.reset_index(drop=True)
                acc_list = []
                dec_list = []
                for time_step in window_df.index:
                    if time_step==0:
                        pass
                    else:
                        acc = window_df[time_step]-window_df[time_step-1]
                        #if acc>0:
                        acc_list.append(acc)
                        #else:
                        #    dec_list.append(acc)
                ave_win_acc = sum(acc_list)/len(acc_list)
                #ave_win_dec = sum(dec_list)/len(dec_list)
                ACC_LIST.append(ave_win_acc)
                #DEC_LIST.append(ave_win_dec)
                w_start+=w_step
                w_end+=w_step
                #prev_ave_win_acc = ave_win_acc
                pbar.update(n=1)
    total = pd.DataFrame(ACC_LIST)
    os.makedirs(__model_path)
    total.to_csv(__model_path+"/"+csv_save_name)
            
    '''
            acc_list = []
            dec_list = []
            for rev in window_df.index:
                if rev==0:
                    pass
                else:
                    acc = window_df[__measurements[0]][rev]-window_df[__measurements[0]][rev-1]
                    if acc<0:
                        acc_list.append(acc)
                    else:
                        dec_list.append(acc)
            ave_win_acc = sum(acc_list)/len(acc_list)
            ave_win_dec = sum(dec_list)/len(dec_list)
            
            if w_start == 0:
                label = 2 #Starting with Steady
                prev_ave_win_acc = ave_win_acc
            else:
                if (ave_win_acc-prev_ave_win_acc)<-0.02:
                    label = 0 #Decel
                elif (ave_win_acc-prev_ave_win_acc)>0.02:
                    label = 1 #Accel
                else:
                    label = 2 #Steady
            #del acc_list
            _fit_df = _fit_df.append({
                'LABEL': label,
                'N_MAX': window_df[_measurements[0]].max(),
                'N_MIN': window_df[_measurements[0]].min(),
                'N_AVE': window_df[_measurements[0]].mean(),
                'N_IN' : window_df[_measurements[0]][window_df.index.min()],
                'N_OUT': window_df[_measurements[0]][window_df.index.max()],
                'A_AVE': ave_win_acc
                },ignore_index=True)
                
    print("~$> Plotting Pearson Correlation Matrix")
    correlations = _fit_df[_fit_df.columns].corr(method='pearson')
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)

    os.makedirs(__model_path)
    if _csv_flag == True:
        _fit_df.to_csv(__model_path+"/"+csv_save_name)

    finish = time.time()
    print(50*"-")
    print("~$> Time for data process was",round(finish-begin,2),"seconds.")
    print(50*"-")
    '''
    return _fit_df
