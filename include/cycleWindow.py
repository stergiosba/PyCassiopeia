#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""
import os
import time
import statistics as stats
from tqdm import tqdm
from .utils import windowUnits
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def cycleWindow(_data_df,_fit_df,_window_settings,__model_path,_csv_flag = True):
    begin = time.time()
    #print(_data_df)
    
    csv_save_name = 'train_data'
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
        cycle_MAX_rev = []
        cycle_AVE_acc = []
        cycle_MAX_acc = []
        cycle_STD_acc = []
        cycle_AVE_dec = []
        cycle_MAX_dec = []
        cycle_P_N_030 = []
        cycle_P_N_3050 = []
        cycle_P_N_5080 = []
        cycle_P_N_80100 = []
        cycle_P_D_12 = []
        cycle_P_D_23 = []

        Cycle_Final = pd.DataFrame()
        with tqdm(total = w_count,desc = "~$> ",unit="win") as pbar:
            for window in range(cycle_df.index.min(),cycle_df.index.max(),w_step):
                window_df = cycle_df[w_start:w_end]
                window_df.plot()
                if len(window_df)!=w_size:
                    continue
                window_df = window_df.reset_index(drop=True)
                acc_list = []
                dec_list = []
                counter_P_N_030 = 0
                counter_P_N_3050 = 0
                counter_P_N_5080 = 0
                counter_P_N_80100 = 0
                counter_P_D_12 = 0
                counter_P_D_23 = 0
                for time_step in window_df.index:
                    if window_df[time_step]<=0.30:
                        counter_P_N_030+=1
                    elif 0.30<window_df[time_step] and window_df[time_step]<0.50:
                        counter_P_N_3050+=1
                    elif 0.50<window_df[time_step] and window_df[time_step]<0.80:
                        counter_P_N_5080+=1
                    else:
                        counter_P_N_80100+=1
                    if time_step==0:
                        pass
                    else:
                        acc = window_df[time_step]-window_df[time_step-1]
                        if acc > 0:
                            acc_list.append(acc)
                        else:
                            dec_list.append(acc)
                            if acc<-0.05:
                                counter_P_D_12+=1
                            if -0.05<acc<-0.01:
                                counter_P_D_23+=1
                if len(dec_list) == 0:
                    ave_win_dec = 0
                    max_win_dec = 0
                else:
                    ave_win_dec = stats.mean(dec_list)
                    max_win_dec = min(dec_list)

                if len(acc_list) == 0:
                    ave_win_acc = 0
                    max_win_acc = 0
                    std_win_acc = 0
                elif len(acc_list) == 1:
                    std_win_acc == 0
                else:
                    ave_win_acc = stats.mean(acc_list)
                    max_win_acc = max(acc_list)
                    std_win_acc = stats.stdev(acc_list)

                cycle_MAX_rev.append(window_df.max())
                cycle_AVE_acc.append(ave_win_acc)
                cycle_MAX_acc.append(max_win_acc)
                cycle_STD_acc.append(std_win_acc)
                cycle_AVE_dec.append(ave_win_dec)
                cycle_MAX_dec.append(max_win_dec)
                cycle_P_N_030.append(counter_P_N_030/len(window_df))
                cycle_P_N_3050.append(counter_P_N_3050/len(window_df))
                cycle_P_N_5080.append(counter_P_N_5080/len(window_df))
                cycle_P_N_80100.append(counter_P_N_80100/len(window_df))
                cycle_P_D_12.append(counter_P_D_12/len(window_df))
                cycle_P_D_23.append(counter_P_D_23/len(window_df))
                w_start+=w_step
                w_end+=w_step
                pbar.update(n=1)
        Cycle_Final['N_MAX'] = cycle_MAX_rev
        Cycle_Final['N_AVE'] = cycle_AVE_acc
        Cycle_Final['A_AVE'] = cycle_AVE_acc
        Cycle_Final['A_MAX'] = cycle_MAX_acc
        Cycle_Final['A_STD'] = cycle_STD_acc
        Cycle_Final['D_AVE'] = cycle_AVE_dec
        Cycle_Final['D_MAX'] = cycle_MAX_dec

        Cycle_Final['P_N_030'] = cycle_P_N_030
        Cycle_Final['P_N_3050'] = cycle_P_N_3050
        Cycle_Final['P_N_5080'] = cycle_P_N_5080
        Cycle_Final['P_N_80100'] = cycle_P_N_80100

        Cycle_Final['P_D_12'] = cycle_P_D_12
        Cycle_Final['P_D_23'] = cycle_P_D_23



        if not os.path.exists(__model_path):
            os.makedirs(__model_path)
        Cycle_Final.to_csv(__model_path+"/"+csv_save_name+"_"+str(cycle)+".csv",index=False)
            
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
