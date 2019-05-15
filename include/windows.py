#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""

import os
import time
import statistics as stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .utils import windowUnits,EPS
import include.network.net_constants as netco


def cycleWindow(_data_df,_features_list,_window_settings,__model_path,_csv_flag = True):
    begin = time.time()

    #exit setting
    w_size = _window_settings[0]
    w_step = _window_settings[1]
    print(50*"-")
    print("~$> Initializing Window Making Processing for Engine Cycles")
    print(50*"-")
    print("~$> Window size",w_size,"seconds.")
    print("~$> Window step",w_step,"seconds.")
    print(50*"-")
    train_df = pd.DataFrame(columns=_features_list)
    test_df = pd.DataFrame(columns=_features_list)

    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(_data_df),w_size,w_step)

    print("~$> Total Windows Progression")
    for cycle in _data_df:
        cycle_df = _data_df[cycle]
        w_start = 0
        w_end = w_size
        Cycle_Final = pd.DataFrame()
        with tqdm(total = w_count,desc = "~$> ",unit="win") as pbar:
            for window in range(cycle_df.index.min(),cycle_df.index.max(),w_step):
                window_df = cycle_df[w_start:w_end]
                if len(window_df)!=w_size:
                    continue
                window_df = window_df.reset_index(drop=True)
                print(window_df)
                for i in window_df.index:
                    if window_df[i]<EPS:
                        window_df[i] = 0
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
                Cycle_Final = Cycle_Final.append({
                'LABEL': cycle,
                'N_MAX': round(window_df.max(),4),
                'A_MAX': round(max_win_acc,4),
                'A_AVE': round(ave_win_acc,4),
                'A_STD': round(std_win_acc,4),
                'D_MAX': round(max_win_dec,4),
                'D_AVE': round(ave_win_dec,4),
                'P_N_030': round(counter_P_N_030/len(window_df),4),
                'P_N_3050': round(counter_P_N_3050/len(window_df),4),
                'P_N_5080': round(counter_P_N_5080/len(window_df),4),
                'P_N_80100':round(counter_P_N_80100/len(window_df),4)
                #'P_D_12':1,
                #'P_D_23':1
                },ignore_index=True)
                w_start+=w_step
                w_end+=w_step
                pbar.update(n=1)
            Cycle_Final = Cycle_Final.astype({'LABEL': int})
            Cycle_Final_Train, Cycle_Final_Test = train_test_split(Cycle_Final, test_size=0.3, shuffle=False)
        train_df = train_df.append(Cycle_Final_Train,sort=False,ignore_index=True)
        test_df = test_df.append(Cycle_Final_Test,sort=False,ignore_index=True)
        frames = [train_df,test_df]
        fit_df = pd.concat(frames, ignore_index=True)

    print(50*"-")    
    print("~$> Plotting Pearson Correlation Matrix")

    correlations = fit_df[fit_df.columns].corr(method='pearson')
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)

    if not os.path.exists(__model_path): os.makedirs(__model_path)

    fit_df.to_csv(__model_path+"/"+netco.TRAINING+".csv",index=False)

def trendWindow(_data_df,_fit_df,__measurements,__window_settings,__model_path,_csv_flag = True):
    begin = time.time()
    csv_save_name = 'train_data.csv'
    #exit setting
    w_size = __window_settings[0]
    w_step = __window_settings[1]
    #settings_path = os.path.exists(__model_path+"/"+csv_save_name)
    print(50*"-")
    print("~$> Initializing Window Making Processing for Speed Trend Prediction")
    print(50*"-")
    print("~$> Window size",w_size,"seconds.")
    print("~$> Window step",w_step,"seconds.")
    print(50*"-")
    
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(_data_df),w_size,w_step)
     
    w_start = 0
    w_end = w_size
    print("~$> Total Windows Progression")
    with tqdm(total = w_count,desc = "~$> ",unit="win") as pbar:
        for window in range(_data_df.index.min(),_data_df.index.max(),w_step):
            window_df = _data_df[w_start:w_end]
            if len(window_df)!=w_size:
                continue
            window_df = window_df.reset_index(drop=True)
            acc_diff_list = []
            for time_step in window_df.index:
                if time_step==0:
                    pass
                else:
                    acc_diff = window_df[__measurements[0]][time_step]-window_df[__measurements[0]][time_step-1]
                    acc_diff_list.append(acc_diff)
            ave_win_acc = stats.mean(acc_diff_list)
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
                'N_MAX': round(window_df[__measurements[0]].max(),4),
                'N_MIN': round(window_df[__measurements[0]].min(),4),
                'N_AVE': round(window_df[__measurements[0]].mean(),4),
                'N_IN' : round(window_df[__measurements[0]][window_df.index.min()],4),
                'N_OUT': round(window_df[__measurements[0]][window_df.index.max()],4),
                'A_AVE': round(ave_win_acc,4)
                },ignore_index=True)
            w_start+=w_step
            w_end+=w_step
            prev_ave_win_acc = ave_win_acc
            pbar.update(n=1)  
    print(50*"-")    
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