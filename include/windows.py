#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""

import os
import statistics as stats

import pandas as pd

from .utils import windowUnits,EPS,ACC_THRESHOLD
import include.network.net_constants as netco

def patternWindow(cycle_df, model_path, features_list, window_settings):
    w_size = window_settings[0]
    w_step = window_settings[1]
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(cycle_df)-1,w_size-1,w_step)

    w_start = 0
    w_end = w_size
    Cycle_Final = pd.DataFrame()
    for window in range(cycle_df.index.min(),cycle_df.index.max(),w_step):
        window_df = cycle_df[w_start:w_end]
        if len(window_df)!=w_size:
            continue
        window_df = window_df.reset_index(drop=True)
        # Checking for values below EPS and making them zero.
        window_df = window_df.apply(lambda x: x if x > EPS else 0)
        # Initializing the counters
        acc_list = []
        dec_list = []
        counter_P_N_030 = 0
        counter_P_N_3050 = 0
        counter_P_N_5070 = 0
        counter_P_N_70100 = 0
        counter_P_D_1 = 0
        counter_P_D_2 = 0
        counter_P_A_1 = 0
        counter_P_A_2 = 0
        for time_step in window_df.index:
            if window_df[time_step]<=0.30:
                counter_P_N_030+=1
            elif 0.30<window_df[time_step] and window_df[time_step]<0.50:
                counter_P_N_3050+=1
            elif 0.50<window_df[time_step] and window_df[time_step]<0.70:
                counter_P_N_5070+=1
            else:
                counter_P_N_70100+=1
            if time_step==0:
                pass
            else:
                acc = window_df[time_step]-window_df[time_step-1]
                if acc > 0:
                    acc_list.append(acc)
                    if 0<acc<0.003:
                        counter_P_A_1+=1
                    elif 0.003:
                        counter_P_A_2+=1
                else:
                    dec_list.append(acc)
                    if -0.005<acc<0:
                        counter_P_D_1+=1
                    else:
                        counter_P_D_2+=1
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
            std_win_acc = 0
        else:
            ave_win_acc = stats.mean(acc_list)
            max_win_acc = max(acc_list)
            std_win_acc = stats.stdev(acc_list)

        Cycle_Final = Cycle_Final.append({
        'LABEL': cycle_df.name,
        'N_MAX': round(window_df.max(),4),
        'N_AVE': round(window_df.mean(),4),
        'A_MAX': round(max_win_acc,4),
        'A_AVE': round(ave_win_acc,4),
        'A_STD': round(std_win_acc,4),
        'D_MAX': round(max_win_dec,4),
        'D_AVE': round(ave_win_dec,4),
        'P_N_030': round(counter_P_N_030/len(window_df),4),
        'P_N_3050': round(counter_P_N_3050/len(window_df),4),
        'P_N_5070': round(counter_P_N_5070/len(window_df),4),
        'P_N_70100':round(counter_P_N_70100/len(window_df),4),
        'P_D_1':round(counter_P_D_1/len(window_df),4),
        'P_D_2':round(counter_P_D_2/len(window_df),4),
        'P_A_1':round(counter_P_A_1/len(window_df),4),
        'P_A_2':round(counter_P_A_2/len(window_df),4)
        },ignore_index=True)
        prev_cycle = cycle_df.name
        w_start+=w_step
        w_end+=w_step
    Cycle_Final = Cycle_Final.astype({'LABEL': int})
    if not os.path.exists(model_path): os.makedirs(model_path)
    Cycle_Final.to_csv(os.path.join(model_path,netco.INFERENCE+"_"+str(cycle_df.name)+".csv"),index=False)
    return Cycle_Final

def trendWindow(cycle_df, model_path, features_list, window_settings):
    w_size = window_settings[0]
    w_step = window_settings[1]
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(cycle_df)-1,w_size-1,w_step)
    w_start = 0
    w_end = w_size
    Cycle_Final = pd.DataFrame(columns=features_list)
    
    for window in range(cycle_df.index.min(),cycle_df.index.max(),w_step):
        window_df = cycle_df[w_start:w_end]
        if len(window_df)!=w_size:
            continue
        # Checking for values below EPS and making them zero.
        
        window_df = window_df.apply(lambda x: x if x > EPS else 0)
        # Initializing the counters
        acc_list = []
        for time_step in window_df.index:
            if time_step==0+w_start:
                pass
            else:
                acc = window_df[time_step]-window_df[time_step-1]
                acc_list.append(acc)
        ave_win_acc = round(stats.mean(acc_list),6)
        max_win_revs = round(window_df.max(),4)
        min_win_revs = round(window_df.min(),4)
        ave_win_revs = round(window_df.mean(),4)
        in_win_revs = round(window_df[window_df.index.min()],4)
        out_win_revs = round(window_df[window_df.index.max()],4)
        if w_start == 0:                    
            if (ave_win_revs>EPS and ave_win_revs<0.3):
                label = 1 #Low Speed Steady
            elif (ave_win_revs>0.3 and ave_win_revs<0.6):
                label = 2 #Mid Speed Steady
            elif (ave_win_revs>0.6):
                label = 3 #High Speed Steady
            prev_label = label
        else:
            if (ave_win_revs<EPS and max_win_revs<EPS and min_win_revs<EPS):
                label = 0 #Dead Stop
            elif (ave_win_revs>EPS and ave_win_revs<0.3 and ave_win_acc<ACC_THRESHOLD and ave_win_acc>-ACC_THRESHOLD):
                label = 1 #Low Speed Steady
            elif (ave_win_revs>0.3 and ave_win_revs<0.6 and ave_win_acc<ACC_THRESHOLD and ave_win_acc>-ACC_THRESHOLD):
                label = 2 #Mid Speed Steady
            elif (ave_win_revs>0.6 and ave_win_acc<ACC_THRESHOLD and ave_win_acc>-ACC_THRESHOLD):
                label = 3 #High Speed Steady
            elif (ave_win_acc>=ACC_THRESHOLD):
                label = 4 #Acceleration
            elif (ave_win_acc<=-ACC_THRESHOLD):
                label = 5 #Deceleration

        Cycle_Final = Cycle_Final.append({
            'LABEL': prev_label,
            'N_MAX': max_win_revs,
            'N_MIN': min_win_revs,
            'N_AVE': ave_win_revs,
            'N_IN' : in_win_revs,
            'N_OUT': out_win_revs,
            'A_AVE': ave_win_acc
        },ignore_index=True)
        prev_label = label
        w_start+=w_step
        w_end+=w_step
    Cycle_Final = Cycle_Final.astype({'LABEL': int})
    if not os.path.exists(model_path): os.makedirs(model_path)
    Cycle_Final.to_csv(os.path.join(model_path,netco.INFERENCE+"_"+cycle_df.name+".csv"),index=False)
    return Cycle_Final