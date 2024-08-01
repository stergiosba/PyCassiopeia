#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019

@author: stergios
"""

import os
import time
import statistics as stats

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import windowUnits, EPS, ACC_THRESHOLD
import include.network.net_constants as netco


# [DEPRECATED]
def cycleWindow(_data_df, features_list, window_settings, model_path):
    begin = time.time()

    # exit setting
    w_size = window_settings[0]
    w_step = window_settings[1]
    print(50 * "-")
    print("~$> Initializing Window Making Processing for Engine Cycles")
    print(50 * "-")
    print("~$> Window size", w_size, "seconds.")
    print("~$> Window step", w_step, "seconds.")
    print(50 * "-")
    fit_df = pd.DataFrame(columns=features_list)
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(_data_df) - 1, w_size - 1, w_step)

    print("~$> Total Windows Progression")
    for cycle in _data_df:
        cycle_df = _data_df[cycle]
        w_start = 0
        w_end = w_size
        Cycle_Final = pd.DataFrame()
        with tqdm(total=w_count, desc="~$> ", unit="win") as pbar:
            for window in range(cycle_df.index.min(), cycle_df.index.max(), w_step):
                window_df = cycle_df[w_start:w_end]
                if len(window_df) != w_size:
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
                    if window_df[time_step] <= 0.30:
                        counter_P_N_030 += 1
                    elif 0.30 < window_df[time_step] and window_df[time_step] < 0.50:
                        counter_P_N_3050 += 1
                    elif 0.50 < window_df[time_step] and window_df[time_step] < 0.70:
                        counter_P_N_5070 += 1
                    else:
                        counter_P_N_70100 += 1
                    if time_step == 0:
                        pass
                    else:
                        acc = window_df[time_step] - window_df[time_step - 1]
                        if acc > 0:
                            acc_list.append(acc)
                            if 0 < acc < 0.003:
                                counter_P_A_1 += 1
                            elif 0.003:
                                counter_P_A_2 += 1
                        else:
                            dec_list.append(acc)
                            if -0.005 < acc < 0:
                                counter_P_D_1 += 1
                            else:
                                counter_P_D_2 += 1
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

                Cycle_Final = Cycle_Final.append(
                    {
                        "LABEL": cycle,
                        "N_MAX": round(window_df.max(), 4),
                        "N_AVE": round(window_df.mean(), 4),
                        "A_MAX": round(max_win_acc, 4),
                        "A_AVE": round(ave_win_acc, 4),
                        "A_STD": round(std_win_acc, 4),
                        "D_MAX": round(max_win_dec, 4),
                        "D_AVE": round(ave_win_dec, 4),
                        "P_N_030": round(counter_P_N_030 / len(window_df), 4),
                        "P_N_3050": round(counter_P_N_3050 / len(window_df), 4),
                        "P_N_5070": round(counter_P_N_5070 / len(window_df), 4),
                        "P_N_70100": round(counter_P_N_70100 / len(window_df), 4),
                        "P_D_1": round(counter_P_D_1 / len(window_df), 4),
                        "P_D_2": round(counter_P_D_2 / len(window_df), 4),
                        "P_A_1": round(counter_P_A_1 / len(window_df), 4),
                        "P_A_2": round(counter_P_A_2 / len(window_df), 4),
                    },
                    ignore_index=True,
                )
                prev_cycle = cycle
                w_start += w_step
                w_end += w_step
                pbar.update(n=1)
            Cycle_Final = Cycle_Final.astype({"LABEL": int})
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            Cycle_Final.to_csv(
                os.path.join(model_path, netco.INFERENCE + "_" + str(cycle) + ".csv"),
                index=False,
            )
        fit_df = fit_df.append(Cycle_Final, sort=False, ignore_index=True)

    print(50 * "-")
    print("~$> Plotting Pearson Correlation Matrix")

    correlations = fit_df[fit_df.columns].corr(method="pearson")
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot=True)
    # plt.show()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    fit_df.to_csv(model_path + "/" + netco.TRAINING + ".csv", index=False)

    finish = time.time()
    print(50 * "-")
    print("~$> Time for data process was", round(finish - begin, 2), "seconds.")
    print(50 * "-")


"""
def trendWindow2(_data_df,features_list,measurements,window_settings,model_path):
    begin = time.time()

    #exit setting
    w_size = window_settings[0]
    w_step = window_settings[1]
    print(50*"-")
    print("~$> Initializing Window Making Processing for Speed Trend Prediction")
    print(50*"-")
    print("~$> Window size",w_size,"seconds.")
    print("~$> Window step",w_step,"seconds.")
    print(50*"-")
    
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(_data_df),w_size,w_step)
    fit_df = pd.DataFrame(columns=features_list)
    w_start = 0
    w_end = w_size
    print("~$> Total Windows Progression")
    with tqdm(total = w_count,desc = "~$> ",unit="win") as pbar:
        for window in range(_data_df.index.min(),_data_df.index.max(),w_step):
            window_df = _data_df[w_start:w_end]
            if len(window_df)!=w_size:
                continue
            window_df = window_df.reset_index(drop=True)
            win_accs = []
            for time_step in window_df.index:
                if time_step==0:
                    pass
                else:
                    print(window_df)
                    acc = window_df[time_step]-window_df[time_step-1]
                    print(window_df)
                    #acc = window_df[measurements[0]][time_step]-window_df[measurements[0]][time_step-1]
                    win_accs.append(acc)
            
            ave_win_acc = round(stats.mean(win_accs),4)
            max_win_revs = round(window_df[measurements[0]].max(),4)
            min_win_revs = round(window_df[measurements[0]].min(),4)
            ave_win_revs = round(window_df[measurements[0]].mean(),4)
            in_win_revs = round(window_df[measurements[0]][window_df.index.min()],4)
            out_win_revs = round(window_df[measurements[0]][window_df.index.max()],4)
            
            ave_win_acc = round(stats.mean(win_accs),4)
            max_win_revs = round(window_df.max(),4)
            min_win_revs = round(window_df.min(),4)
            ave_win_revs = round(window_df.mean(),4)
            in_win_revs = round(window_df[window_df.index.min()],4)
            out_win_revs = round(window_df[window_df.index.max()],4)
            if w_start == 0:
                label = 1 #Starting with Steady
                prev_label = label
            else:
                if (ave_win_revs==0 and max_win_revs==0 and min_win_revs==0):
                    label = 0 #Dead Stop
                elif (ave_win_revs<0.5 and ave_win_acc<ACC_THRESHOLD and ave_win_acc>-ACC_THRESHOLD):
                    label = 1 #Low Speed Steady
                elif (ave_win_revs>0.5 and ave_win_acc<ACC_THRESHOLD and ave_win_acc>-ACC_THRESHOLD):
                    label = 2 #High Speed Steady
                elif (ave_win_acc>=ACC_THRESHOLD):
                    label = 3 #Acceleration
                elif (ave_win_acc<=-ACC_THRESHOLD):
                    label = 4 #Deceleration
            fit_df = fit_df.append({
                'LABEL': prev_label,
                'N_MAX': max_win_revs,
                'N_MIN': min_win_revs,
                'N_AVE': ave_win_revs,
                'N_IN' : in_win_revs,
                'N_OUT': out_win_revs,
                'A_AVE': ave_win_acc,
                },ignore_index=True)
            w_start+=w_step
            w_end+=w_step
            prev_label = label
            pbar.update(n=1)  
    print(50*"-")    
    print("~$> Plotting Pearson Correlation Matrix")
    correlations = fit_df[fit_df.columns].corr(method='pearson')
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot = True)
    #plt.show(block=False)
    if not os.path.exists(model_path): os.makedirs(model_path)
    fit_df.to_csv(model_path+"/"+netco.TRAINING+".csv",index=False)

    finish = time.time()
    print(50*"-")
    print("~$> Time for data process was",round(finish-begin,2),"seconds.")
    print(50*"-")
"""


def trendWindow(_data_df, features_list, window_settings, model_path):
    begin = time.time()
    # exit setting
    w_size = window_settings[0]
    w_step = window_settings[1]
    print(50 * "-")
    print("~$> Initializing Window Making Processing for Speed Trend Prediction")
    print(50 * "-")
    print("~$> Window size", w_size, "seconds.")
    print("~$> Window step", w_step, "seconds.")
    print(50 * "-")
    fit_df = pd.DataFrame(columns=features_list)
    # [Finding maximum count of correct length windows]
    w_count = windowUnits(len(_data_df) - 1, w_size - 1, w_step)

    print("~$> Total Windows Progression")
    counter = 0
    for cycle in _data_df:
        cycle_df = _data_df[cycle]
        w_start = 0
        w_end = w_size
        Cycle_Final = pd.DataFrame(columns=features_list)
        ins = 0
        with tqdm(total=w_count, desc="~$> ", unit="win") as pbar:
            for window in range(cycle_df.index.min(), cycle_df.index.max(), w_step):
                window_df = cycle_df[w_start:w_end]
                if len(window_df) != w_size:
                    continue
                # window_df = window_df.reset_index(drop=True)
                # Checking for values below EPS and making them zero.

                window_df = window_df.apply(lambda x: x if x > EPS else 0)
                # Initializing the counters
                acc_list = []
                for time_step in window_df.index:
                    if time_step == 0 + w_start:
                        pass
                    else:
                        acc = window_df[time_step] - window_df[time_step - 1]
                        acc_list.append(acc)
                ave_win_acc = round(stats.mean(acc_list), 6)
                max_win_revs = round(window_df.max(), 4)
                min_win_revs = round(window_df.min(), 4)
                ave_win_revs = round(window_df.mean(), 4)
                in_win_revs = round(window_df[window_df.index.min()], 4)
                out_win_revs = round(window_df[window_df.index.max()], 4)
                if w_start == 0:
                    if ave_win_revs > EPS and ave_win_revs < 0.3:
                        label = 1  # Low Speed Steady
                    elif ave_win_revs > 0.3 and ave_win_revs < 0.6:
                        label = 2  # Mid Speed Steady
                    elif ave_win_revs > 0.6:
                        label = 3  # High Speed Steady
                    prev_label = label
                else:
                    if ave_win_revs < EPS and max_win_revs < EPS and min_win_revs < EPS:
                        label = 0  # Dead Stop
                    elif (
                        ave_win_revs > EPS
                        and ave_win_revs < 0.3
                        and ave_win_acc < ACC_THRESHOLD
                        and ave_win_acc > -ACC_THRESHOLD
                    ):
                        label = 1  # Low Speed Steady
                    elif (
                        ave_win_revs > 0.3
                        and ave_win_revs < 0.6
                        and ave_win_acc < ACC_THRESHOLD
                        and ave_win_acc > -ACC_THRESHOLD
                    ):
                        label = 2  # Mid Speed Steady
                    elif (
                        ave_win_revs > 0.6
                        and ave_win_acc < ACC_THRESHOLD
                        and ave_win_acc > -ACC_THRESHOLD
                    ):
                        label = 3  # High Speed Steady
                    elif ave_win_acc >= ACC_THRESHOLD:
                        label = 4  # Acceleration
                    elif ave_win_acc <= -ACC_THRESHOLD:
                        label = 5  # Deceleration

                Cycle_Final = Cycle_Final.append(
                    {
                        "LABEL": prev_label,
                        "N_MAX": max_win_revs,
                        "N_MIN": min_win_revs,
                        "N_AVE": ave_win_revs,
                        "N_IN": in_win_revs,
                        "N_OUT": out_win_revs,
                        "A_AVE": ave_win_acc,
                    },
                    ignore_index=True,
                )
                """
                font = {#'family':'',
                'color':'black',
                'weight':'normal',
                'size': 14
                }
                if ins<=2:
                    plt.xlabel(r'$\mathbf{Time}$ (sec)',fontdict=font)
                    plt.ylabel(r'$\mathbf{\beta}$ / $\mathbf{\beta_{max}}$',fontdict=font)
                    plt.ylim(0,1)

                    #plt.plot(w_start,window_df[window_df.index.min()],'m>')
                    #plt.plot(w_end-1,window_df[window_df.index.max()],'ro')
                    #plt.plot(window_df.idxmax(),window_df.max(),'y<')
                    #plt.plot(window_df.idxmin(),window_df.min(),'gx')
                    if ins==0:
                        plt.text(w_start+10,0.75,'Window '+str(ins+1),bbox=dict(facecolor='red', alpha=0.8))
                        plt.axvline(x=w_start,ymin=0,ymax=0.75,color='red',linestyle='-',linewidth=2)
                        plt.axvline(x=w_end-1,ymin=0,ymax=0.75,color='red',linestyle='-',linewidth=2)
                        plt.annotate('', xy=(w_start,0.7), xytext=(w_end-1,0.7),arrowprops={'arrowstyle': '<->','lw': 3}, va='center')
                        plt.annotate('', xy=(w_end,0.6), xytext=(w_end+w_step-1,0.6),arrowprops={'arrowstyle': '<->','lw': 3,'color':'orange'}, va='center')
                        plt.text(w_end+25,0.65,'Step '+str(ins+1),bbox=dict(facecolor='orange', alpha=0.8))
                    elif ins==1:
                        plt.text(w_start+10,0.85,'Window '+str(ins+1),bbox=dict(facecolor='blue', alpha=0.8))
                        plt.axvline(x=w_start,ymin=0,ymax=0.85,color='blue',linestyle='-',linewidth=2)
                        plt.axvline(x=w_end-1,ymin=0,ymax=0.85,color='blue',linestyle='-',linewidth=2)
                        plt.annotate('', xy=(w_start,0.8), xytext=(w_end-1,0.8),arrowprops={'arrowstyle': '<->','lw': 3}, va='center')
                        plt.annotate('', xy=(w_end,0.6), xytext=(w_end+w_step-1,0.6),arrowprops={'arrowstyle': '<->','lw': 3,'color':'orange'}, va='center')
                        plt.text(w_end+25,0.65,'Step '+str(ins+1),bbox=dict(facecolor='orange', alpha=0.8))
                    else:
                        plt.text(w_start+10,0.95,'Window '+str(ins+1),bbox=dict(facecolor='green', alpha=0.8))
                        plt.axvline(x=w_start,ymin=0,ymax=0.95,color='green',linestyle='-',linewidth=2)
                        plt.axvline(x=w_end-1,ymin=0,ymax=0.95,color='green',linestyle='-',linewidth=2)
                        plt.annotate('', xy=(w_start,0.9), xytext=(w_end-1,0.9),arrowprops={'arrowstyle': '<->','lw': 3}, va='center')
                    
                ax=window_df.plot(color=(0/255,100/255,200/255))
                """
                prev_label = label
                w_start += w_step
                w_end += w_step
                pbar.update(n=1)
                # ins+=1
            Cycle_Final = Cycle_Final.astype({"LABEL": int})
            # ax.grid()
            counter += 1
            # major_ticks = np.arange(0, 600, 50)
            # minor_ticks = np.arange(0, 101, 5)
            # ax.set_xticks(major_ticks)
            # plt.show()
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            # plt.savefig(os.path.join(model_path,'trained_model_'+str(counter)+'.png'),dpi=800)
            Cycle_Final.to_csv(
                os.path.join(model_path, netco.INFERENCE + "_" + str(cycle) + ".csv"),
                index=False,
            )
        fit_df = fit_df.append(Cycle_Final, sort=False, ignore_index=True)

    print(50 * "-")
    print("~$> Plotting Pearson Correlation Matrix")

    correlations = fit_df[fit_df.columns].corr(method="pearson")
    heat_ax = sns.heatmap(correlations, cmap="YlGnBu", annot=True)
    # plt.show()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    fit_df.to_csv(model_path + "/" + netco.TRAINING + ".csv", index=False)
    finish = time.time()
    print(50 * "-")
    print("~$> Time for data process was", round(finish - begin, 2), "seconds.")
    print(50 * "-")
