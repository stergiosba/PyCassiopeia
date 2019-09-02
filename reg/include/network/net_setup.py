#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019
@author: Khanax
"""
import math
import time
import statistics as stats

import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils import EPS,windowUnits

'''
def labelMaker(class_range,val):
    ret = []
    for i in range(class_range):
        if i == val:
            ret.append(1)
        else:
            ret.append(0)
    return ret
'''

def labelMaker(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
        
# n_x = num__input_features
# n_y = expected output (num classes)
def create_placeholders(n_x, n_y):
    X1 = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
    Y1 = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")
    return X1, Y1

def network_cost_function(final_layer, Y, loss):
    """
    Computes the cost in softmax cross entropy or RMSE or MSE fashion
    
    Arguments:
    final_layer -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as final_layer
    
    Returns:
    cost - Tensor of the cost function
    """
    if loss == 'softmax_cross_entropy' or loss=='SCE':
        logits = tf.transpose(final_layer)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    elif loss == 'reduce_mean_square' or loss=='RMSE':
        cost = tf.reduce_mean(tf.square(final_layer - Y))
    elif loss == 'mean_squared_error' or loss=='MSE':
        cost = tf.losses.mean_squared_error(Y,final_layer)
    
    return cost
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]# number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size)# number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

def cycleInference(_data_df,_features_list,_window_settings,__model_path,_csv_flag = True):
    '''TO REMOVE'''
    begin = time.time()
    print(_window_settings)
    w_size = int(_window_settings[0])
    w_step = int(_window_settings[1])
    print(50*"-")
    print("~$> Initializing Window Making Processing for Engine Cycles")
    print(50*"-")
    print("~$> Window size",w_size,"seconds.")
    print("~$> Window step",w_step,"seconds.")
    print(50*"-")
    visual_df = pd.DataFrame(columns=_features_list)

    w_start = 0
    w_end = w_size

    for window in range(_data_df.index.min(),_data_df.index.max(),w_step):
        window_df = _data_df[w_start:w_end]
        window_df = window_df.loc[:,'E_REV']
        if len(window_df)!=w_size:
            continue
        window_df = window_df.reset_index(drop=True)
        window_df = window_df.apply(lambda x: x if x > EPS else 0)
        acc_list = []
        dec_list = []
        counter_P_N_030 = 0
        counter_P_N_3050 = 0
        counter_P_N_5070 = 0
        counter_P_N_70100 = 0
        counter_P_D_12 = 0
        counter_P_D_23 = 0
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
            std_win_acc = 0
        else:
            ave_win_acc = stats.mean(acc_list)
            max_win_acc = max(acc_list)
            std_win_acc = stats.stdev(acc_list)
        visual_df = visual_df.append({
        'LABEL': -1881,
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
        'P_N_70100':round(counter_P_N_70100/len(window_df),4)
        #'P_D_12':1,
        #'P_D_23':1
        },ignore_index=True)
        
        w_start+=w_step
        w_end+=w_step
    visual_df = visual_df.astype({'LABEL': int})

    print(50*"-")
    return visual_df