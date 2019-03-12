#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:34 2019

@author: stergios
"""
import time
import pandas as pd
from numpy.random import seed
from .windowMaker import windowMaker
from .utils import nullClean
seed(1)

def dataProcess(_path,_model_path,_features_list,_measurement,_window_settings):
    begin = time.time()
    print(50*"-")
    print("~$> Initializing Data Processing")
    print(50*"-")
    full_df = pd.read_csv(_path)
    #full_df = full_df[_measurement]
    full_df = (full_df[[_measurement]])/full_df[_measurement].max()
    print("~$> Loading the dataset from " +_path)
    full_df = nullClean(full_df,_measurement)
    full_df.plot()
    print(full_df[full_df[_measurement].isnull()].size)
    print("~$> All missing datapoints have been restored")
    fit_df = pd.DataFrame(columns = _features_list)
    print("~$> Datas for",full_df.shape[0],"seconds.")
    data_df = full_df.head(120000)
    #data_df = full_df.head(full_df.shape[0])
    print(data_df)
    csv_flag = True
    fit_df = windowMaker(data_df, fit_df, _measurement, _window_settings, csv_flag, _model_path)

    finish = time.time()
    print(50*"-")
    print("~$> Time for data process was",round(finish-begin,2),"seconds.")
    print(50*"-")
    return fit_df 