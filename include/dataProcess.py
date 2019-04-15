#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:34 2019

@author: stergios
"""
import os
import pandas as pd
from numpy.random import seed
from .trendWindow import trendWindow
from .cycleWindow import cicleWindow
from .utils import nullDf, normDf
seed(1)

def dataProcess(_path,_model_path,_features_list,_measurements,_window_settings):
    print(50*"-")
    print("~$> Initializing Data Processing")
    print(50*"-")
    full_df = pd.DataFrame()
    files = pd.DataFrame(sorted(os.listdir(_path)))
    indx = [53,0,14,33,92,15,73,7,52,66,93,76,91,69]
    indx = [14,10,73,91,33,9,15]
    for item in indx:
        file_df = pd.read_csv(_path+"/"+files[0][item],usecols=_measurements,engine='python')
        file_df = nullDf(file_df,_measurements)
        full_df = full_df.append(file_df,ignore_index=True)
    full_df.loc[:,_measurements[0]] *= 220
    #full_df = full_df[_measurement]
    print("~$> Loading the dataset from " +_path)
    full_df = nullDf(full_df,_measurements)
    full_df.plot()
    fit_df = pd.DataFrame(columns = _features_list)
    print("~$> Datas for",full_df.shape[0],"seconds.")
    fit_df = cicleWindow(full_df, fit_df, _measurements, _window_settings, _model_path)
    
    return (fit_df,full_df)