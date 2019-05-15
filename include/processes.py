#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:34 2019

@author: stergios
"""
import os

import pandas as pd
from numpy.random import seed

from .utils import fixNullDataFrame
from .windows import trendWindow, cycleWindow

seed(1)

def trendProcess(data_path,model_path,features_list,measurements,window_settings):
    print(50*"-")
    print("~$> Initializing Data Processing")
    print(50*"-")
    full_df = pd.DataFrame()
    files = pd.DataFrame(sorted(os.listdir(data_path)))
    indx = [53,0,14,33,92,15,73,7,52,66,93,76,91,69]
    for item in indx:
        file_df = pd.read_csv(data_path+"/"+files[0][item],usecols=measurements,engine='python')
        file_df = fixNullDataFrame(file_df,measurements)
        full_df = full_df.append(file_df,ignore_index=True)
    full_df.loc[:,measurements[0]]
    print("~$> Loading the dataset from " +data_path)
    fit_df = pd.DataFrame(columns = features_list)
    print("~$> Datas for",full_df.shape[0],"seconds.")
    trendWindow(full_df, fit_df, measurements, window_settings, model_path)

def cycleProcess(data_path,model_path,features_list,window_settings):
    print(50*"-")
    print("~$> Initializing Data Processing")
    print(50*"-")
    print("~$> Loading the dataset from " + data_path)
    full_df = pd.read_csv(data_path+"/templates.csv", header=None, engine="python").T
    full_df = full_df.head(800)
    print("~$> Loaded the dataset")
    cycleWindow(full_df, features_list, window_settings, model_path)
    

