#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:34 2019

@author: stergios
"""
import os

import pandas as pd
from numpy.random import seed

from .utils import cleanTrendData
from .windows import trendWindow, cycleWindow

seed(1)

def trendProcess(data_path,model_path,features_list,measurements,window_settings):
    print(50*"#")
    print("~$> Initializing Data Processing")
    print(50*"#")
    '''
    #indeces = [9,14,32,69]
    indeces = [9,14,33,92,15,73,7,66,93,76,91,69]
    #indeces = [9,69]
    full_df = cleanTrendData(data_path,measurements,indeces)
    full_df.loc[:,measurements[0]]
    '''
    full_df = pd.read_csv(data_path+'/swap_corrected_templates_soft_dtw_clusters7_gamma1_3.csv')
    print("~$> All missing datapoints have been restored")
    print("~$> Loading the dataset from " +data_path)
    print("~$> Collected data for",full_df.shape[0],"seconds.")
    #trendWindow(full_df, features_list, measurements, window_settings, model_path)
    trendWindow(full_df, features_list, window_settings, model_path)

def cycleProcess(data_path,model_path,features_list,window_settings):
    print(50*"#")
    print("~$> Initializing Data Processing")
    print(50*"#")
    print("~$> Loading the dataset from " + data_path)
    full_df = pd.read_csv(data_path+'/swap_corrected_templates_soft_dtw_clusters7_gamma1_3.csv')
    print(full_df)
    print("~$> Loaded the dataset")
    cycleWindow(full_df, features_list, window_settings, model_path)
    

