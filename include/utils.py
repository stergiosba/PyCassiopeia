# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:32:50 2019

@author: stergios
"""
import pandas as pd
from sklearn import preprocessing
#from tensorflow import set_random_seed
import numpy as np
import math

#TODO CLEAN THIS FILE UP AND ADD COMMENTS

def windowUnits(max_length,size,step):
    position = 0
    counter = 0
    for i in range(0,max_length,step):
        if i==0:
            position+=size
        else:
            position+=step
            if position>max_length:
                break
        counter+=1
    return counter

def nullDf(df,_measurements):
    for measurement in _measurements:
        print("~$> Calculated",df[df[measurement].isnull()].size,"missing datapoints.")
        if df[df[measurement].isnull()].size >10000:
            print("~$> Dataframe is missing to many data")
        if 0 in df[df[measurement].isnull()].index:
            df[measurement][0] = 0
        for i in df[df[measurement].isnull()].index:
            df[measurement][i] = df[measurement][i-1]
        print("~$> All missing datapoints have been restored")
    return df
    
def normDf(df):
    print("~$> Normalizing Dataframe")
    d = {}
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    for i in range(len(df.columns)):
        d[df.columns[i]]=x_scaled[:,i]
    print("~$> Dataframe has been normalized")
    return pd.DataFrame(data=d)

