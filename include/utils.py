# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:32:50 2019

@author: stergios
"""
import os
import pandas as pd
from sklearn import preprocessing
import numpy as np
import math

EPS = 1.0e-5
MAX_NULL = 1.0e4
ACC_THRESHOLD = 0.002

# [Mathematical solution for proper window sizes]
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

# [Returns the null-free full dataframe from indeces.]
def cleanTrendData(data_path,measurements,indeces):
    full_df = pd.DataFrame()
    files = pd.DataFrame(sorted(os.listdir(data_path)))
    for index in indeces:
        index_df = pd.read_csv(data_path+"/"+files[0][index],usecols=measurements,engine='python')
        print('~$> Processing Dataframe with index:',index)
        for measurement in measurements:
            print('~$> Calculated',index_df[index_df[measurement].isnull()].size,'missing datapoints.')
            if index_df[index_df[measurement].isnull()].size > MAX_NULL:
                print("~$> Dataframe is missing to many data")
                break
            if 0 in index_df[index_df[measurement].isnull()].index:
                index_df[measurement][0] = 0
            for i in index_df[index_df[measurement].isnull()].index:
                index_df[measurement][i] = index_df[measurement][i-1]
            index_df[index_df < 0] = 0
        full_df = full_df.append(index_df,ignore_index=True)
        print(50*"-")
    return full_df
 
def normalizeDataFrame(df):
    '''
    Normalizing Dataframe Columns except for LABEL column
    '''
    print("~$> Normalizing Dataframe")
    d = {}
    min_max_scaler = preprocessing.MinMaxScaler()
    if 'LABEL' in df.columns :
        saved_labels = df[['LABEL']]
        x_scaled = min_max_scaler.fit_transform(df.values)
        for i in range(len(df.columns)):
            d[df.columns[i]]=x_scaled[:,i]
        print("~$> Dataframe has been normalized")
        final_df = pd.DataFrame(data=d)
        final_df['LABEL'] = saved_labels
    else:
        x_scaled = min_max_scaler.fit_transform(df.values)
        for i in range(len(df.columns)):
            d[df.columns[i]]=x_scaled[:,i]
        print("~$> Dataframe has been normalized")
        final_df = pd.DataFrame(data=d)        
    return final_df

