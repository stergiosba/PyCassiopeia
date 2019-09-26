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

class Point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def show(self):
        print(self.x,"-",self.y)

def LI(p1,p2,x):
    lamda = (p2.y-p1.y)/(p2.x-p1.x)
    return round(p1.y+lamda*(x-p1.x),4)

def normalizeDataFrame(df):
    '''
    Efficient Normalizing Dataframe Columns except for LABEL column if it exists.

    -Normalization at x2 speed compared to df[df.columns] = preprocessing.MinMaxScaler().fit_transform(df[df.columns])
    '''
    d = {}
    scaler = preprocessing.MinMaxScaler()
    if 'LABEL' in df.columns :
        saved_labels = df[['LABEL']]
        df = df.drop('LABEL',axis=1)
        x_scaled = scaler.fit_transform(df.values)
        for i in range(len(df.columns)):
            d[df.columns[i]]=x_scaled[:,i]
        final_df = pd.DataFrame(data=d)
        final_df['LABEL'] = saved_labels
    else:
        x_scaled = scaler.fit_transform(df.values)
        for i in range(len(df.columns)):
            d[df.columns[i]]=x_scaled[:,i]
        final_df = pd.DataFrame(data=d)
    if 'D_AVE' in df.columns and 'D_MAX' in df.columns:
        final_df['D_AVE'] = final_df['D_AVE'].apply(lambda x: 1-x)
        final_df['D_MAX'] = final_df['D_MAX'].apply(lambda x: 1-x)

    normalizers = pd.DataFrame([scaler.data_min_,scaler.data_max_],columns=df.columns)
    #normalizers = normalizers.rename(index={0: "MIN", 1: "MAX", 2: "RANGE"})
    normalizers = normalizers.T
    return final_df,normalizers

