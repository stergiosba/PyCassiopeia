import os

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

from include.network import network
import include.network.net_constants as netco
import include.network.net_setup as nets
from include.processes import trendProcess, cycleProcess

def trend_windows_execute(window_size,window_step):
    measurements = ['RE_HMI_ECUA_020115']#"PR_air_scav","RE_gov_idx_meas"]
    features_list = netco.TREND_FEATURES
    if window_size>=window_size:
        model_path = os.getcwd()+"/models/"+netco.TREND+"/model_"+str(window_size)+"_"+str(window_step)
        if not os.path.exists(model_path):
            print("~$> Creating Model")
            print("~$> Model Window Size:",window_size)
            print("~$> Model Window Step:",window_step)
            data_path = os.getcwd()+"/data"
            window_settings = [window_size,window_step]
            trendProcess(data_path,model_path,features_list,measurements,window_settings)
        else:
            print("~$> The Model already exists.")
            fit_df = pd.read_csv(model_path+"/train.csv",usecols=features_list,engine='python')
            fit_df.plot()
            plt.show()
    else:
        print("Cant Have smaller size than step")

def cycles_windows_execute(window_size,window_step):
    features_list = netco.CYCLES_FEATURES
    if window_size>=window_size:
        model_path = os.getcwd()+"/models/"+netco.CYCLES+"/model_"+str(window_size)+"_"+str(window_step)
        if not os.path.exists(model_path):
            print("~$> Creating Model")
            print("~$> Model Window Size:",window_size)
            print("~$> Model Window Step:",window_step)
            data_path = os.getcwd()
            window_settings = [window_size,window_step]
            cycleProcess(data_path,model_path,features_list,window_settings)
        else:
            print("~$> The Model already exists.")
            fit_df = pd.read_csv(model_path+"/train.csv",usecols=features_list,engine='python')
            fit_df.plot()
            plt.show()
    else:
        print("Cant Have smaller size than step")