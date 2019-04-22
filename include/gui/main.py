import os
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow import set_random_seed
#import matplotlib.animation as animation

from include.dataProcess import trendProcess, cycleProcess
from include.network import network
import include.network.net_constants as netco
import include.network.net_setup as nets

measurements = ['RE_HMI_ECUA_020115']#"PR_air_scav","RE_gov_idx_meas"]
#window_settings = [500,20] #Window Size, Window Step
features_list = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']

def trend_windows_execute(window_size,window_step):
    features_list = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']
    if window_size>=window_size:
        model_path = os.getcwd()+"/models/"+netco.TREND+"/model"+str(window_size)+"_"+str(window_step)
        if not os.path.exists(model_path):
            print("~$> Creating Model")
            print("~$> Model Window Size:",window_size)
            print("~$> Model Window Step:",window_step)
            data_path = os.getcwd()+"/data"
            window_settings = [window_size,window_step]
            (fit_df,full_df)= trendProcess(data_path,model_path,features_list,measurements,window_settings)
    else:
        print("Cant Have smaller size than step")

def cycles_windows_execute(window_size,window_step):
    features_list = ['LABEL','N_MAX','N_MIN','N_AVE','A_MAX','A_AVE','A_STD','D_AVE','ADS','P_N_030','P_N_3050','P_N_70','P_D_12','P_D_23']
    if window_size>=window_size:
        model_path = os.getcwd()+"/models/"+netco.CYCLES+"/model"+str(window_size)+"_"+str(window_step)
        if not os.path.exists(model_path):
            print("~$> Creating Model")
            print("~$> Model Window Size:",window_size)
            print("~$> Model Window Step:",window_step)
            data_path = os.getcwd()
            window_settings = [window_size,window_step]
            fit_df = cycleProcess(data_path,model_path,features_list,window_settings)
    else:
        print("Cant Have smaller size than step")



'''
def execute(window_size,window_step):
    if window_size>=window_size:
        model_path = os.getcwd()+"/models/model"+str(window_size)+"_"+str(window_step)
        if not os.path.exists(model_path):
            print("~$> Creating Model")
            print("~$> Model Window Size:",window_size)
            print("~$> Model Window Step:",window_step)
            data_path = os.getcwd()+"/data"
            window_settings = [window_size,window_step]
            (fit_df,full_df)= dataProcess(data_path,model_path,features_list,measurements,window_settings)
        else:
            print("~$> Loading Model")
            print("~$> Model Window Size:",window_size)
            print("~$> Model Window Step:",window_step)
            fit_df = pd.read_csv(model_path+"/train_data.csv",usecols=features_list)
        class_names = ['Deceleration', 'Acceleration', 'Steady']
        class_names_labels = ['D', 'A', 'S']
        #graph1 = tf.Graph()
        layers_structure = np.array([[120,6],[100,120],[3,100]])
        NN_DC = network.Network("NN_DC", root_path=model_path, structure=layers_structure, net_graph=tf.Graph())
        fit_df['PRED_LABEL'] = NN_DC.train(fit_df,learning_rate=0.0001,epochs=1,minibatch_size=32)
        #NN_DC.save(save_path)
        fit_df = fit_df.head(100)
        fit_df = fit_df.reset_index(drop=True)
        fig = plt.figure(3)
        plot_feat = 'N_AVE'
        #ax = fit_df['N_MAX'].plot()
        ax = fit_df[plot_feat].plot()
        for i in range(len(fit_df)):
            y = fit_df[plot_feat][i]
            label = class_names_labels[int(fit_df['LABEL'][i])]+"/"+class_names_labels[fit_df['PRED_LABEL'][i]] + " ? "+str(round(fit_df['A_AVE'][i],2))
            if (fit_df['LABEL'][i] == fit_df['PRED_LABEL'][i]):
                ax.text(i, y, label,bbox=dict(facecolor='green', alpha=0.5))
            else:
                ax.text(i, y, label,bbox=dict(facecolor='red', alpha=0.5))
        #plt.savefig('foo.png')
        #NN_DC.show_data(ncon.TRAINING)
        NN_DC.show_data(ncon.TESTING)
        plt.show()
        subprocess.run(["tensorboard", "--logdir="+NN_DC.root_path+"/train"])
        NN_DC.inference()
    else:
        print("Cant Have smaller size than step")
'''
                