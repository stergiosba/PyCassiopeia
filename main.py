import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from numpy.random import seed
from tensorflow import set_random_seed
#import matplotlib.animation as animation

from include.dataProcess import dataProcess
from include.network import network
import include.network.network_constants as ncon

plt.style.use('ggplot')
measurements = ['RE_HMI_ECUA_020115']#"PR_air_scav","RE_gov_idx_meas"]
#window_settings = [500,20] #Window Size, Window Step
features_list = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']
used_features = ['N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']

window_sizes = [20,40,60,120,200,600,1200]
window_steps = [2,5,10,20,60,100,200]

window_sizes = [60]
window_steps = [10]
def main():
    for i in window_sizes:
        for j in window_steps:
            window_settings = []
            if i>=j:
                window_settings.append(i)
                window_settings.append(j)
                model_path = os.getcwd()+"/models/model"+str(window_settings[0])+"_"+str(window_settings[1])
                if not os.path.exists(model_path):
                    print("~$> Creating Model")
                    print("~$> Model Window Size:",window_settings[0])
                    print("~$> Model Window Step:",window_settings[1])
                    data_dir = os.getcwd()+"/data"
                    (fit_df,full_df)= dataProcess(data_dir,model_path,features_list,measurements,window_settings)
                else:
                    print("~$> Loading Model")
                    print("~$> Model Window Size:",window_settings[0])
                    print("~$> Model Window Step:",window_settings[1])
                    fit_df = pd.read_csv(model_path+"/train_data.csv",usecols=features_list)
                class_names = ['Deceleration', 'Acceleration', 'Steady']
                class_names_labels = ['D', 'A', 'S']
                #graph1 = tf.Graph()
                layers_structure = np.array([[120,6],[100,120],[3,100]])
                NN_DC = network.Network("NN_DC", save_path=model_path, structure=layers_structure, net_graph=tf.Graph())
                fit_df['PRED_LABEL'] = NN_DC.train(fit_df,learning_rate=0.0001,epochs=1,minibatch_size=32)
                #NN_DC.save(save_path)
                fit_df = fit_df.head(100)
                fit_df = fit_df.reset_index(drop=True)
                fig = plt.figure()
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
                plt.show()
                NN_DC.show_data(ncon.TESTING)
                subprocess.run(["tensorboard", "--logdir="+NN_DC.path+"/train"])
                NN_DC.inference()
                


if __name__ == "__main__":
    main()