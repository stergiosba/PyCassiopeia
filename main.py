from include import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
import os
plt.style.use('ggplot')

measurements = ['RE_HMI_ECUA_020115']#"PR_air_scav","RE_gov_idx_meas"]
#window_settings = [500,20] #Window Size, Window Step
features_list = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']

window_sizes = [20,40,60,120,200,600,1200]
window_steps = [2,5,10,20,60,100,200]

window_sizes = [180]
window_steps = [10]
for i in window_sizes:
    for j in window_steps:
        window_settings = []
        if i>=j:
            window_settings.append(i)
            window_settings.append(j)
            print(window_settings)
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
            
            NN_DC = Network(np.array([[120,6],[100,120],[3,100]]))
            final_predictions = NN_DC.train(fit_df,learning_rate=0.0001,epochs=3,minibatch_size=16)
            fit_df['PRED_LABEL'] = final_predictions
            fit_df = fit_df.tail(1500)
            fit_df = fit_df.reset_index(drop=True)
            fig = plt.figure()
            plot_feat = 'N_AVE'
            #ax = fit_df['N_MAX'].plot()
            ax = fit_df[plot_feat].plot()
            for i in range(len(fit_df)):
                y = fit_df[plot_feat][i]
                if (fit_df['LABEL'][i] == fit_df['PRED_LABEL'][i]):
                    ax.text(i, y, str(fit_df['PRED_LABEL'][i]),bbox=dict(facecolor='green', alpha=0.5))
                else:
                    ax.text(i, y, str(fit_df['PRED_LABEL'][i]),bbox=dict(facecolor='red', alpha=0.5))
            

            """
            X_data = fit_df.drop(["LABEL"],axis=1)
            X = X_data.values
            Y_data = fit_df.iloc[:,:1]
            Y = Y_data.values
            Y = np.array([labelMaker(i[0]) for i in Y])
            
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,shuffle=False)# random_state=0)
            X_train = X_train.transpose()
            X_test = X_test.transpose()
            Y_train = Y_train.transpose()
            Y_test = Y_test.transpose()
            
            num__input_features = X_train.shape[0]
            num_output_features = Y_train.shape[0]
            
            (parameters,train_acc,test_acc,predictions) = CAS_NN(X_data.transpose(),X_train, Y_train, X_test, Y_test, minibatch_size = 32)
            
            
            exit_path = exitSettings(model_path, window_settings, X_data.shape[0], train_acc, test_acc)
            """
            
            
            