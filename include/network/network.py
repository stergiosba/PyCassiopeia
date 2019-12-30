# -*- coding: utf-8 -*-
"""
Code for the Neural Network Classes.
Contains all calculation features and I/O Controls.

@author: Khanax
"""
import json
import os
import re
import sys
import time
import codecs

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import scipy.io

import include.network.net_constants as netco
import include.network.net_setup as nets
from include.utils import normalizeDataFrame, round_down
from include.network.online import onlineData,onlineData2
#tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create grap

class Network(tf.keras.Sequential):
    '''
        General Network Class
    '''
    def __init__(self,
                edition,
                flag,
                base_name,
                root_path,
                features):
        super(Network, self).__init__()
        self.root_path = root_path
        self.edition = edition
        self.features = features
        
        if flag==netco.CREATE:
            self.base_name = base_name
            # On new network creation we check for previous versions up to 10
            self.version_control()
            print(f"~$> Creating Network: |-> {self.cass_name}")
        elif flag==netco.LOAD:
            self.cass_name = base_name
            print(f"~$> Loading Network: |-> {self.cass_name}")
            self.cli_name = "~$/"+self.cass_name+">"
            self.version_path = os.path.join(self.root_path,self.cass_name)
            self.version = self.cass_name.split("_")[-1]
        
    def version_control(self):
        versions_dir = []
        if not os.path.exists(self.root_path): os.makedirs(self.root_path)
        for filename in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,filename)):
                if re.match(re.escape(self.base_name),filename):
                    versions_dir.append(filename)
        versions_dir = sorted(versions_dir,reverse=True)
        if versions_dir == []:
            self.version = 1
        else:
            self.version = int(versions_dir[0].split("_")[-1])+1
        self.version = str(self.version)
        self.cass_name = self.base_name+"_"+self.version
        self.cli_name = "~$/"+self.base_name+"_"+self.version+">"
        self.version_path = self.root_path+"/"+self.cass_name

    def build_control(self):
        builds_dir = []
        if not os.path.exists(self.version_path+"/Builds"):
            os.makedirs(self.version_path+"/Builds")
        for filename in os.listdir(self.version_path+"/Builds"):
            if os.path.isdir(os.path.join(self.version_path+"/Builds",filename)):
                if re.match(re.escape(netco.BUILD),filename):
                    builds_dir.append(filename)
        builds_dir = sorted(builds_dir,reverse=True)
        if builds_dir == []:
            self.build_version = 1
        else:
            self.build_version = int(builds_dir[0].split("_")[-1])+1
        self.build_version = str(self.build_version)
        build_code = 'Builds/'+netco.BUILD+'_'+self.build_version
        self.build_path = os.path.join(self.version_path,build_code)

    def layers_import(self,json_path):
        print(f"{self.cli_name} Importing Network Structure from: {json_path}")
        obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        self.structure = np.array(b_new["network_structure"])#,dtype=int)

    def construct(self):
        for level_counter,level in enumerate(self.structure,start=0):
            if level_counter==0:
                flatten = tf.keras.layers.Flatten(name=netco.FLATTEN)
                setattr(self,netco.FLATTEN,flatten)
            activation_function = level[2]
            if activation_function == netco.LINEAR:
                # [LINEAR layer]
                layer = tf.keras.layers.Dense(int(level[0]),
                kernel_regularizer=tf.keras.regularizers.l2(l=0),
                name=netco.LINEAR+'_'+str(level_counter))
            elif activation_function == netco.TANH:
                # [TANH layer]
                layer =tf.keras.layers.Dense(int(level[0]),
                activation=tf.nn.tanh,
                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                name=netco.TANH+'_'+str(level_counter))
            elif activation_function == netco.SIGMOID:
                # [SIGMOID layer]
                layer = tf.keras.layers.Dense(int(level[0]),
                activation=tf.nn.sigmoid,
                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                name=netco.SIGMOID+'_'+str(level_counter))
            elif activation_function == netco.RELU:
                # [RELU layer]
                layer = tf.keras.layers.Dense(int(level[0]),
                activation=tf.nn.relu,
                kernel_regularizer=tf.keras.regularizers.l2(l=0.01),
                name=netco.RELU+'_'+str(level_counter))
            elif activation_function == netco.SOFTMAX:
                # [SOFTMAX layer]
                layer = tf.keras.layers.Dense(int(level[0]),
                activation=tf.nn.softmax,
                kernel_regularizer=tf.keras.regularizers.l2(l=0),
                name=netco.SOFTMAX+'_'+str(level_counter))
            setattr(self,netco.LAYER+str(level_counter),layer)
            #if layer_counter!=len(self.structure)-1:
            #    self.add(tf.keras.layers.Dropout(0.2))

        self.optimizer = tf.optimizers.Adam(
            learning_rate=self.learning_rate,beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
        )
        self.compile()

    def call(self,X):
        #X = getattr(self,netco.FLATTEN)(X)
        for level_counter,_ in enumerate(self.structure,start=0):
            X = getattr(self,netco.LAYER+str(level_counter))(X)
        return X
    
    @tf.function
    def train_step(self, samples,labels):
        with tf.GradientTape() as tape:
            current_loss = self.loss_fun(labels, self(samples))
            grads = tape.gradient(current_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.train_loss(current_loss)
        #self.train_accuracy(labels, self(samples))

    @tf.function
    def test_step(self, samples, labels):
        t_loss = self.loss_fun(labels, self(samples))
        self.test_loss(t_loss)
        #self.test_accuracy(labels, self(samples))

    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        '''Network train function. \n
        |-> Arguments(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)
        '''
        self.function = netco.TRAIN
        # On a new training process of the same network we check for new build/train up to 10
        costs_flag = False
        self.build_control()
        os.mkdir(self.build_path)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        X_data = self.normalize(data)
        X_data = X_data.astype('float32')
        
        if self.edition == netco.TREND or self.edition == netco.CYCLES:
            X_data = X_data.astype({'LABEL': int})

        X_train, X_test = train_test_split(X_data, test_size=test_size)
        print(X_train)
        Y_train = X_train.pop('LABEL')
        Y_test = X_test.pop('LABEL')
        self.test_df = Y_test
        print(self.test_df)
        DATASET_TRAIN = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
        DATASET_TEST = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))
        DATASET_TEST = DATASET_TEST.batch(minibatch_size)
        
        if shuffle=='True':
            #DATASET_TRAIN = DATASET_TRAIN.shuffle(len(X_train)-1).batch(minibatch_size)
            DATASET_TRAIN = DATASET_TRAIN.shuffle(1024).batch(minibatch_size)
            print(f"{self.cli_name} Splitting Dataset with size {test_size}. Shuffling: Enabled!")
        else:
            DATASET_TRAIN = DATASET_TRAIN.batch(minibatch_size)
            print(f"{self.cli_name} Splitting Dataset with size {test_size}. Shuffling: Disabled!")
        
        tf.random.set_seed(31)
        seed = 3
        '''
        self.network_metrics = pd.DataFrame(
            columns=[
                netco.TRAIN_LOSS,netco.TRAIN_ACC,
                netco.TEST_LOSS,netco.TEST_ACC
                ])
        '''
        self.network_metrics = pd.DataFrame(
            columns=[
                netco.TRAIN_LOSS,
                netco.TEST_LOSS
                ])
        self.construct()
        begin = time.time()
        # [Training Loop]
        with tqdm(total = epochs, unit="epo") as pbar:
            for epoch in range(epochs):
                self.train_loss.reset_states()
                #self.train_accuracy.reset_states()
                self.test_loss.reset_states()
                #self.test_accuracy.reset_states()

                for samples,labels in DATASET_TRAIN:
                    self.train_step(samples,labels)
            
                for test_samples, test_labels in DATASET_TEST:
                    self.test_step(test_samples, test_labels)

                train_epo_cost = round(self.train_loss.result().numpy(),6)
                #train_epo_acc = round(self.train_accuracy.result().numpy(),6)*100
                test_epo_cost = round(self.test_loss.result().numpy(),6)
                #test_epo_acc = round(self.test_accuracy.result().numpy(),6)*100
                #self.network_metrics.loc[epoch] = [train_epo_cost, train_epo_acc, test_epo_cost, test_epo_acc]
                self.network_metrics.loc[epoch] = [train_epo_cost, test_epo_cost]
                pbar.set_description(f"~$> Loss: {train_epo_cost:.3f}, Test Loss: {test_epo_cost:.3f}")
                pbar.refresh() # to show immediately the update
                pbar.update(n=1)
                if epoch % 50 == 0:
                    print('')
        print(self.network_metrics)

        numpy_trainable_variables = {}
        for var in self.trainable_variables:
            var_type = var.name.split('/')[2].split(':')[0]
            layer = var.name.split('/')[1].split('_')[1]
            if var_type == 'kernel':
                var_name = 'W'+layer
            elif var_type == 'bias':
                var_name = 'b'+layer
            numpy_trainable_variables[var_name] = var.numpy()
        finish = time.time()
        self.training_time = round(finish-begin,4)
        self.test_predictions = self.predict(DATASET_TEST).flatten()
        self.network_training_summary_report(numpy_trainable_variables)

    def normalize(self,data):
        '''Normalize data and save normalizers for later inference'''
        normalizers_file = "data_normalizers.json"
        save_path = os.path.join(self.version_path,normalizers_file)
        if self.function == netco.TRAIN:
            final_df, normalizers = normalizeDataFrame(data)
            normalizers.to_json(path_or_buf=save_path,orient='columns')
        elif self.function == netco.INFERENCE:
            normalizers = pd.read_json(save_path)
            final_df = pd.DataFrame(columns=data.columns)
            for column in data.columns:
                if (column=='D_MAX' or column=='D_AVE'):
                    final_df[column] = data[column].apply(lambda x: 1-(x-normalizers.loc[column].min())/(normalizers.loc[column].max()-normalizers.loc[column].min()))
                else:
                    final_df[column] = data[column].apply(lambda x: (x-normalizers.loc[column].min())/(normalizers.loc[column].max()-normalizers.loc[column].min()))
        return final_df

    def network_training_summary_report(self,numpy_trainable_variables):
        '''Save a training summary report for the trained network'''
        settings_file = self.cass_name+".training_summary.txt"
        layers_matlab_file = self.cass_name+".mat"
        exit_path = os.path.join(self.build_path,settings_file)
        layers_path = os.path.join(self.build_path,layers_matlab_file)
        lines = [
                f'[GENERAL-INFORMATION]',
                f'Network Edition: {self.edition}',
                f'Network Name: {self.cass_name}',
                f'Network Version: {self.version}',
                f'Network Build Version: {self.build_version}',
                f'',
                f'[TRAINING-EVALUATION-REPORT]',
                f'Epochs: {self.epochs}',
                f'Learning Rate: {self.learning_rate}',
                f'Minibatch Size: {self.minibatch_size}',
                #f'Training Accuracy: {self.network_metrics[netco.TRAIN_ACC].loc[self.epochs-1]} %',
                #f'Testing Accuracy: {self.network_metrics[netco.TEST_ACC].loc[self.epochs-1]} %',
                f'Training Time: {self.training_time} seconds',
                f'',
                f'[PATHS]',
                f'root_path: {self.root_path}',
                f'version_path: {self.version_path}',
                f'build_path: {self.build_path}'
                ]
        with open(exit_path,"w") as file:
            for line in lines:
                file.write(line+'\n')
        print(f"{self.cli_name} Exporting Information File to {self.build_path}")
        scipy.io.savemat(layers_path, numpy_trainable_variables)

class NNClassifier(Network):
    def __init__(self,edition,flag,base_name,root_path,features):
        Network.__init__(self,edition,flag,base_name,root_path,features)
        
    def get_general(self):
        self.loss_fun = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')            

    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        self.get_general()
        super().train(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)
        self.plot_accuracy()
        self.plot_costs()
        plt.show()

    def plot_costs(self):
        '''Timeplots of costs'''
        t = np.arange(0.0, self.epochs, 1)
        fig, ax = plt.subplots()
        ax.plot(t, self.network_metrics[netco.TRAIN_LOSS], label='Train Error')
        ax.plot(t, self.network_metrics[netco.TEST_LOSS], label='Test Error')
        ax.set(xlabel='Epochs', ylabel='Softmax Cross Entropy',title=f'Training and Testing Errors for {self.edition}')
        ax.grid()
        plt.legend()
        fig.savefig(self.build_path+'/costs.png',dpi=800)

    def plot_accuracy(self):
        '''Timeplots of accuracy'''
        t = np.arange(0.0, self.epochs, 1)
        fig, ax = plt.subplots()
        ax.plot(t, self.network_metrics[netco.TRAIN_ACC], label='Train Accuracy')
        ax.plot(t, self.network_metrics[netco.TEST_ACC], label='Test Accuracy')
        ax.set(xlabel='Epochs', ylabel='Accuracy',title=f'Training and Testing Accuracy for {self.edition}')
        ax.grid()
        plt.legend()
        plt.ylim(round_down(min(self.network_metrics['TEST_ACC'])-5,2),100)
        fig.savefig(self.build_path+'/accuracy.png',dpi=800)

    def inference(self,data,window_settings):
        '''def inference(self,data,window_settings):
        
        '''
        self.function = netco.INFERENCE 
        
        #full_df = data
        self.layers_import(self.version_path+"/network_structure.json")
        self.graph = tf.Graph()
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.version_path+'/model.ckpt.meta')
        self.layers = {}
        for op in self.graph.get_operations():
            for layer_counter,layer in enumerate(self.structure,start=0):
                pattern_W = '^Weights/W'+str(layer_counter)+'$'
                pattern_b = '^Biases/b'+str(layer_counter)+'$'
                if re.search(pattern_W, str(op.name)) is not None:
                    self.layers["W"+str(layer_counter)] = tf.cast(self.graph.get_tensor_by_name(str(op.name)+str(':0')), tf.float64,name="W"+str(layer_counter))

                if re.search(pattern_b, str(op.name)) is not None:
                    self.layers["b"+str(layer_counter)] = tf.cast(self.graph.get_tensor_by_name(str(op.name)+str(':0')), tf.float64,name="b"+str(layer_counter))
        # Start reading datas
        X = tf.cast(self.graph.get_tensor_by_name("X:0"), tf.float64)
        final = self.forward_propagation(X)
        '''
        with tf.compat.v1.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.version_path+"/model.ckpt")
            n_X_data = self.normalize(full_df)
            X_inf = n_X_data.values.transpose()
            predictions = sess.run(tf.argmax(final),feed_dict={X: X_inf})
        

        full_df[netco.PREDICTION_LABEL] = predictions
        '''
        #full_df.plot()

        full_df = pd.read_csv(os.path.join(self.root_path,'samples',data))

        #full_df=full_df.head(1500)
        #full_df = full_df[900:1700]
        window_size = int(self.root_path.split('/')[-1].split('_')[1])
        window_step = int(self.root_path.split('/')[-1].split('_')[2])
        window_settings =[window_size,window_step]
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.version_path+"/model.ckpt")
            fit_df = onlineData2(full_df['E_REV'],window_settings)
            n_X_data = self.normalize(fit_df)
            X_inf = n_X_data.values.transpose()
            predictions = sess.run(tf.argmax(final),feed_dict={X: X_inf}).T
        
        
        font = {#'family':'',
        'weight':'normal',
        'size': 14,
        }
        CLASSES = {0:'PC 1',1:'PC 2',2:'PC 3',3:'PC 4',4:'PC 5',5:'PC 6'}
        #plt.show()
        matplotlib.rc('font', **font)
        fig, ax = plt.subplots(figsize=(10,8))
        t = np.arange(0, len(full_df), 1)
        t2 = np.arange(window_size-1, len(full_df), 1)
        ax.plot(t, full_df['E_REV'], label='Pitch Cycle',color='green')
        x_lines = [window_size-1]
        hold = 0
        prev_hold = 0
        for i,prediction in enumerate(predictions):
            if i==0:
                pass
            else:
                if prev_prediction != prediction:
                    x_lines.append(i+window_size-1)
            prev_prediction = prediction
        x_lines.append(len(full_df))
        for i,x in enumerate(x_lines):
            if i==0:
                pass
            else:
                xs = np.arange(x_lines[i-1], x_lines[i])

                ys = xs*0+0.8
                ax.plot(xs, ys, label='Predicted Cycle' if i == 1 else "",color='red')
                if len(xs)==1:
                    prediction = predictions[int(xs[-1]-window_size+1)]
                else:
                    prediction = predictions[int(xs[-1]-window_size+1)]
                if i%2==0:
                    ax.text(int(xs.mean()), 0.82, CLASSES[prediction],color='red',fontsize=20)
                else:
                    ax.text(int(xs.mean()), 0.76, CLASSES[prediction],color='red',fontsize=20)
                
                ax.plot(xs[0], 0.8, 'r<')
                ax.plot(xs[-1], 0.8, 'r>')

        x_lines=[0]
        labels = list(full_df['LABEL'])
        for i,label in enumerate(labels):
            if i==0:
                pass
            else:
                if prev_laber!= label:
                    x_lines.append(i)
            prev_laber = label
        x_lines.append(len(full_df))
        for i,x in enumerate(x_lines):
            if i==0:
                pass
            else:
                xs = np.arange(x_lines[i-1], x_lines[i])
                ys = xs*0+0.9
                ax.plot(xs, ys, label='Actual Cycle' if i == 1 else "",color='blue')
                label = labels[int(xs.mean())]
                if i%2==0:
                    ax.text(int(xs.mean()), 0.92, CLASSES[label],color='blue',fontsize=20)
                else:
                    ax.text(int(xs.mean()), 0.86, CLASSES[label],color='blue',fontsize=20)
                
                ax.plot(xs[0], 0.9, 'b<')
                ax.plot(xs[-1], 0.9, 'b>')
        
        ax.set(xlabel=r'$\mathbf{Time}$ (sec)', ylabel=r'$\mathbf{\beta}$ / $\mathbf{\beta_{max}}$',title='Prediction Cycle')
        ax.grid()
        plt.ylim(0,1)
        plt.legend()
        fig.savefig(os.path.join(self.version_path,data.split('.')[0]+'_cycle_inference.png'),dpi=800)
        
        
        pd.DataFrame(predictions).to_csv(os.path.join(self.version_path,data.split('.')[0]+'_cycle_inference.csv'))
        full_df.to_csv(os.path.join(self.version_path,data.split('.')[0]+'_cycle_data.csv'))
        full_df = full_df.tail(len(predictions))
        print(full_df)
        n = 0
        
        for i in range(len(predictions)):
            if int(list(full_df['LABEL'])[i])==predictions[i]:
                n+=1
        acc=n/len(predictions)
        print(acc)
        plt.show()
        
        return full_df
       
class NNRegressor(Network):
    def __init__(self,edition,flag,base_name,root_path,features):
        Network.__init__(self,edition,flag,base_name,root_path,features)
        
    def get_general(self):
        self.loss_fun = tf.keras.losses.MeanSquaredError()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        #self.train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
        #self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        #self.test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
        #self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')   


    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        self.get_general()
        cycle_full = self.root_path.split('/')[-1]
        cycle = cycle_full.split('_')[-1]
        cycle = str(int(cycle)+1)
        
        super().train(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)

        self.plot_trained(cycle)
        self.plot_costs(cycle)
        self.plot_predictions(cycle)
        #plt.close('all')
        
        plt.show()

    def plot_trained(self,cycle):
        fig, ax = plt.subplots()
        t = np.arange(0.0, len(self.test_df), 1)
        ax.plot(t, self.test_df, label='NMPC',color='blue')
        ax.plot(t, self.test_predictions, label='NN',color='red')
        ax.set(xlabel='Time [Secs]', ylabel='Torque [Nm]',title=self.edition+' Torque (Cycle '+cycle+')')
        ax.grid()
        plt.legend()
        #fig.savefig(self.build_path+'/trained_model_cycle_'+cycle+'.png',dpi=800)
        frame = pd.DataFrame()
        frame['LABEL'] = self.test_df
        frame['PREDICTION'] = self.test_predictions
        frame.to_csv(os.path.join(self.build_path,'results_'+self.edition+'_'+cycle+'.csv'),index=False)


    def plot_predictions(self,cycle):
        fig, ax = plt.subplots()
        ax.scatter(self.test_df, self.test_predictions,color='red')
        ax.set(xlabel='True Values [Torque]', ylabel='Predictions [Torque]',title=self.edition+' Predictions (Cycle '+cycle+')')
        ax.axis('equal')
        ax.axis('square')
        ax.grid()

        if self.edition==netco.ENGINE:
            plt.xlim([0,plt.xlim()[1]])
            plt.ylim([0,plt.ylim()[1]])
        elif self.edition==netco.MOTOR:
            plt.xlim([-plt.xlim()[1],plt.xlim()[1]])
            plt.ylim([-plt.ylim()[1],plt.ylim()[1]])
        _ = plt.plot([-2000, 2000], [-2000, 2000])
        
        #fig.savefig(self.build_path+'/predictions_cycle_'+cycle+'.png',dpi=800)

    def plot_costs(self,cycle):
        '''Timeplots of costs'''
        t = np.arange(0.0, self.epochs, 1)
        fig, ax = plt.subplots()
        ax.plot(t, self.network_metrics[netco.TRAIN_LOSS], label='Train Error')
        ax.plot(t, self.network_metrics[netco.TEST_LOSS], label='Test Error')
        ax.set(xlabel='Epochs', ylabel='Mean Absolute Error (Torque)',title=self.edition+' Training and Testing Errors (Cycle '+cycle+')')
        ax.grid()
        plt.legend()
        #fig.savefig(self.build_path+'/costs_cycle_'+cycle+'.png',dpi=800)

        

