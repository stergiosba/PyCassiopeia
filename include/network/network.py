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

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.io

import include.network.net_constants as netco
import include.network.net_setup as nets
from include.utils import normalizeDataFrame
from include.network.online import onlinePrediction
#tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create grap

class Network():
    '''General Network Class
    '''
    def __init__(self,edition,flag,base_name,root_path,features):
        self.root_path = root_path
        self.layers = {}
        self.edition = edition
        self.features = features
        if flag==netco.CREATE:
            self.base_name = base_name
            # On new network creation we check for previous versions up to 10
            self.version_control()
            print("~$> Creating Network: |-> "+self.name)
        elif flag==netco.LOAD:
            self.name = base_name
            print("~$> Loading Network: |-> "+self.name)
            self.cli_name = "~$/"+self.name+">"
            self.version_path = os.path.join(self.root_path,self.name)
            self.version = self.name.split("_")[-1]
 
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
        self.name = self.base_name+"_"+self.version
        self.cli_name = "~$/"+self.base_name+"_"+self.version+">"
        self.version_path = self.root_path+"/"+self.name

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
        print(self.cli_name+" Importing Network Structure from: "+json_path)
        obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        self.structure = np.array(b_new["network_structure"])#,dtype=int)

    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        '''Network train function. \n
        |-> Arguments(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)
        '''
        # On a new training process of the same network we check for new build/train up to 10
        costs_flag = False
        self.build_control()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        graph = tf.Graph()
        self.graph = graph
        begin = time.time()
        X_data = normalizeDataFrame(data)
        if self.edition == netco.TREND or self.edition == netco.CYCLES:
            X_data = X_data.astype({'LABEL': int})

        if shuffle=='True':
            shuffle = True
            print(self.cli_name+" Splitting Dataset with size "+str(test_size)+". Shuffling: Enabled!")
        else:
            shuffle = False
            print(self.cli_name+" Splitting Dataset with size "+str(test_size)+". Shuffling: Disabled!")

        X_train, X_test = train_test_split(X_data, test_size=test_size, shuffle=shuffle)
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        self.train_df = X_train
        self.test_df = X_test

        if self.edition == netco.TREND:
            Y_train = nets.labelMaker(X_train[['LABEL']],outputs).transpose()
            Y_test = nets.labelMaker(X_test[['LABEL']],outputs).transpose()
        elif self.edition == netco.CYCLES:
            Y_train = nets.labelMaker(X_train[['LABEL']],outputs).transpose()
            Y_test = nets.labelMaker(X_test[['LABEL']],outputs).transpose()
        elif self.edition == netco.ENGINE:
            Y_train = X_train[['LABEL']].values.transpose()
            Y_test = X_test[['LABEL']].values.transpose()
        else:
            Y_train = X_train[['LABEL']].values.transpose()
            Y_test = X_test[['LABEL']].values.transpose()
        # [Plotting Train and Test Data at the stage of training]
        X_train = X_train.drop(["LABEL"],axis=1)
        X_test = X_test.drop(["LABEL"],axis=1)
        X_train = X_train.values.transpose()
        X_test = X_test.values.transpose()
        
        Y_fake = X_data[['LABEL']]
        X_fake = X_data.drop(["LABEL"],axis=1)
        X_fake = X_fake.values.transpose()
        
        tf.compat.v1.set_random_seed(1)
        tf.compat.v1.reset_default_graph()
        seed = 3
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        with self.graph.as_default():
            serialized_tf_example = tf.compat.v1.placeholder(tf.string, name='tf_example')
            feature_configs = {'x': tf.io.FixedLenFeature(shape=[n_y], dtype=tf.float32),}
            tf_example = tf.io.parse_example(serialized_tf_example, feature_configs)
            # [Create Placeholders of shape (n_x, n_y)]
            X, Y = nets.create_placeholders(n_x, n_y)
            # [Initialize layer parameters]
            self.init_layers()
            # [Add Forward Propogation to the graph]
            final = self.forward_propagation(X)
            final = tf.identity(final, name="Final")
            # [Adding cost function for the Adam optimizer to the graph]
            #cost = nets.softmax_cross_entropy(final, Y)
            if self.edition == netco.TREND or self.edition == netco.CYCLES:
                loss = 'softmax_cross_entropy'
                output_function = tf.argmax
            else:
                loss = 'mean_squared_error'
                output_function = tf.identity
            cost = nets.network_cost_function(final, Y, loss)
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)
            saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session(graph=self.graph) as sess:
            # [Session Initialization]
            sess.run(tf.global_variables_initializer())
            values, indices = tf.nn.top_k(final, n_y)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(n_y)]))
            #prediction_classes = table.lookup(tf.to_int64(indices))
            prediction_classes = table.lookup(tf.cast(indices,tf.int64))
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.build_path)
            train_writer = tf.compat.v1.summary.FileWriter(self.version_path+"/train", sess.graph)

            # [Training loop]
            print(self.cli_name+"\n")
            with tqdm(total = epochs,desc = "Cost:  ",unit="epo") as pbar:
                for epoch in range(epochs):
                    # [Calculate cost related to an epoch]
                    epoch_cost = 0
                    num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                    seed = seed + 1
                    minibatches = nets.random_mini_batches(X_train, Y_train, minibatch_size, seed)
                    with tqdm(total=num_minibatches,desc = "Progress:  ",unit="mini") as minibar:
                        for minibatch in minibatches:
                            # [Select a minibatch]
                            (minibatch_X, minibatch_Y) = minibatch
                        
                            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                            epoch_cost += minibatch_cost / num_minibatches
                            minibar.update(n=1)
                    
                    # [Print progress and the cost every epoch]
                    if epoch % 1 == 0:
                        pbar.set_description('Cost: {} '.format(round(epoch_cost,4)))
                        pbar.refresh() # to show immediately the update
                        pbar.update(n=1)
                        costs.append(round(epoch_cost,4))
                    if epoch % 50 == 0:
                        print('')
            if costs_flag:
                print("")
                print(costs)
            # [Saving the parameters in a variable]
            self.layers = sess.run(self.layers)
            print(self.cli_name,"Finished Neural Training!")
            
            classification_inputs = tf.compat.v1.saved_model.utils.build_tensor_info(serialized_tf_example)
            classification_outputs_classes = tf.compat.v1.saved_model.utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = tf.compat.v1.saved_model.utils.build_tensor_info(values)

            classification_signature = (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.CLASSIFY_INPUTS:classification_inputs},
                outputs={
                    tf.saved_model.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
                    tf.saved_model.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores},
                method_name=tf.saved_model.CLASSIFY_METHOD_NAME))
            
            tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(final)

            prediction_signature = (
                tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x},
                    outputs={'scores': tensor_info_y},
                    method_name=tf.saved_model.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                main_op=tf.compat.v1.tables_initializer(),
                strip_default_attrs=True)

            builder.save()
            
            finish = time.time()
            self.training_time = round(finish-begin,2)
            print(self.cli_name,"Time for training was",self.training_time,"seconds")

            # [Calculating the correct predictions]
            correct_prediction = tf.equal(output_function(final), output_function(Y))
            self.predictions_train = sess.run(output_function(final),feed_dict={X: X_train}).T
            self.predictions_test = sess.run(output_function(final),feed_dict={X: X_test}).T
            self.predictions = np.concatenate((self.predictions_train,self.predictions_test),axis=0)
            tf.compat.v1.summary.histogram("predictions", final)
            # [Calculate accuracy on the test set]
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.train_acc = round(100*accuracy.eval({X: X_train, Y: Y_train}),2)
            self.test_acc = round(100*accuracy.eval({X: X_test, Y: Y_test}),2)
            print(self.cli_name+" Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print(self.cli_name+" Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            saver.save(sess, self.version_path+'/model.ckpt')
            '''
            tt=pd.DataFrame(sess.run(output_function(final),feed_dict={X: X_fake}))
            ax = Y_fake.plot()
            tt.plot(ax=ax)
            plt.savefig(self.build_path+'/trained_model.pdf')
            '''
            self.network_training_summary_report()
            #plt.show()
            
        return self.predictions
        
    def init_layers(self):
        """
        Initializes parameters to build a neural network with tensorflow. The shapes are:
            W1 : [num_hidden_layer_1, num_input_features]
            b1 : [num_hidden_layer_1, 1]
            ...[hidden layers]
            Wn : [num_hidden_layer_n, num_hidden_layer_n-1]
            bn : [num_hidden_layer_n, 1]
        Returns:
            self.layers -- a dictionary of tensors containing W1, b1, W2, b2...
        """
        for layer_counter,layer in enumerate(self.structure,start=0):
            name_W = "W"+str(layer_counter)
            name_b = "b"+str(layer_counter)
            with tf.compat.v1.variable_scope('Weights'):
                self.layers[name_W] = tf.compat.v1.get_variable(name_W, [int(layer[0]), int(layer[1])], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            with tf.compat.v1.variable_scope('Biases'):
                self.layers[name_b] = tf.compat.v1.get_variable(name_b, [int(layer[0]), 1], initializer = tf.zeros_initializer())
            
    def forward_propagation(self,X):
        # [A: List of linear combinations]
        # [Z: LIst of layer outputs after the activation function has taken effect on A]
        A = []
        Z = []
        Z.append(X)
        for layer_counter,layer in enumerate(self.structure,start=0):
            activation_function = layer[2]
            if activation_function == netco.LINEAR:
                # [LINEAR layer]
                A.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], Z[layer_counter]), self.layers["b"+str(layer_counter)]))
                Z.append(A[layer_counter])
            elif activation_function == netco.TANH:
                # [TANH layer]
                A.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], Z[layer_counter]), self.layers["b"+str(layer_counter)]))
                Z.append(tf.nn.tanh(A[layer_counter]))
            elif activation_function == netco.SIGMOID:
                # [SIGMOID layer]
                A.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], Z[layer_counter]), self.layers["b"+str(layer_counter)]))
                Z.append(tf.nn.sigmoid(A[layer_counter]))
            elif activation_function == netco.RELU:
                # [RELU layer]
                A.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], Z[layer_counter]), self.layers["b"+str(layer_counter)]))
                Z.append(tf.nn.relu(A[layer_counter]))
            else:
                raise NameError('There is an error with the activation functions in network_structure.json. Check the file and retry.')
        # [Return last layer]
        return Z[-1]

    def export_predictions(self,start,end):
        '''[DEPRECATED]'''
        full_df = pd.read_csv(os.getcwd()+"/templates.csv", header=None, engine="python").T
        self.test_df['PRED_LABEL'] = self.predictions_test
        show_df = self.test_df[start:end]
        show_df = show_df.reset_index()
        fig = plt.figure(3)
        plt.title('test')
        plt.ylim(0, 1)
        plot_feat = 'N_MAX'
        ax = show_df[plot_feat].plot()
        for i in range(len(show_df)):
            y = show_df[plot_feat][i]
            label = str(show_df['LABEL'][i])+"/"+str(show_df['PRED_LABEL'][i])
            if (show_df['LABEL'][i] == show_df['PRED_LABEL'][i]):
                ax.text(i, y, label,bbox=dict(facecolor='green', alpha=0.5))
            else:
                ax.text(i, y, label,bbox=dict(facecolor='red', alpha=0.5))

        self.test_df['PRED_LABEL'] = self.predictions_test
        show1_df = self.test_df[start:end]
        show1_df = show1_df.reset_index()
        fig1 = plt.figure(4)
        plt.title('train')
        plt.ylim(0, 1)
        plot_feat = 'N_MAX'
        ax = show1_df[plot_feat].plot()
        for i in range(len(show1_df)):
            if (show1_df['LABEL'][i] == show1_df['PRED_LABEL'][i]):
                ax.text(i, 0.8, show1_df['LABEL'][i],bbox=dict(facecolor='green', alpha=0.5))
                ax.text(i, 0.6, show1_df['PRED_LABEL'][i],bbox=dict(facecolor='green', alpha=0.5))
            else:
                ax.text(i, 0.8, show1_df['LABEL'][i],bbox=dict(facecolor='red', alpha=0.5))
                ax.text(i, 0.6, show1_df['PRED_LABEL'][i],bbox=dict(facecolor='red', alpha=0.5))
        plt.show()

    def network_training_summary_report(self):
        settings_file = self.name+".training_summary.txt"
        layers_matlab_file = self.name+".layers.m"
        exit_path = os.path.join(self.build_path,settings_file)
        layers_path = os.path.join(self.build_path,layers_matlab_file)
        lines = [
                '[GENERAL-INFORMATION]',
                'Network Edition: ' + self.edition,
                'Network Name: ' + self.name,
                'Network Version: ' + self.version,
                'Network Build Version: ' + self.build_version,
                '',
                '[TRAINING-EVALUATION-REPORT]',
                'Epochs: ' + str(self.epochs),
                'Learning Rate: ' + str(self.learning_rate),
                'Minibatch Size: ' + str(self.minibatch_size),
                'Training Accuracy: ' + str(self.train_acc) + "%",
                'Testing Accuracy: ' + str(self.test_acc) + "%",
                'Training Time: ' + str(self.training_time) + " seconds",
                '',
                '[PATHS]',
                'root_path: \"'+self.root_path+'\"',
                'version_path: \"'+self.version_path+'\"',
                'build_path: \"'+self.build_path+'\"'
                ]
        with open(exit_path,"w") as file:
            for line in lines:
                file.write(line+'\n')
        print(self.cli_name+" Exporting Information File to "+self.build_path)
        scipy.io.savemat(layers_path, self.layers)

    def inference(self,window_settings,sample):
        '''[TO BE CHANGED]'''
        begin = time.time()
        print(self.cli_name)
        df = pd.read_csv(self.root_path+"/samples/"+sample,engine='python')
        df = df.head(1000)
        X_data = nets.cycleInference(df[['E_REV']],netco.CYCLES_FEATURES,window_settings,self.root_path,True)

        n_X_data = normalizeDataFrame(X_data)
        n_X_data = n_X_data.drop(["LABEL"],axis=1)
        n_X_data = n_X_data.values.transpose()
        self.layers_import(self.version_path+"/network_structure.json")
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.version_path+'/model.ckpt.meta')
        
        with tf.compat.v1.Session(graph=self.graph) as sess:
            saver.restore(sess, self.version_path+"/model.ckpt")
            final = self.graph.get_tensor_by_name("Final:0")
            X = self.graph.get_tensor_by_name("X:0")
            self.predictions = sess.run(tf.argmax(final),feed_dict={X: n_X_data})
        font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }
        fig1 = plt.figure(4)
        plt.title('Inference Speed Profile', fontdict=font)
        plt.xlabel('Time (s)', fontdict=font)
        plt.ylabel('Engine Speed (rpm)', fontdict=font)
        plt.ylim(0, 1)
        plt.style.use('ggplot')
        plot_feat = 'E_REV'
        ax = df[plot_feat].plot()
        x_lines = []
        hold = 0
        prev_hold = 0
        for i in range(len(X_data)):
            if self.predictions[i]!=self.predictions[i-1]:
                hold = int(window_settings[0])+i*int(window_settings[1])
                x_lines.append([prev_hold-1,hold-1,i-1])
                prev_hold = hold

        x_lines.append([x_lines[-1][1]-1,len(df)-1,len(X_data)-1])
        del x_lines[0]
        for pair in x_lines:
            xs = np.arange(pair[0], pair[1])
            ys = xs*0+0.5
            prediction = str(self.predictions[pair[2]])
            ax.text((pair[0]+pair[1])/2, 0.6, prediction,bbox=dict(facecolor='red', alpha=0.5))
            plt.plot(xs, ys,'-b')
            plt.plot(pair[0], 0.5, 'b<')
            plt.plot(pair[1]-1, 0.5, 'b>')
        
        print(round(time.time()-begin,1),'Seconds')
        
        plt.show()

class NNClassifier(Network):
    def __init__(self,edition,flag,base_name,root_path,features):
        Network.__init__(self,edition,flag,base_name,root_path,features)

    def inference(self,data,window_settings):
        full_df = pd.read_csv(os.path.join(self.root_path,'samples',data))
        test_df = full_df[['E_REV']]
        self.layers_import(self.version_path+"/network_structure.json")
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            saver = tf.compat.v1.train.import_meta_graph(self.version_path+'/model.ckpt.meta')
        targets=[]
        # Start reading datas
        low_indx = 0
        high_indx = 0
        for index, row in test_df.iterrows():
            if index<int(window_settings[0]):
                targets.append(row[0])
            else:
                low_indx+=1
                print(50*'-')
                del targets[0]
                targets.append(row[0])
                window_df = pd.Series(targets)
                X_data = onlinePrediction(self.edition,window_df)
                n_X_data = normalizeDataFrame(X_data).T
                with tf.compat.v1.Session(graph=self.graph) as sess:
                    saver.restore(sess, self.version_path+"/model.ckpt")
                    final = self.graph.get_tensor_by_name("Final:0")
                    X = self.graph.get_tensor_by_name("X:0")
                    self.predictions = sess.run(tf.argmax(final),feed_dict={X: n_X_data})
                #required_df = full_df.iloc[high]
                #required_df['LABEL'] = go['LABEL'][0]
                #print(required_df)
                print(low_indx,high_indx)
                print(self.predictions)
            high_indx+=1
        


class NNRegressor(Network):
    def __init__(self,edition,flag,base_name,root_path,features):
        Network.__init__(self,edition,flag,base_name,root_path,features)

    def inference(self,data):
        sim_dir = os.path.join(os.getcwd(),netco.SIMULATIONS)
        sim_df = pd.read_csv(os.path.join(sim_dir,'simulation_0.csv'))

