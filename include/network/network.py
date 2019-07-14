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

import include.network.net_constants as netco
import include.network.net_setup as nets
from include.utils import normalizeDataFrame
#tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create grap

class Network():
    def __init__(self,edition,flag,name,root_path,features):
        self.name = name
        self.root_path = root_path
        self.layers = {}
        self.edition = edition
        self.features = features
        if flag==netco.CREATE:
            print("~$> Creating Network")
            # On new network creation we check for previous versions up to 10
            self.version_control()
        elif flag==netco.LOAD:
            print("~$> Loading Network")
            self.cli_name = "~$/"+self.name+">"
            self.version_path = os.path.join(self.root_path,self.name)
            self.version = self.name.split("_")[-1]
 
    def version_control(self):
        versions_dir = []
        for filename in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,filename)):
                if re.match(re.escape(self.name),filename):
                    versions_dir.append(filename)
        versions_dir = sorted(versions_dir,reverse=True)
        if versions_dir == []:
            self.version = 1
        else:
            self.version = int(versions_dir[0].split("_")[-1])+1
        self.version = str(self.version)
        self.cli_name = "~$/"+self.name+"_"+self.version+">"
        self.version_path = self.root_path+"/"+self.name+"_"+self.version

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
        self.structure = np.array(b_new["network_structure"],dtype=int)

    #[DEPRECATED]
    '''
    def show_data(self,flag):
        plt.figure(1)
        plt.subplot(211)
        if flag == netco.TRAINING:
            time_domain = np.arange(0.0,len(self.train_df),1)
            plt.title('Training Data')
            plt.plot(time_domain, self.train_df.drop(['A_AVE'],axis=1).values, marker='.')
            plt.subplot(212)
            plt.title('Training Data ACC')
            plt.plot(time_domain, self.train_df[["A_AVE"]], marker='.')
            plt.show()
        elif flag == netco.TESTING:
            time_domain = np.arange(0.0,len(self.test_df),1)
            plt.title('Testing Data')
            plt.plot(time_domain, self.test_df.drop(['A_AVE'],axis=1).values, marker='.')
            plt.subplot(212)
            plt.title('Testing Data ACC')
            plt.plot(time_domain, self.test_df[["A_AVE"]], marker='.')
            plt.show()
        else:
            print("ERROR BAD DATA FLAG")
    '''
    def train(self,epochs,learning_rate,minibatch_size):
        # On a new training process of the same network we check for new build/train up to 10
        costs_flag = False
        self.build_control()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        graph = tf.Graph()
        self.graph = graph
        begin = time.time()
        X_data = pd.read_csv(self.root_path+"/"+netco.TRAINING+".csv",usecols=self.features)
        X_data = normalizeDataFrame(X_data)
        X_data = X_data.astype({'LABEL': int})
        if self.edition == netco.TREND:
            X_train, X_test = train_test_split(X_data, test_size=0.3, shuffle=False)
            outputs = netco.TREND_OUTPUTS
        else:
            X_train, X_test = train_test_split(X_data, test_size=0.3, shuffle=True)
            outputs = netco.CYCLES_OUTPUTS
        
        self.train_df = X_train
        self.test_df = X_test
        # [Plotting Train and Test Data at the stage of training]
        Y_train = nets.labelMaker(X_train[['LABEL']],outputs).transpose()
        Y_test = nets.labelMaker(X_test[['LABEL']],outputs).transpose()
        X_train = X_train.drop(["LABEL"],axis=1)
        X_test = X_test.drop(["LABEL"],axis=1)
        X_train = X_train.values.transpose()
        X_test = X_test.values.transpose()

        tf.set_random_seed(1)
        tf.reset_default_graph()
        seed = 3
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        with self.graph.as_default():
            serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
            feature_configs = {'x': tf.FixedLenFeature(shape=[n_y], dtype=tf.float32),}
            tf_example = tf.parse_example(serialized_tf_example, feature_configs)
            # [Create Placeholders of shape (n_x, n_y)]
            X, Y = nets.create_placeholders(n_x, n_y)
            # [Initialize layer parameters]
            self.init_layers()
            # [Add Forward Propogation to the graph]
            final = self.forward_propagation(X)
            final = tf.identity(final, name="Final")
            # [Adding cost function for the Adam optimizer to the graph]
            cost = nets.function_cross_entropy(final, Y)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            saver = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:
            # [Session Initialization]
            sess.run(tf.global_variables_initializer())
            values, indices = tf.nn.top_k(final, n_y)
            table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(n_y)]))
            prediction_classes = table.lookup(tf.to_int64(indices))
            builder = tf.saved_model.builder.SavedModelBuilder(self.build_path)
            train_writer = tf.summary.FileWriter(self.version_path+"/train", sess.graph)

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
            if costs_flag:
                print("")
                print(costs)

            # [Saving the parameters in a variable]
            self.layers = sess.run(self.layers)
            print(self.cli_name+" Finished Neural Training!")
            
            classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

            classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs},
                outputs={
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores},
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
            
            tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(final)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x},
                    outputs={'scores': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                main_op=tf.tables_initializer(),
                strip_default_attrs=True)

            builder.save()
            
            finish = time.time()
            self.training_time = round(finish-begin,2)
            print(self.cli_name+" Time for training was",round(finish-begin,2),"seconds")

            # [Calculating the correct predictions]
            correct_prediction = tf.equal(tf.argmax(final), tf.argmax(Y))
            self.predictions_train = sess.run(tf.argmax(final),feed_dict={X: X_train})
            self.predictions_test = sess.run(tf.argmax(final),feed_dict={X: X_test})
            self.predictions = np.concatenate((self.predictions_train,self.predictions_test),axis=0)
            tf.summary.histogram("predictions", final)
            # [Calculate accuracy on the test set]
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            self.train_acc = round(100*accuracy.eval({X: X_train, Y: Y_train}),2)
            self.test_acc = round(100*accuracy.eval({X: X_test, Y: Y_test}),2)
            print(self.cli_name+" Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print(self.cli_name+" Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
            saver.save(sess, self.version_path+'/model')
            self.network_training_summary_report()

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
        for layer_counter,layer in enumerate(self.structure,start=1):
            name_W = "W"+str(layer_counter)
            name_b = "b"+str(layer_counter)
            with tf.variable_scope('Weights'):
                self.layers[name_W] = tf.get_variable(name_W, [layer[0], layer[1]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            with tf.variable_scope('Biases'):
                self.layers[name_b] = tf.get_variable(name_b, [layer[0], 1], initializer = tf.zeros_initializer())
            
    def forward_propagation(self,X):
        Z = []
        A = []
        A.append(X)
        for layer,_ in enumerate(self.structure,start=1):
            if layer == len(self.structure):
                # [Last layer - LINEAR]
                Z.append(tf.add(tf.matmul(self.layers["W"+str(layer)], A[layer-1]), self.layers["b"+str(layer)]))
                A.append(Z[layer-1])
            else:
                # [All hidden layers - TANH]
                Z.append(tf.add(tf.matmul(self.layers["W"+str(layer)], A[layer-1]), self.layers["b"+str(layer)]))
                A.append(tf.nn.sigmoid(Z[layer-1]))
        # [Return last layer]
        return A[-1]

    #[DEPRECATED]
    def export_predictions(self,start,end):
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
        exit_path = os.path.join(self.build_path,settings_file)
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
    
    #[TO BE CHANGED]
    def inference(self,window_settings):
        begin = time.time()
        df = pd.read_csv(self.root_path+"/visual.csv",engine='python')
        X_data = nets.cycleInference(df,netco.CYCLES_FEATURES,window_settings,self.root_path,True)
        print(X_data)
        n_X_data = normalizeDataFrame(X_data)
        n_X_data = n_X_data.drop(["LABEL"],axis=1)
        n_X_data = n_X_data.values.transpose()

        self.layers_import(self.version_path+"/network_structure.json")
        self.graph = tf.Graph()
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.version_path+'/model.meta')
        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, self.version_path+"/model")
            final = self.graph.get_tensor_by_name("Final:0")
            X = self.graph.get_tensor_by_name("X:0")
            self.predictions_visual = sess.run(tf.argmax(final),feed_dict={X: n_X_data})

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
            if self.predictions_visual[i]!=self.predictions_visual[i-1]:
                hold = int(window_settings[0])+i*int(window_settings[1])
                x_lines.append([prev_hold-1,hold-1,i-1])
                prev_hold = hold

        x_lines.append([x_lines[-1][1]-1,len(df)-1,len(X_data)-1])
        del x_lines[0]
        for pair in x_lines:
            xs = np.arange(pair[0], pair[1])
            ys = xs*0+0.5
            prediction = str(self.predictions_visual[pair[2]])
            ax.text((pair[0]+pair[1])/2, 0.6, prediction,bbox=dict(facecolor='red', alpha=0.5))
            plt.plot(xs, ys,'-b')
            plt.plot(pair[0], 0.5, 'b<')
            plt.plot(pair[1]-1, 0.5, 'b>')
        
        print(round(time.time()-begin,1),'Seconds')
        plt.show()
