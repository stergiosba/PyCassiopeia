# -*- coding: utf-8 -*-
"""
Code for the Neural Network Class.
Contains all calculation features and I/O Controls.

@author: stergios
"""
import os
import sys
import time
import json
import codecs

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split

from .network_setup import *
from .network_constants import *
#tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create grap

class Network():
    def __init__(self,name="Network",save_path=os.getcwd()+"/Network",structure=np.array([]),net_graph=tf.Graph(),export=True):
        print("~$> Creating Network")
        self.name = name
        self.cli_name = "~$/"+self.name+">"
        self.path = save_path+"/"+self.name
        self.structure = structure
        self.print_cost = True
        self.graph = net_graph
        self.layers = {}
        if export:
            print(self.cli_name+" Exporting Network Information")
            #self.layers_export()
            
    def layers_export(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with open(self.path+"/info.txt","w") as file:
            file.write(30*"-"+"\n")
            file.write("Network Structure Information\n")
            file.write(30*"#"+"\n")
            file.write("Layer Format: [IN,OUT]\n")
            file.write(30*"#"+"\n")
            layer_counter = 1
            for layer in self.structure:
                if layer_counter == 1:
                    file.write("Start: ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+"]\n")
                elif layer_counter == len(self.structure):
                    file.write("Exit:  ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+"]\n")
                else:
                    file.write("Hidden ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+"]\n")
                layer_counter+=1
            file.write(30*"-"+"\n")
        b = self.structure.tolist() # nested lists with same data, indices
        file_path = self.path+"/exit.json" ## your path variable
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)

    def layers_import(self,json_path):
        obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        self.structure = np.array(b_new["network_structure"])

    def show_data(self,flag):
        plt.figure(1)
        plt.subplot(211)
        if flag == TRAINING:
            time_domain = np.arange(0.0,len(self.train_df),1)
            plt.title('Training Data')
            plt.plot(time_domain, self.train_df.drop(['A_AVE'],axis=1).values, marker='.')
            plt.subplot(212)
            plt.title('Training Data ACC')
            plt.plot(time_domain, self.train_df[["A_AVE"]], marker='.')
            plt.show()
        elif flag == TESTING:
            time_domain = np.arange(0.0,len(self.test_df),1)
            plt.title('Testing Data')
            plt.plot(time_domain, self.test_df.drop(['A_AVE'],axis=1).values, marker='.')
            plt.subplot(212)
            plt.title('Testing Data ACC')
            plt.plot(time_domain, self.test_df[["A_AVE"]], marker='.')
            plt.show()
        else:
            print("ERROR BAD DATA FLAG")
    
    def train(self,pd_dataframe,learning_rate,epochs,minibatch_size):
        begin = time.time()
        X_data = pd_dataframe.drop(["LABEL"],axis=1)
        X = X_data.values
        Y_data = pd_dataframe.iloc[:,:1]
        Y = Y_data.values
        Y = np.array([labelMaker(i[0]) for i in Y])
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,shuffle=False)#, random_state=0)
        
        self.train_df = pd.DataFrame(X_train,columns=X_data.columns)
        self.test_df = pd.DataFrame(X_test,columns=X_data.columns)
        
        # [Plotting Train and Test Data at the stage of training]
        
        X_train = X_train.transpose()
        X_test = X_test.transpose()
        Y_train = Y_train.transpose()
        Y_test = Y_test.transpose()
        with self.graph.as_default():
            tf.set_random_seed(1)
            #ops.reset_default_graph()
            seed = 3
            (n_x, m) = X_train.shape
            n_y = Y_train.shape[0]
            costs = []
            serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
            feature_configs = {'x': tf.FixedLenFeature(shape=[6], dtype=tf.float32),}
            tf_example = tf.parse_example(serialized_tf_example, feature_configs)
            # [Create Placeholders of shape (n_x, n_y)]
            X, Y = create_placeholders(n_x, n_y)
            # [Initialize layer parameters]
            self.init_layers()
            # [Add Forward Propogation to the graph]
            final = self.forward_propagation(X)
            # [Adding cost function for the Adam optimizer to the graph]
            cost = function_cross_entropy(final, Y)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            init = tf.global_variables_initializer()
            with tf.Session(graph=self.graph) as sess:
                # [Session Initialization]
                sess.run(init)
                values, indices = tf.nn.top_k(final, 3)
                table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(3)]))
                prediction_classes = table.lookup(tf.to_int64(indices))

                builder = tf.saved_model.builder.SavedModelBuilder(self.path)
                train_writer = tf.summary.FileWriter(self.path+"/train", sess.graph)
                # [Training loop]
                with tqdm(total = epochs,desc = self.cli_name+" ",unit="epo") as pbar:
                    for epoch in range(epochs):
                        # [Calculate cost related to an epoch]
                        epoch_cost = 0
                        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                        seed = seed + 1
                        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
                        for minibatch in minibatches:
                            # [Select a minibatch]
                            (minibatch_X, minibatch_Y) = minibatch
                        
                            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                            epoch_cost += minibatch_cost / num_minibatches
                        
                        # [Print progress and the cost every epoch]
                        if self.print_cost == True and epoch % 1 == 0:
                            pbar.update(n=1)
                            print(" ~ Cost:",round(epoch_cost,4))
                        if self.print_cost == True and epoch % 1 == 0:
                            costs.append(epoch_cost)

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
                print(self.cli_name+" Time for training was",round(finish-begin,2),"seconds")

                print(final)
                # [Calculating the correct predictions]
                correct_prediction = tf.equal(tf.argmax(final), tf.argmax(Y))
                predictions_train = sess.run(tf.argmax(final),feed_dict={X: X_train})
                predictions_test = sess.run(tf.argmax(final),feed_dict={X: X_test})
                predictions = np.concatenate((predictions_train,predictions_test),axis=0)
                tf.summary.histogram("predictions", final)
                # [Calculate accuracy on the test set]
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print(self.cli_name+" Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
                print(self.cli_name+" Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return predictions
        
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
        layer_counter = 1
        for layer in self.structure:
            name_W = "W"+str(layer_counter)
            name_b = "b"+str(layer_counter)
            with tf.name_scope('weights'):
                self.layers[name_W] = tf.get_variable(name_W, [layer[0], layer[1]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            with tf.name_scope('biases'):
                self.layers[name_b] = tf.get_variable(name_b, [layer[0], 1], initializer = tf.zeros_initializer())
            layer_counter+=1
            
    def forward_propagation(self,X):
        Z = []
        A = []
        A.append(X)
        layer_counter = 1
        for layer in self.structure:
            if layer_counter == len(self.structure):
                # [Last layer - LINEAR]
                Z.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], A[layer_counter-1]), self.layers["b"+str(layer_counter)]))
                A.append(Z[layer_counter-1])
            else:
                # [All hidden layers - TANH]
                Z.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], A[layer_counter-1]), self.layers["b"+str(layer_counter)]))
                A.append(tf.nn.tanh(Z[layer_counter-1]))
            layer_counter+=1
        # [Return last layer]

        return A[-1]

    def inference(self):#,pd_dataframe):
        #self.layers_import(self.path+"/exit.json")
        #print(self.structure)
        
        with tf.Session(graph=self.graph) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], self.path)
            W1 = self.graph.get_tensor_by_name("W1:0")
            print(W1)
        print(":ok")
