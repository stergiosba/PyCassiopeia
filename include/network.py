# -*- coding: utf-8 -*-
"""
Code for the Neural Network Class.
Contains all calculation features and I/O Controls.

@author: stergios
"""
import os
import time
import numpy as np
import json
import codecs
from .utils import *
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
#tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create grap

class Network():
    def __init__(self,structure=np.array([]),export=False):
        print("~$> Creating Network")
        self.structure = structure
        self.print_cost = True
        self.layers = {}
        if export:
            print("~$> Exporting Network Information")
            self.layers_export()
            
    def layers_export(self):
        network_dir = os.getcwd()+"/network"
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)
        with open(network_dir+"/info.txt","w") as file:
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
        file_path = network_dir+"/exit.json" ## your path variable
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)

    def layers_import(self,json_path):
        obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        self.structure = np.array(b_new["network_structure"])
    
    def train(self,pd_dataframe,learning_rate,epochs,minibatch_size):
        begin = time.time()
        X_data = pd_dataframe.drop(["LABEL"],axis=1)
        X_data.plot()
        X = X_data.values
        Y_data = pd_dataframe.iloc[:,:1]
        Y = Y_data.values
        Y = np.array([labelMaker(i[0]) for i in Y])
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,shuffle=False)#, random_state=0)
        pd.DataFrame(X_train).plot()
        pd.DataFrame(X_test).plot()
        X_train = X_train.transpose()
        X_test = X_test.transpose()
        Y_train = Y_train.transpose()
        Y_test = Y_test.transpose()
        
        ops.reset_default_graph()
        tf.set_random_seed(1)
        seed = 3
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []
        # Create Placeholders of shape (n_x, n_y)
        X, Y = create_placeholders(n_x, n_y)
        # Initialize parameters
        self.init_layers()
        final = self.forward_propagation(X)
        cost = compute_cost(final, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
            # Do the training loop
            for epoch in range(epochs):
    
                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
    
                for minibatch in minibatches:
    
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    ### END CODE HERE ###
                    
                    epoch_cost += minibatch_cost / num_minibatches
    
                # Print the cost every epoch
                if self.print_cost == True and epoch % 1 == 0:
                    print("~/CassNN$> Epoch:",epoch,"/",epochs,"~ Cost:",round(epoch_cost,4))
                if self.print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
            # lets save the parameters in a variable
            self.layers = sess.run(self.layers)
            print("~/CassNN$> Finished Neural Training!")
            finish = time.time()
            print("~/CassNN$> Time for training was",round(finish-begin,2),"seconds")
            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(final), tf.argmax(Y))
            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            #train_acc = accuracy.eval({X: X_train, Y: Y_train})
            #test_acc = accuracy.eval({X: X_test, Y: Y_test})
            
            print("~/CassNN$> Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print("~/CassNN$> Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        #print(final)
        return final
        
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
            self.layers[name_W] = tf.get_variable(name_W, [layer[0], layer[1]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            self.layers[name_b] = tf.get_variable(name_b, [layer[0], 1], initializer = tf.zeros_initializer())
            layer_counter+=1
            
    def forward_propagation(self,X):
        Z = []
        A = []
        A.append(X)
        layer_counter = 1
        for layer in self.structure:
            if layer_counter == len(self.structure):
                #Linear Last Layer
                Z.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], A[layer_counter-1]), self.layers["b"+str(layer_counter)]))
                A.append(Z[layer_counter-1])
            else:
                #All hidden layers
                Z.append(tf.add(tf.matmul(self.layers["W"+str(layer_counter)], A[layer_counter-1]), self.layers["b"+str(layer_counter)]))
                A.append(tf.nn.tanh(Z[layer_counter-1]))
            layer_counter+=1
        
        return A[-1]
        
                
                
                
        