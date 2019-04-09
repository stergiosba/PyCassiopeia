# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:32:50 2019

@author: stergios
"""
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
#from tensorflow import set_random_seed
import numpy as np
import math

def windowUnits(max_length,size,step):
    position = 0
    counter = 0
    for i in range(0,max_length,step):
        if i==0:
            position+=size
        else:
            position+=step
            if position>max_length:
                break
        counter+=1
    return counter

def nullDf(df,_measurements):
    for measurement in _measurements:
        print("~$> Calculated",df[df[measurement].isnull()].size,"missing datapoints.")
        if df[df[measurement].isnull()].size >10000:
            print("~$> Dataframe is missing to many data")
        if 0 in df[df[measurement].isnull()].index:
            df[measurement][0] = 0
        for i in df[df[measurement].isnull()].index:
            df[measurement][i] = df[measurement][i-1]
        print("~$> All missing datapoints have been restored")
    return df
    
def normDf(df):
    print("~$> Normalizing Dataframe")
    d = {}
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.values)
    for i in range(len(df.columns)):
        d[df.columns[i]]=x_scaled[:,i]
    print("~$> Dataframe has been normalized")
    return pd.DataFrame(data=d)
    
def labelMaker(val):
    if val == 0:
        return [1, 0, 0]
    elif val == 1:
        return [0, 1, 0]
    else: 
        return [0, 0, 1]

# n_x = num__input_features
# n_y = expected output (num classes)
def create_placeholders(n_x, n_y):
    X1 = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y1 = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X1, Y1


def initialize_parameters(num_input_features):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [num_hidden_layer, num_input_features]
                        b1 : [num_hidden_layer, 1]
                        W2 : [num_output_layer_1, num_hidden_layer]
                        b2 : [num_output_layer_1, 1]
                        W3 : [num_output_layer_2, num_output_layer_1]
                        b3 : [num_output_layer_2, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """ 
    tf.set_random_seed(1)           
    W1 = tf.get_variable("W1", [10, num_input_features], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [10, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [150, 10], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [150, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [3, 150], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [3, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters
    
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: 
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters 
    "W1", "b1", "W2", "b2", "W3", "b3"
    the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.tanh(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.tanh(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    '''
     It is important to note that the forward propagation stops at z3. 
     The reason is that in tensorflow the last linear layer output is 
     given as input to the function computing the loss. 
     Therefore, you don't need a3!
    '''
    return Z3
    
def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (3, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
   
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # The newer recommended function in Tensor flow
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return cost
    
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

