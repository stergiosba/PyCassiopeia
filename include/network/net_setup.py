#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:32:40 2019
@author: Khanax
"""
import math
import time
import statistics as stats

import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils import EPS,windowUnits
import include.network.net_constants as netco

def labelMaker(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
        
# n_x = num__input_features
# n_y = expected output (num classes)
def create_placeholders(n_x=None, n_y=None):
    X1 = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y1 = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X1, Y1

def SOFTMAX_CROSS_ENTROPY(final_layer,Y):
    """
    Computes the softmax cross entropy cost.
    """
    logits = tf.transpose(final_layer)
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

SCE=SOFTMAX_CROSS_ENTROPY

def MEAN_SQR_ERROR(final_layer,Y):
    return tf.losses.mean_squared_error(final_layer,Y)

MSE = MEAN_SQR_ERROR
'''
def network_cost_function(final_layer, Y, loss):
    """
    Computes the cost in softmax cross entropy or RMSE or MSE fashion
    
    Arguments:
    final_layer -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as final_layer
    
    Returns:
    cost - Tensor of the cost function
    """
    if loss == netco.SOFTMAX_CROSS_ENTROPY or loss=='SCE':
        logits = tf.transpose(final_layer)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    elif loss == netco.REDUCED_MEAN_SQR_ERROR or loss=='RMSE':
        cost = tf.reduce_mean(tf.square(final_layer - Y))
    elif loss == netco.MEAN_SQR_ERROR or loss=='MSE':
        cost = tf.losses.mean_squared_error(Y,final_layer)
    
    return cost
'''

def init_Weight(shape,name):
    initializer = tf.initializers.glorot_uniform()
    return tf.Variable(initializer(shape),name=name,trainable=True, dtype=tf.float32)

def init_Bias(shape,name):
    initializer = tf.initializers.zeros()
    return tf.Variable(initializer(shape),name=name,trainable=True, dtype=tf.float32)

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
    
    m = X.shape[1]# number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size)# number of mini batches of size mini_batch_size in your partitionning
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