# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:07:01 2019

@author: stergios
"""
from .utils import *
import time
from tensorflow.python.framework import ops
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

def CAS_NN(X_data,X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 20, minibatch_size = 32, print_cost = True):
    begin = time.time()
    print("~/CassNN$> Starting Neural Training!")
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x)
    print(parameters["W1"])
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    cost_tf = tf.get_variable('cost', shape=[], initializer=tf.truncated_normal_initializer(mean=-1, stddev=0))
    cost_summary = tf.summary.scalar(name='Cost_summary', tensor=cost_tf)
    
    w1_summary = tf.summary.histogram('W1_summary', parameters["W1"])
    w2_summary = tf.summary.histogram('W2_summary', parameters["W2"])
    w3_summary = tf.summary.histogram('W3_summary', parameters["W3"])
    #z3_summary = tf.summary.histogram('Z3_summary', Z3)
    merged = tf.summary.merge_all()
    # Initialize all the variables
    init = tf.global_variables_initializer()
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("output", sess.graph)
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

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
            summary = sess.run(merged)
            writer.add_summary(summary, epoch)


            # Print the cost every epoch
            if print_cost == True and epoch % 1 == 0:
                print("~/CassNN$> Epoch:",epoch,"/",num_epochs,"~ Cost:",round(epoch_cost,4))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
            #if 
            #    [==============================]
            #writer.add_summary(cost_summary, epoch)
        # plot the cost
        """
        costs_ax = plt.plot()
        costs_ax.plot(np.squeeze(costs))
        costs_ax.ylabel('cost')
        costs_ax.xlabel('iterations (per tens)')
        costs_ax.title("Learning rate =" + str(learning_rate))
        plt.show()
        """

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("~/CassNN$> Finished Neural Training!")
        finish = time.time()
        print("~/CassNN$> Time for training was",round(finish-begin,2),"seconds")
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        predictions = sess.run(tf.argmax(Z3),feed_dict={X: X_data})
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_acc = accuracy.eval({X: X_train, Y: Y_train})
        test_acc = accuracy.eval({X: X_test, Y: Y_test})
        
        print("~/CassNN$> Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("~/CassNN$> Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        writer.close()
        return (parameters,train_acc,test_acc,predictions)