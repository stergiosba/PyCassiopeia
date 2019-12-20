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

    def train_step(self,I,O):
        with tf.GradientTape as tape:
            current_loss = self.loss(I, O)
            grads = tape.gradient(current_loss, weights)
            self.optimizer.apply_gradients(zip(grads, weights))
            print(tf.reduce_mean(current_loss))

    
    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        '''Network train function. \n
        |-> Arguments(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)
        '''
        self.function = netco.TRAINING
        # On a new training process of the same network we check for new build/train up to 10
        costs_flag = False
        self.build_control()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        graph = tf.Graph()
        self.graph = graph
        begin = time.time()
        X_data = self.normalize(data)
        X_data = X_data.astype('float32')
        if self.edition == netco.TREND or self.edition == netco.CYCLES:
            X_data = X_data.astype({'LABEL': int})
        Y_data = X_data.pop('LABEL')
        X_train, X_test = train_test_split(X_data, test_size=test_size)
        Y_train, Y_test = train_test_split(Y_data, test_size=test_size)
        DATASET_TRAIN = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
        DATASET_TEST = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values))
        
        if shuffle=='True':
            shuffle = True
            DATASET_TRAIN = DATASET_TRAIN.shuffle(1024).batch(minibatch_size)
            print(self.cli_name+" Splitting Dataset with size "+str(test_size)+". Shuffling: Enabled!")
        else:
            shuffle = False
            DATASET_TRAIN = DATASET_TRAIN.batch(minibatch_size)
            print(self.cli_name+" Splitting Dataset with size "+str(test_size)+". Shuffling: Disabled!")
        
        #X_train = X_train.reset_index(drop=True)
        #X_test = X_test.reset_index(drop=True)
        #self.train_df = X_train
        #self.test_df = X_test
        '''
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
        '''
        # [Plotting Train and Test Data at the stage of training]
        #X_train = X_train.drop(["LABEL"],axis=1)
        #X_test = X_test.drop(["LABEL"],axis=1)
        X_train = X_train.values.transpose()
        X_test = X_test.values.transpose()
        
        tf.random.set_seed(1)
        #tf.compat.v1.reset_default_graph()
        seed = 3
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        self.costs_train = []
        self.costs_test = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.init_layers()
        
        # [Add Forward Propogation to the graph]
        final = tf.identity(self.forward_propagation(X_train),name="Final")
        final = final.numpy()
        print(final)
        with tqdm(total=num_minibatches,desc = "Progress:  ",unit="mini") as minibar:
            for epoch in range(epochs):
                pass
            


        #print(final.numpy())
        print("EDW")
        '''
        with self.graph.as_default():
            #serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
            #feature_configs = {'x': tf.io.FixedLenFeature(shape=[n_y], dtype=tf.float32),}
            #tf_example = tf.io.parse_example(serialized_tf_example, feature_configs)
            # [Create Placeholders of shape (n_x, n_y)]
            #X, Y = nets.create_placeholders(n_x, n_y)
            # [Initialize layer parameters]
            self.init_layers()
            
            # [Add Forward Propogation to the graph]
            final = tf.identity(self.forward_propagation(X_train),name="Final")
            print(self.Weights['W1'].numpy())
            #print(final.numpy())
            print("EDW")
            # [Adding cost function for the Adam optimizer to the graph]
            #cost = nets.softmax_cross_entropy(final, Y)
            #cost = nets.network_cost_function(final, Y, loss)
            #values, indices = tf.nn.top_k(final, n_y)
            #table = tf.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(n_y)]))
            #prediction_classes = table.lookup(tf.to_int64(indices))
            #prediction_classes = table.lookup(tf.cast(indices,tf.int64))
            #builder = tf.saved_model.builder.SavedModelBuilder(self.build_path)
            #train_writer = tf.summary.FileWriter(self.version_path+"/train", sess.graph)
            
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
                    self.costs_test.append(round(sess.run(cost,feed_dict={X: X_test, Y:Y_test}),4))
                    self.costs_train.append(round(epoch_cost,4))
                    
                    if self.edition == netco.TREND or self.edition == netco.CYCLES:
                        correct_prediction = tf.equal(output_function(final), output_function(Y))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                        train_acc = accuracy.eval({X: X_train, Y: Y_train})
                        test_acc = accuracy.eval({X: X_test, Y: Y_test})
                    else:
                        predictions_train = sess.run(output_function(final),feed_dict={X: X_train})
                        predictions_test = sess.run(output_function(final),feed_dict={X: X_test})
                        train_acc = mean_absolute_error(Y_train, predictions_train)
                        test_acc = mean_absolute_error(Y_test, predictions_test)
                    self.accuracy_train.append(round(train_acc,4))
                    self.accuracy_test.append(round(test_acc,4))

                    pbar.set_description('Cost: {} '.format(round(epoch_cost,4)))
                    pbar.refresh() # to show immediately the update
                    pbar.update(n=1)
                    if epoch % 50 == 0:
                        print('')
            # [Saving the parameters in a variable]
            self.layers = sess.run(self.layers)
            print(self.cli_name,"Finished Neural Training!")
            
            classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

            classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    tf.saved_model.CLASSIFY_INPUTS:classification_inputs},
                outputs={
                    tf.saved_model.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
                    tf.saved_model.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores},
                method_name=tf.saved_model.CLASSIFY_METHOD_NAME))
            
            tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(final)
            
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
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
                main_op=tf.tables_initializer(),
                strip_default_attrs=True)

            builder.save()
            
            
            finish = time.time()
            self.training_time = round(finish-begin,2)
            print(self.cli_name,"Time for training was",self.training_time,"seconds")

            # [Calculating the correct predictions]
            
            self.predictions_train = sess.run(output_function(final),feed_dict={X: X_train}).T
            self.predictions_test = sess.run(output_function(final),feed_dict={X: X_test}).T
            self.predictions = np.concatenate((self.predictions_train,self.predictions_test),axis=0)
            tf.summary.histogram("predictions", final)
            print(self.cli_name+" Train Accuracy:", self.accuracy_train[-1])
            print(self.cli_name+" Test Accuracy:",self.accuracy_test[-1])

            saver.save(sess, self.version_path+'/model.ckpt')

            X_ordered= X_data.drop(["LABEL"],axis=1)
            X_ordered = X_ordered.values.transpose()
            self.ordered_predictions = sess.run(output_function(final),feed_dict={X: X_ordered}).T
            self.labels = X_data['LABEL']
            
            self.network_training_summary_report()
              
        return self.predictions
        '''
        return 0
        
        
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
        self.Weights = {}
        self.Biases = {}
        for layer_counter,layer in enumerate(self.structure,start=0):
            name_W = 'W'+str(layer_counter)
            name_b = 'b'+str(layer_counter)
            self.Weights[name_W] = nets.init_Weight([int(layer[0]), int(layer[1])],name_W)
            self.Biases[name_b] = nets.init_Bias([int(layer[0]), 1],name_b)
            
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
                A.append(tf.add(tf.matmul(self.Weights["W"+str(layer_counter)], Z[layer_counter]), self.Biases["b"+str(layer_counter)]))
                Z.append(A[layer_counter])
            elif activation_function == netco.TANH:
                # [TANH layer]
                A.append(tf.add(tf.matmul(self.Weights["W"+str(layer_counter)], Z[layer_counter]), self.Biases["b"+str(layer_counter)]))
                Z.append(tf.nn.tanh(A[layer_counter]))
            elif activation_function == netco.SIGMOID:
                # [SIGMOID layer]
                A.append(tf.add(tf.matmul(self.Weights["W"+str(layer_counter)], Z[layer_counter]), self.Biases["b"+str(layer_counter)]))
                Z.append(tf.nn.sigmoid(A[layer_counter]))
            elif activation_function == netco.RELU:
                # [RELU layer]
                A.append(tf.add(tf.matmul(self.Weights["W"+str(layer_counter)], Z[layer_counter]), self.Biases["b"+str(layer_counter)]))
                Z.append(tf.nn.relu(A[layer_counter]))
            else:
                raise NameError('There is an error with the activation functions in network_structure.json. Check the file and retry.')
        # [Return last layer]
        return Z[-1]


    def normalize(self,data):
        '''Normalize data and save normalizers for later inference'''
        normalizers_file = "data_normalizers.json"
        save_path = os.path.join(self.version_path,normalizers_file)
        if self.function == netco.TRAINING:
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

    def network_training_summary_report(self):
        '''Save a training summary report for the trained network'''
        settings_file = self.name+".training_summary.txt"
        layers_matlab_file = self.name+".layers.mat"
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
                'Training Accuracy: ' + str(self.accuracy_train[-1]) + "%",
                'Testing Accuracy: ' + str(self.accuracy_test[-1]) + "%",
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

class NNClassifier(Network):
    def __init__(self,edition,flag,base_name,root_path,features):
        Network.__init__(self,edition,flag,base_name,root_path,features)
        self.get_general()

    def get_general(self):
        self.loss = netco.SOFTMAX_CROSS_ENTROPY
        self.output_function = tf.argmax
        self.optimizer = tf.optimizers.Adam(
                learning_rate=0.001,beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
            )

    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        super().train(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)
        self.plot_accuracy()
        self.plot_costs()

    def plot_costs(self):
        '''Timeplots of costs'''
        t = np.arange(0.0, self.epochs, 1)
        fig, ax = plt.subplots()
        ax.plot(t, self.costs_train, label='Train Error')
        ax.plot(t, self.costs_test, label='Test Error')
        ax.set(xlabel='Epochs', ylabel='Softmax Cross Entropy',title='Training and Testing Errors for '+self.edition)
        ax.grid()
        plt.legend()
        fig.savefig(self.build_path+'/costs.png',dpi=800)
        plt.show()

    def plot_accuracy(self):
        '''Timeplots of accuracy'''
        t = np.arange(0.0, self.epochs, 1)
        fig, ax = plt.subplots()
        ax.plot(t, self.accuracy_train, label='Train Accuracy')
        ax.plot(t, self.accuracy_test, label='Test Accuracy')
        ax.set(xlabel='Epochs', ylabel='Accuracy',title='Training and Testing Accuracy for '+self.edition)
        ax.grid()
        plt.legend()
        plt.ylim(round_down(min(self.accuracy_test)-0.05,2),1)
        fig.savefig(self.build_path+'/accuracy.png',dpi=800)
        plt.show()

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
        self.get_general()
        
    def get_general(self):
        self.loss = netco.MEAN_SQR_ERROR
        self.output_function = tf.identity
        self.optimizer = tf.optimizers.Adam(
                learning_rate=0.001,beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
            )

    def train(self,data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs):
        cycle_full = self.root_path.split('/')[-1]
        cycle = cycle_full.split('_')[-1]
        cycle = str(int(cycle)+1)
        
        super().train(data,epochs,learning_rate,minibatch_size,shuffle,test_size,outputs)
        self.plot_trained(cycle)
        self.plot_costs(cycle)
        self.plot_predictions(cycle)
        plt.close('all')
        
        #plt.show()

    def plot_trained(self,cycle):
        fig, ax = plt.subplots()
        t = np.arange(0.0, len(self.labels), 1)
        ax.plot(t, self.labels, label='NMPC',color='blue')
        ax.plot(t, self.ordered_predictions, label='NN',color='red')
        ax.set(xlabel='Time [Secs]', ylabel='Torque [Nm]',title=self.edition+' Torque (Cycle '+cycle+')')
        ax.grid()
        plt.legend()
        fig.savefig(self.build_path+'/trained_model_cycle_'+cycle+'.png',dpi=800)
        frame = pd.DataFrame()
        frame['LABEL'] = self.labels
        frame['PREDICTION'] = self.ordered_predictions
        frame.to_csv(os.path.join(self.build_path,'results_'+self.edition+'_'+cycle+'.csv'),index=False)


    def plot_predictions(self,cycle):
        fig, ax = plt.subplots()
        ax.scatter(self.test_df['LABEL'], self.predictions_test,color='red')
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
        
        fig.savefig(self.build_path+'/predictions_cycle_'+cycle+'.png',dpi=800)

    def plot_costs(self,cycle):
        '''Timeplots of costs'''
        t = np.arange(0.0, self.epochs, 1)
        fig, ax = plt.subplots()
        ax.plot(t, self.costs_train, label='Train Error')
        ax.plot(t, self.costs_test, label='Test Error')
        ax.set(xlabel='Epochs', ylabel='Mean Absolute Error (Torque)',title=self.edition+' Training and Testing Errors (Cycle '+cycle+')')
        ax.grid()
        plt.legend()
        fig.savefig(self.build_path+'/costs_cycle_'+cycle+'.png',dpi=800)

        

