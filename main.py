import time
from include import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import save_model
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

plt.style.use('ggplot')

measurement = 'RE_HMI_ECUA_020115'
window_features = [10,5] #Window Size, Window Step
saved_list = ['N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE','LABEL']
fit_df , fit_labels = dataProcess("./data.csv",saved_list,measurement,window_features)
class_names = ['Deceleration', 'Acceleration', 'Steady']

fit_df.shape
model = keras.Sequential([
    keras.layers.Dense(12, activation=tf.nn.tanh),
    keras.layers.Dense(12, activation=tf.nn.tanh),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(fit_df, fit_labels, epochs=100)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
tf.keras.models.save_model(model,"model.model")
print("Saved model to disk")
