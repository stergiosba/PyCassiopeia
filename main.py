from include import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
import os
plt.style.use('ggplot')

measurement = 'RE_HMI_ECUA_020115'
window_features = [2000,500] #Window Size, Window Step
saved_list = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']
#saved_list = ['N_MAX','N_MIN','N_AVE','N_IN','N_OUT','LABEL']
fit_df = dataProcess('./data.csv',os.getcwd()+'/saved',saved_list,measurement,window_features)
class_names = ['Deceleration', 'Acceleration', 'Steady']

#fit_df = _fit_df.sample(frac=1).reset_index(drop=True)
#fit_data = fit_df["LABEL"].values
X_data = fit_df.drop(["LABEL"],axis=1)
X = X_data.values
Y_data = fit_df.iloc[:,:1]
Y = Y_data.values
Y = np.array([labelMaker(i[0]) for i in Y])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_train = X_train.transpose()
X_test = X_test.transpose()
Y_train = Y_train.transpose()
Y_test = Y_test.transpose()

num__input_features = X_train.shape[0]
num_output_features = Y_train.shape[0]
'''
with tf.Session() as sess:
    X, Y = create_placeholders(num__input_features, num_output_features)
    parameters = initialize_parameters(num__input_features)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
'''
#parameters = CAS_NN(X_train, Y_train, X_test, Y_test, minibatch_size = 2)

'''
model = keras.Sequential([
    keras.layers.Dense(12, activation=tf.nn.tanh),
    keras.layers.Dense(12, activation=tf.nn.tanh),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1000)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
tf.keras.models.save_model(model,"model.model")
print("Saved model to disk")
'''
