import time
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

print(50*"-")
print("~$> Initializing")
f_fullpath = "./data1.csv"
full_df = pd.read_csv(f_fullpath)
measurement = 'RE_HMI_ECUA_020115'
full_df = (full_df[[measurement]])/full_df[measurement].max()
#full_df.plot()
print("~$> Loading the dataset from " +f_fullpath)
print("~$> Calculated",full_df[full_df[measurement].isnull()].size,"missing datapoints.")
for i in full_df[full_df[measurement].isnull()].index:
    full_df[measurement][i] = 0
print("~$> All missing datapoints have been restored")
print(50*"-")

saved_list = ['N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE','LABEL']
fit_df = pd.DataFrame(columns = saved_list)
#data_size = 6000
data_size = full_df.shape[0]
#data_size = 500
print("~$> You have chosen to keep",data_size,"seconds.")
datas_df = full_df.head(data_size)
#datas_df = full_df[65000:66000]
#datas_df = datas_df.sample(frac=1).reset_index(drop=True)
segment_size = 100
print("~$> You have chosen to segmentize the data per",segment_size,"seconds.")
begin = time.time()
ave_df = pd.DataFrame()
window_count = 0
for segment in range(0,data_size,segment_size):
    segment_df = datas_df[segment:segment+segment_size]
    segment_df = segment_df.reset_index(drop=True)
    window_size = 20
    window_step = 5
    for window in range(segment_df.index.min(),segment_df.index.max(),window_step):
        window_df = segment_df[window:window+window_size]
        window_df = window_df.reset_index(drop=True)
        acc_list = []
        for rev in window_df.index:
            if rev==0:
                pass
            else:
               acc = window_df[measurement][rev]-window_df[measurement][rev-1]
               acc_list.append(acc)
        ave_df = ave_df.append(acc_list)
        if acc < -0.00005:
            label =0 #Decel
        elif acc > 0.00005:
            label =1 #Accel
        else:
            label =2#S
        del acc_list
        fit_df = fit_df.append({
            'N_MAX': window_df[measurement].max(),
            'N_MIN': window_df[measurement].min(),
            'N_AVE': window_df[measurement].mean(),
            'N_IN' : window_df[measurement][window_df.index.min()],
            'N_OUT': window_df[measurement][window_df.index.max()],
            'A_AVE': ave_df[0].mean(),
            'LABEL': label
            },ignore_index=True)
        '''
        window_df.plot()
        plt.plot(window_df[measurement].idxmax(),window_df[measurement].max(),'mo') 
        plt.plot(window_df[measurement].idxmin(),window_df[measurement].min(),'ro')
        plt.axhline(y=window_df[measurement].mean(), color='green', linestyle=':',linewidth=2, markersize=12)
        plt.plot(window_df[measurement].index.min(),window_df[measurement][window_df.index.min()],'y<')
        plt.plot(window_df[measurement].index.max(),window_df[measurement][window_df.index.max()],'g>')
        plt.text(window_df[measurement].index.min(), window_df[measurement].mean(), label, style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        '''
        
        window_count+=1
        if window_count%500==0:
            print("WINDOW",window_count,"END")
#fit_df = fit_df.transpose()
fit_data_size = fit_df.shape[0]
fit_df = fit_df.sample(frac=1).reset_index(drop=True)
#fit_df.sample(frac=1)
'''
train_df = fit_df[0:int(round(fit_data_size*0.7,0))]
train_labels = train_df["LABEL"].values
train_df=train_df.drop(["LABEL"],axis=1)
train_df = train_df.values
test_df = fit_df[int(round(fit_data_size*0.7,0)):fit_data_size]
test_labels = test_df["LABEL"].values
test_df=test_df.drop(["LABEL"],axis=1)
test_df = test_df.values
'''
fit_labels = fit_df["LABEL"].values
fit_df=fit_df.drop(["LABEL"],axis=1)
fit_df = fit_df.values
finish = time.time()
print("Total Windows Number is",window_count)
print("Elapsed time was",round(finish-begin,2),"seconds.")


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
model.fit(fit_df, fit_labels, epochs=100**1)
'''
test_loss, test_acc = model.evaluate(test_df, test_labels)
print('Test accuracy:', test_acc)
predictions = model.predict(test_df)
'''
'''
#print(predictions)
f = open("./Extruded/results.txt", "a")
for i in range(0,len(test_labels)):
    if np.argmax(predictions[i])!=test_labels[i]:
        lo = "no"
    else:
        lo = "yes"
    f.write(str(np.argmax(predictions[i]))+" / "+str(test_labels[i])+" --> "+lo)
    f.write("\n")
'''

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
tf.keras.models.save_model(model,"model.model")
print("Saved model to disk")

del datas_df, window_df
