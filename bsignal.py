import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_name = 'E_REV'
list_parameters = [[0.15,0.18,20],[0.18,0.22,200],[0.22,0.25,500],[0.40,0.42,100]]
df = pd.DataFrame(columns=[column_name])

for parameters in list_parameters:
    b = parameters[0]
    a = parameters[1]
    values = parameters[2]
    temp = pd.DataFrame(columns=[column_name])
    temp[column_name] = (b - a) * np.random.random_sample(values) + a
    df = pd.concat([df,temp],ignore_index=True)

df.to_csv("visual1.csv",index=False)