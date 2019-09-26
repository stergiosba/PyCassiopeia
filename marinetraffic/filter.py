import os
import time
from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import normalizeDataFrame

filter_path = 'filter'
if not os.path.exists(filter_path): os.makedirs(filter_path)
df = pd.read_csv('marinetraffic.csv')

for ship_id in df['SHIP_ID'].unique():
    ship_df = df[df['SHIP_ID'] == ship_id]
    ship_df = ship_df.reset_index(drop=True)
    ship_df = ship_df.drop('SHIP_ID',axis=1)
    timestamps = ship_df['TIMESTAMP']
    ship_df = ship_df.drop('TIMESTAMP',axis=1)
    ship_df,_ = normalizeDataFrame(ship_df)
    difs = []
    cumulative = []
    cumulative_time = 0
    for index,timestamp in enumerate(timestamps):
        date_present = time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        date_present_in_seconds = time.mktime(date_present)
        if index == 0:
            time_passed = 0
        else:
            time_passed = date_present_in_seconds-date_past_in_seconds
        cumulative_time+=time_passed
        cumulative.append(cumulative_time)
        difs.append(time_passed)
        date_past = date_present
        date_past_in_seconds = time.mktime(date_past)

    if mean(difs)<400:
        ship_df['TIMESTAMP'] = cumulative
        ship_df.to_csv(filter_path+"/ship_"+str(ship_id)+".csv",index=False)
        ship_df.plot()
        plt.title(ship_id)
plt.show()


