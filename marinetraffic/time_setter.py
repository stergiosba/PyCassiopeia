import os
import time
from statistics import mean

import pandas as pd
import matplotlib.pyplot as plt
from utils import Point,LI

filter_path = 'filter'
final_path = 'final'
if not os.path.exists(final_path): os.makedirs(final_path)

for file in os.listdir(filter_path):
    ship_id = file.split('.')[0]
    print('Calculating for: '+ship_id)
    ship_df = pd.read_csv(os.path.join(filter_path,file),engine='python')
    ship_df = ship_df.astype({'TIMESTAMP': int})
    speeds = []
    for index, row in ship_df[['SPEED','TIMESTAMP']].iterrows():
        if index==0:
            past_row = row
        else:
            start = Point(past_row['TIMESTAMP'],past_row['SPEED'])
            end = Point(row['TIMESTAMP'],row['SPEED'])
            past_row = row
            for x in range(int(start.x),int(end.x)):
                    y = LI(start,end,x)
                    speeds.append(y)
            speeds.append(round(end.y,4))
    out_df = pd.DataFrame({'SPEED':speeds})
    out_df.to_csv(os.path.join(final_path,file),index=False)
