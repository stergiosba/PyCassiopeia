import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
seed_offset = 88771
samples_path = "samples/"
#full_df = pd.read_pickle('templates_Euclidean barycenter_clusters7.pkl').T
#full_df = pd.read_csv('swap_corrected_templates_soft_dtw_clusters7_gamma1.csv')
full_df = pd.read_csv('swap_corrected_templates_soft_dtw_clusters7_gamma1_5.csv')
#full_df = full_df.head(1500)

class Point():
    def __init__(self,x,y):
        self.x=x
        self.y=y
    
    def show(self):
        print(self.x,"-",self.y)

def LI(p1,p2,x):
    lamda = (p2.y-p1.y)/(p2.x-p1.x)
    return round(p1.y+lamda*(x-p1.x),4)

samples = 5

if not os.path.exists(samples_path):
    os.makedirs(samples_path)

for sample in range(samples):
    time = 0
    np.random.seed(seed_offset+sample)
    sections = np.random.randint(5,6, size=1, dtype=int)[0]
    random_sections = np.random.randint(len(full_df.columns), size=sections, dtype=int)
    #random_sections = np.random.randint(2, size=sections, dtype=int)
    frames = []
    for i,section in enumerate(random_sections):
        sectionrange = np.random.randint(150,201, size=1, dtype=int)[0]
        section_df = full_df[[str(section)]]
        random_low = np.random.randint(0,len(section_df)-sectionrange+1, size=1)[0]
        print(random_low)
        #print(section,random_low)
        section_df = section_df[random_low:random_low+sectionrange]
        last_df = section_df
        #first_point = Point(len(section_df),section_df.iloc[0][0])
        #first_point.show()
        #last_point = Point(len(section_df),section_df.iloc[-1][0])
        #last_point.show()
        time+=len(section_df)
        frames.append(section_df)

    print(time)
        

    sample_df = pd.concat(frames, ignore_index=True)
    sample_df = sample_df.stack().reset_index()
    sample_df = sample_df.drop(sample_df.columns[0], axis=1)
    sample_df.columns = ['LABEL','E_REV']
    sample_df['E_REV'] = sample_df['E_REV']
    sample_df.to_csv(samples_path+'sample_'+str(sample)+'.csv',index=False)
    sample_df.plot()
    plt.ylim(0,1)
    plt.show()
