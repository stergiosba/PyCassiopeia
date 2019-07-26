import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
seed_offset = 0
samples_path = "samples/"
#full_df = pd.read_pickle('templates_Euclidean barycenter_clusters7.pkl').T
full_df = pd.read_pickle('swap_corrected_templates_soft_dtw_clusters7_gamma1.pkl')
full_df = full_df.head(1600)
samples = 10

if not os.path.exists(samples_path):
    os.makedirs(samples_path)
for sample in range(samples):
    np.random.seed(seed_offset+sample)
    sections = np.random.randint(20,25, size=1, dtype=int)[0]
    random_sections = np.random.randint(len(full_df.columns), size=sections, dtype=int)
    frames = []
    for i,section in enumerate(random_sections):
        sectionrange = np.random.randint(50,200, size=1, dtype=int)[0]
        section_df = full_df[[section]]
        random_low = np.random.randint(len(section_df)-sectionrange+1, size=1)[0]
        section_df = section_df[random_low:random_low+sectionrange]
        frames.append(section_df)

    sample_df = pd.concat(frames, ignore_index=True)
    sample_df = sample_df.stack().reset_index()
    sample_df = sample_df.drop(sample_df.columns[0], axis=1)
    sample_df.columns = ['LABEL','E_REV']
    sample_df['E_REV'] = sample_df['E_REV']
    sample_df.to_csv(samples_path+'sample_'+str(sample)+'.csv',index=False)

