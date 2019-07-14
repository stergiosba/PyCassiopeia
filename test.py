#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:38:34 2019

@author: stergios
"""
import os
import matplotlib.pyplot as plt
from include.utils import cleanTrendData
measurements = ['RE_HMI_ECUA_020115']#"PR_air_scav","RE_gov_idx_meas"]

indeces = [0,14,32,69]
indeces = [9,69]
indeces = [9,14,32,69]
indeces = [9,14,33,92,15,73,7,66,93,76,91,69]
full_df = cleanTrendData(os.getcwd()+"/data",measurements,indeces)

full_df.plot()
plt.show()