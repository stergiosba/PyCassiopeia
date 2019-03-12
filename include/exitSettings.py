#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:24:01 2019

@author: stergios
"""

import os

def flagControl(___window_features):
    lines = [
            "[CASS_NN_SETTINGS]",
            "[window_size]: "+str(___window_features[0]),
            "[window_step]: "+str(___window_features[1])]
    with open("cassnn.settitgs.txt","w") as file:
        for line in lines:
            file.write(line+'\n')
    print("~$> Exporting Setting File")

def exitSettings(_save_path, _window_features, win_max, train_acc, test_acc):
    settings_file = "cassnn.info.txt"
    exit_path = _save_path+"/"+settings_file
    lines = [
            "[CASS_NN_SETTINGS]",
            "[window_size]: "+str(_window_features[0]),
            "[window_step]: "+str(_window_features[1]),
            "[window_max_num]: "+str(win_max),
            "[EVALUATION]",
            "[train_acc]: "+str(train_acc),
            "[test_acc]: "+str(test_acc),
            ]
    with open(exit_path,"w") as file:
        for line in lines:
            file.write(line+'\n')
    print("~$> Exporting Information File")
    
    return exit_path