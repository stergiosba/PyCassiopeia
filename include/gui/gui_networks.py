import os
import json
import codecs
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from .gui_toplevels import creationToplevelClassificationGUI, trainToplevelClassificationGUI
from .gui_toplevels import creationToplevelControlGUI, trainToplevelControlGUI, inferenceToplevelControlGUI
from .gui_main import cycles_windows_execute,trend_windows_execute
import include.network.network as net
import include.network.net_constants as netco

class classficiationNetworksFrame(tk.Frame):
    '''Frame for the classification networks.
    '''
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.grandparent = parent.parent
        net_canvas = tk.Canvas(self,bg=parent.theme.selected_bg,width = self.grandparent.width,height=self.grandparent.height)
        net_canvas.grid(row=0,column=0)
        
        # [Information Section]
        info_text = tk.Text(net_canvas,height=50,width=50)
        info_text.insert(tk.END,"In order to train a classifier follow these steps:\n\n1) Select Network Edition.\n\n2) Select a Window Size/Step Pair (model).\n\n3) Press Create to Define Network Architecture..\n\n4) Press Train and Select Training Parameters.\n\n5) Press Infer to use the trained network.")
        info_text.config(state=tk.DISABLED)
        info_text.place(x=1,y=1)

         # [Network Setup Section]
        label = tk.Label(net_canvas,text="Neural Network Settings",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        label.place(x=401,y=10)

        network_edition_l = tk.Label(net_canvas,text="Edition:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        network_edition_l.place(x=401,y=40)
        self.network_edition = ttk.Combobox(net_canvas,state="readonly")
        self.network_edition['values'] = (netco.CYCLES,netco.TREND)
        self.network_edition.current(0)
        self.network_edition.place(x=521,y=40)

        def callback(network_frame):
            if network_frame.network_edition.get() == netco.TREND:
                if not os.path.exists(os.path.join(os.getcwd(),"models/"+netco.TREND)):
                    network_frame.model['values'] = list([""])
                else:
                    network_frame.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.TREND))
            elif network_frame.network_edition.get() == netco.CYCLES:
                if not os.path.exists(os.path.join(os.getcwd(),"models/"+netco.CYCLES)):
                    network_frame.model['values'] = list([""])
                else:
                    network_frame.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.CYCLES))

        model_l = tk.Label(net_canvas,text="Model:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        model_l.place(x=401,y=60)
        self.model = ttk.Combobox(net_canvas,state="readonly")
        if self.network_edition.get() == netco.TREND:
            if not os.path.exists(os.path.join(os.getcwd(),"models/"+netco.TREND)):
                self.model['values'] = list([""])
            else:
                self.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.TREND))
        elif self.network_edition.get() == netco.CYCLES:
            if not os.path.exists(os.path.join(os.getcwd(),"models/"+netco.CYCLES)):
                self.model['values'] = list([""])
            else:
                self.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.CYCLES))

        self.model.current(0)
        self.model.place(x=521,y=60)
        self.network_edition.bind("<<ComboboxSelected>>", lambda _ : callback(self))

        # [Create Section]
        create_l = tk.Label(net_canvas,text="Creation Setup:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        create_l.place(x=401,y=100)
        create_button = tk.Button(net_canvas,text="Create",command=lambda:self.gen_create_box(),height = 2, width = 20)
        create_button['bg'] = parent.theme.bg
        create_button['fg'] = parent.theme.fg
        create_button.place(x=521,y=90)

        # [Train Section]
        train_l = tk.Label(net_canvas,text="Training Setup:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        train_l.place(x=401,y=150)
        train_button = tk.Button(net_canvas,text="Train",command=lambda:self.gen_train_box(),height = 2, width = 20)
        train_button['bg'] = parent.theme.bg
        train_button['fg'] = parent.theme.fg
        train_button.place(x=521,y=140)

        # [Visualization Section]
        ''''
        label = tk.Label(net_canvas,text="Visualization Settings",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        label.place(x=411,y=10)

        start_l = tk.Label(self,text="Start Point:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        start_l.place(x=411,y=40)
        start = tk.StringVar()
        start_entry = tk.Entry(self,textvariable=start)
        start_entry.place(x=511,y=40)

        end_l = tk.Label(self,text="End Point:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        end_l.place(x=411,y=60)
        end = tk.StringVar()
        end_entry = tk.Entry(self,textvariable=end)
        end_entry.place(x=511,y=60)

        visual_button = tk.Button(net_canvas,text="Visualize",command=lambda:self.network.export_predictions(int(start.get()),int(end.get())))
        visual_button['bg'] = parent.theme.bg
        visual_button['fg'] = parent.theme.fg
        visual_button.place(x=411,y=80)
        '''
        # [Inference Section]
        inference_l = tk.Label(net_canvas,text="Inference Setup:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        inference_l.place(x=401,y=200)
        inference_button = tk.Button(net_canvas,text="Infer",command=lambda:self.gen_inference_box(),height = 2, width = 20)
        inference_button['bg'] = parent.theme.bg
        inference_button['fg'] = parent.theme.fg
        inference_button.place(x=521,y=190)

    def gen_create_box(self):
        _win = creationToplevelClassificationGUI(self)

    def gen_train_box(self):
        _win = trainToplevelClassificationGUI(self)

    def gen_inference_box(self):
        _win = inferenceToplevelControlGUI(self)

class controlNetworksFrame(tk.Frame):
    '''Frame for the control networks.
    '''
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.grandparent = parent.parent
        net_canvas = tk.Canvas(self,bg=parent.theme.selected_bg,width = self.grandparent.width,height=self.grandparent.height)
        net_canvas.grid(row=0,column=0)

        # [Information Section]
        info_text = tk.Text(net_canvas,height=50,width=50)
        info_text.insert(tk.END,"In order to train a controller follow these steps:\n\n1) Select Add New Controller.\n\n2) Press Create to Define Controller Network Architecture.\n\n3) Press Train and Select Training Parameters.\n\n4) Press Infer to use the trained controller.")
        info_text.config(state=tk.DISABLED)
        info_text.place(x=1,y=1)

        # [Network Setup Section]
        label = tk.Label(net_canvas,text="Neural Network Settings",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        label.place(x=401,y=10)
        '''
        self.cycles_root_path = os.path.join(os.getcwd(),netco.CLASSIFIERS,netco.CYCLES)
        model_cycles_l = tk.Label(net_canvas,text="Cycles Model:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        model_cycles_l.place(x=1,y=40)
        self.model_cycles = ttk.Combobox(net_canvas,state="readonly",postcommand = self.callback_cycles)
        if not os.path.exists(self.cycles_root_path):
            self.model_cycles['values'] = list([""])
        else:
            self.model_cycles['values'] = list(os.listdir(self.cycles_root_path))
        self.model_cycles.current(0)
        self.model_cycles.place(x=111,y=40)

        self.trend_root_path = os.path.join(os.getcwd(),netco.CLASSIFIERS,netco.TREND)
        model_trend_l = tk.Label(net_canvas,text="Trend Model:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        model_trend_l.place(x=1,y=60)
        self.model_trend = ttk.Combobox(net_canvas,state="readonly",postcommand = self.callback_trend)
        if not os.path.exists(self.trend_root_path):
            self.model_trend['values'] = list([""])
        else:
            self.model_trend['values'] = list(os.listdir(self.trend_root_path))
        self.model_trend.current(0)
        self.model_trend.place(x=111,y=60)
        '''
        self.controllers_root_path = os.path.join(os.getcwd(),netco.CONTROLLERS)
        controller_l = tk.Label(net_canvas,text="Controller:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        controller_l.place(x=401,y=40)
        self.controller = ttk.Combobox(net_canvas,state="readonly",postcommand = self.callback_control)
        if not os.path.exists(self.controllers_root_path):
            self.controller['values'] = list(["Add new Controller"])
        else:
            list_values = list(os.listdir(self.controllers_root_path))
            list_values.append("Add new Controller")
            self.controller['values'] = list_values
        self.controller.current(0)
        self.controller.place(x=521,y=40)

        # [Create Section]
        create_l = tk.Label(net_canvas,text="Creation Setup:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        create_l.place(x=401,y=100)
        create_button = tk.Button(net_canvas,text="Create",command=lambda:self.gen_create_box(),height = 2, width = 20)
        create_button['bg'] = parent.theme.bg
        create_button['fg'] = parent.theme.fg
        create_button.place(x=521,y=90)

        # [Train Section]
        train_l = tk.Label(net_canvas,text="Training Setup:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        train_l.place(x=401,y=150)
        train_button = tk.Button(net_canvas,text="Train",command=lambda:self.gen_train_box(),height = 2, width = 20)
        train_button['bg'] = parent.theme.bg
        train_button['fg'] = parent.theme.fg
        train_button.place(x=521,y=140)

        # [Visualization Section]
        '''
        label = tk.Label(net_canvas,text="Visualization Settings",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        label.place(x=411,y=10)

        start_l = tk.Label(self,text="Start Point:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        start_l.place(x=411,y=40)
        start = tk.StringVar()
        start_entry = tk.Entry(self,textvariable=start)
        start_entry.place(x=511,y=40)

        end_l = tk.Label(self,text="End Point:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        end_l.place(x=411,y=60)
        end = tk.StringVar()
        end_entry = tk.Entry(self,textvariable=end)
        end_entry.place(x=511,y=60)

        visual_button = tk.Button(net_canvas,text="Visualize",command=lambda:self.network.export_predictions(int(start.get()),int(end.get())))
        visual_button['bg'] = parent.theme.bg
        visual_button['fg'] = parent.theme.fg
        visual_button.place(x=411,y=80)
        '''

        inference_l = tk.Label(net_canvas,text="Inference Setup:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        inference_l.place(x=401,y=200)
        inference_button = tk.Button(net_canvas,text="Infer",command=lambda:self.gen_inference_box(),height = 2, width = 20)
        inference_button['bg'] = parent.theme.bg
        inference_button['fg'] = parent.theme.fg
        inference_button.place(x=521,y=190)

    def callback_cycles(self):
        if os.path.exists(self.cycles_root_path):
            list_values = list(os.listdir(self.cycles_root_path))
        else:
            list_values = ['']
        self.model_cycles['values'] = list_values

    def callback_trend(self):
        if os.path.exists(self.trend_root_path):
            list_values = list(os.listdir(self.trend_root_path))
        else:
            list_values = ['']
        self.model_trend['values'] = list_values

    def callback_control(self):
        if os.path.exists(self.controllers_root_path):
            list_values = list(os.listdir(self.controllers_root_path))
        else:
            list_values = []
        list_values.append("Add new Controller")
        self.controller['values'] = list_values

    def gen_create_box(self):
        _win = creationToplevelControlGUI(self)

    def gen_train_box(self):
        _win = trainToplevelControlGUI(self)

    def gen_inference_box(self):
        _win = inferenceToplevelControlGUI(self)