import os
import json
import codecs
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from .gui_toplevels import creationToplevelClassificationGUI, trainToplevelClassificationGUI, trainToplevelControlGUI, inferenceToplevelGUI
from .gui_main import cycles_windows_execute,trend_windows_execute
import include.network.network as net
import include.network.net_constants as netco

class simulationFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        net_canvas = tk.Canvas(self,bg=parent.theme.selected_bg,width = grandparent.width,height=grandparent.height)
        net_canvas.grid(row=0,column=0)

        # [Network Setup Section]
        label = tk.Label(net_canvas,text="Neural Network Settings",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        label.place(x=1,y=10)

        network_edition_l = tk.Label(net_canvas,text="Edition:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        network_edition_l.place(x=1,y=40)
        self.network_edition = ttk.Combobox(net_canvas,state="readonly")
        self.network_edition['values'] = (netco.CYCLES,netco.TREND)
        self.network_edition.current(0)
        self.network_edition.place(x=111,y=40)

        def callback(network_frame):
            if network_frame.network_edition.get() == netco.TREND:
                if not os.path.exists(os.getcwd()+"/models/"+netco.TREND):
                    network_frame.model['values'] = list([""])
                else:
                    network_frame.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.TREND))
            elif network_frame.network_edition.get() == netco.CYCLES:
                if not os.path.exists(os.getcwd()+"/models/"+netco.CYCLES):
                    network_frame.model['values'] = list([""])
                else:
                    network_frame.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.CYCLES))

        model_l = tk.Label(net_canvas,text="Model:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        model_l.place(x=1,y=60)
        self.model = ttk.Combobox(net_canvas,state="readonly")
        if self.network_edition.get() == netco.TREND:
            if not os.path.exists(os.getcwd()+"/models/"+netco.TREND):
                self.model['values'] = list([""])
            else:
                self.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.TREND))
        elif self.network_edition.get() == netco.CYCLES:
            if not os.path.exists(os.getcwd()+"/models/"+netco.CYCLES):
                self.model['values'] = list([""])
            else:
                self.model['values'] = list(os.listdir(os.getcwd()+"/models/"+netco.CYCLES))

        self.model.current(0)
        self.model.place(x=111,y=60)
        self.network_edition.bind("<<ComboboxSelected>>", lambda _ : callback(self))

        layers_l = tk.Label(net_canvas,text="Layers:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        layers_l.place(x=1,y=80)
        layers = ttk.Combobox(net_canvas,state="readonly")
        layers['values'] = list(range(1,6))
        layers.current(0)
        layers.place(x=111,y=80)
        
        # [Create Section]
        create_l = tk.Label(net_canvas,text="Structure:",bg=net_canvas['bg'],fg = parent.theme.selected_fg)
        create_l.place(x=1,y=100)
        create_button = tk.Button(net_canvas,text="Create Network",command=lambda:self.gen_create_box(int(layers.get())))
        create_button['bg'] = parent.theme.bg
        create_button['fg'] = parent.theme.fg
        create_button.place(x=111,y=100)

        # [Train Section]
        train_button = tk.Button(net_canvas,text="Training Setup",command=lambda:self.gen_train_box())
        train_button['bg'] = parent.theme.bg
        train_button['fg'] = parent.theme.fg
        train_button.place(x=111,y=220)

        # [Visualization Section]
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

        inf_button = tk.Button(net_canvas,text="Inference",command=lambda:self.gen_inference_box())
        inf_button['bg'] = parent.theme.bg
        inf_button['fg'] = parent.theme.fg
        inf_button.place(x=411,y=120)

    def gen_create_box(self,layers):
        _win = creationToplevelGUI(self,layers)

    def gen_train_box(self):
        _win = trainToplevelGUI(self)

    def gen_inference_box(self):
        _win = inferenceToplevelGUI(self)