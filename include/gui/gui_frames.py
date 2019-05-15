import os
import json
import codecs
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from .gui_main import cycles_windows_execute,trend_windows_execute
import include.network.network as net
import include.network.net_constants as netco


class cycleFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        cycles_set_canvas = tk.Canvas(self,bg=parent.theme.selected_bg,width=grandparent.width,height=grandparent.height)
        cycles_set_canvas.grid(row=0,column=0)
        label = tk.Label(cycles_set_canvas,text="Engine Cycles Window Settings",bg=cycles_set_canvas['bg'],fg=parent.theme.selected_fg)
        label.place(x=1,y=10)

        cycles_window_size_l = tk.Label(cycles_set_canvas,text="Window Size:",bg=cycles_set_canvas['bg'],fg=parent.theme.selected_fg)
        cycles_window_size_l.place(x=1,y=40)
        cycles_window_size = ttk.Combobox(cycles_set_canvas,state="readonly")
        cycles_window_size['values'] = (30,60,90,180,360)
        cycles_window_size.current(len(cycles_window_size['values'])-1)
        cycles_window_size.place(x=111,y=40)

        cycles_window_step_l = tk.Label(cycles_set_canvas,text="Window Step:",bg=cycles_set_canvas['bg'],fg=parent.theme.selected_fg)
        cycles_window_step_l.place(x=1,y=60)
        cycles_window_step = ttk.Combobox(cycles_set_canvas,state="readonly")
        cycles_window_step['values'] = (1,3,5,9,10,20)
        cycles_window_step.current(0)
        cycles_window_step.place(x=111,y=60)

        cycle_ex_button = tk.Button(cycles_set_canvas,text="Execute",command=lambda:cycles_windows_execute(int(cycles_window_size.get()),int(cycles_window_step.get())))
        cycle_ex_button['bg'] = parent.theme.bg
        cycle_ex_button['fg'] = parent.theme.fg
        cycle_ex_button.place(x=251,y=200)

class trendFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        trend_set_canvas = tk.Canvas(self,bg=parent.theme.selected_bg,width = grandparent.width,height=grandparent.height)
        trend_set_canvas.grid(row=0,column=0)
        label = tk.Label(trend_set_canvas,text="Speed Trend Window Settings",bg=trend_set_canvas['bg'],fg = parent.theme.selected_fg)
        label.place(x=1,y=10)

        trend_window_size_l = tk.Label(trend_set_canvas,text="Window Size:",bg=trend_set_canvas['bg'],fg = parent.theme.selected_fg)
        trend_window_size_l.place(x=1,y=40)
        trend_window_size = ttk.Combobox(trend_set_canvas,state="readonly")
        trend_window_size['values'] = (30,60,90,180)
        trend_window_size.current(len(trend_window_size['values'])-1)
        trend_window_size.place(x=111,y=40)

        trend_window_step_l = tk.Label(trend_set_canvas,text="Window Step:",bg=trend_set_canvas['bg'],fg = parent.theme.selected_fg)
        trend_window_step_l.place(x=1,y=60)
        trend_window_step = ttk.Combobox(trend_set_canvas,state="readonly")
        trend_window_step['values'] = (3,5,9,10,20,30)
        trend_window_step.current(1)
        trend_window_step.place(x=111,y=60)

        trend_ex_button = tk.Button(trend_set_canvas,text="Execute",command=lambda:trend_windows_execute(int(trend_window_size.get()),int(trend_window_step.get())))
        trend_ex_button['bg'] = parent.theme.bg
        trend_ex_button['fg'] = parent.theme.fg
        trend_ex_button.place(x=251,y=200)

class networksFrame(tk.Frame):
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
        self.network_edition['values'] = (netco.TREND,netco.CYCLES)
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

class creationToplevelGUI(tk.Toplevel):
    def __init__(self,parent,layers):
        tk.Toplevel.__init__(self)
        self.settingsGUI("Network Structure")
        self.parent = parent
        self.layers = layers
        in_label = tk.Label(self,text="In",bg=self['bg'],fg = "#EFB509")
        in_label.grid(row=0,column=0,sticky="nsew")
        out_label = tk.Label(self,text="Out",bg=self['bg'],fg = "#EFB509")
        out_label.grid(row=0,column=1,sticky="nsew")

        def limitSize(*args):
            values_in= []
            values_out = []
            for i in range(len(self.inValues)):
                values_in.append(self.inValues[i].get())
                values_out.append(self.outValues[i].get())
                if len(values_out[i]) > 3:
                    self.outValues[i].set(values_out[i][:3])
                if i>0:
                    self.inValues[i].set(values_out[i-1])
                if len(values_in[i]) > 3:
                    self.inValues[i].set(values_in[i][:3])
                
        self.inValues = []
        self.outValues = []
        for i in range(layers):
            if i == 0:
                if self.parent.network_edition.get() == netco.TREND:
                    self.inValues.append(tk.StringVar(value=str(len(netco.TREND_FEATURES)-1)))
                elif self.parent.network_edition.get() == netco.CYCLES:
                    self.inValues.append(tk.StringVar(value=str(len(netco.CYCLES_FEATURES)-1)))
            else:
                self.inValues.append(tk.StringVar())
            self.inValues[i].trace('w',limitSize)
            inEntry = tk.Entry(self,textvariable=self.inValues[i])
            inEntry.grid(row=i+1,column=0)

            if i == layers-1:
                if self.parent.network_edition.get() == netco.TREND:
                    self.outValues.append(tk.StringVar(value=netco.TREND_OUTPUTS))
                elif self.parent.network_edition.get() == netco.CYCLES:
                    self.outValues.append(tk.StringVar(value=netco.CYCLES_OUTPUTS))
            else:
                self.outValues.append(tk.StringVar())
            self.outValues[i].trace('w',limitSize)
            outEntry = tk.Entry(self,textvariable=self.outValues[i])
            outEntry.grid(row=i+1,column=1)
        
        save_button = tk.Button(self,text="Save",command=lambda:self.create_network())
        save_button.grid(row=layers+1,column=0)
        
        self.mainloop()

    def settingsGUI(self,title,width=400,heigth=400):
        self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg="#000000")

    def create_network(self):
        root_path = os.getcwd()+"/models/"+self.parent.network_edition.get()+"/"+self.parent.model.get()
        if self.parent.network_edition.get() == netco.TREND: network_name = netco.NN_TREND
        if self.parent.network_edition.get() == netco.CYCLES: network_name = netco.NN_CYCLES
        self.network = net.Network(netco.CREATE,network_name,root_path)

        lista = []
        for i in range(self.layers):
            lista.append([self.outValues[i].get(),self.inValues[i].get()])
        self.structure = np.array(lista)
        self.parent.structure = self.structure
        if not os.path.exists(self.network.version_path):
            os.makedirs(self.network.version_path)
        with open(self.network.version_path+"/info.txt","w") as file:
            file.write(30*"-"+"\n")
            file.write("Network Structure Information\n")
            file.write("Layer Format: [IN,OUT]\n")
            file.write(30*"#"+"\n")
            layer_counter = 1
            for layer in self.structure:
                if layer_counter == 1:
                    file.write("Start: ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+"]\n")
                elif layer_counter == len(self.structure):
                    file.write("Exit:  ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+"]\n")
                else:
                    file.write("Hidden ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+"]\n")
                layer_counter+=1
            file.write(30*"-"+"\n")
        b = self.structure.tolist() # nested lists with same data, indices
        file_path = self.network.version_path+"/network_structure.json"
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)

class trainToplevelGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.settingsGUI("Training Setup")
        self.parent = parent
        self.root_path = os.getcwd()+'/models/'+self.parent.network_edition.get()+'/'+self.parent.model.get()

        _label = tk.Label(self,text="Training Parameters",bg=self['bg'],fg = "#EFB509")
        _label.grid(row=0,column=0,sticky="nsew")

        networks = []
        for file in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,file)):
                networks.append(file)

        network_l = tk.Label(self,text="Network:",bg=self['bg'],fg = "#EFB509")
        network_l.place(x=1,y=40)

        self.network_version = ttk.Combobox(self,state="readonly")
        self.network_version['values'] = sorted(networks)      
        self.network_version.current(0)
        self.network_version.place(x=111,y=40)
        self.network = net.Network(netco.LOAD,self.network_version.get(),self.root_path)
        self.network.layers_import(self.network.version_path+"/network_structure.json")

        def callback():
            self.network = net.Network(netco.LOAD,self.network_version.get(),self.root_path)
            self.network.layers_import(self.network.version_path+"/network_structure.json")

        self.network_version.bind("<<ComboboxSelected>>", lambda _ : callback())

        epochs_l = tk.Label(self,text="Epochs:",bg=self['bg'],fg = "#EFB509")
        epochs_l.place(x=1,y=60)
        epochs = ttk.Combobox(self,state="readonly")
        epochs['values'] = (1,5,10,20,50,80,100,200,500,1000,2000)
        epochs.current(0)
        epochs.place(x=111,y=60)

        learning_rate_l = tk.Label(self,text="Learning Rate:",bg=self['bg'],fg = "#EFB509")
        learning_rate_l.place(x=1,y=80)
        learning_rate = ttk.Combobox(self,state="readonly")
        learning_rate['values'] = (0.0001,0.001,0.01,0.1,1,10)
        learning_rate.current(0)
        learning_rate.place(x=111,y=80)    

        mini_batch_l = tk.Label(self,text="Minibatch Size:",bg=self['bg'],fg = "#EFB509")
        mini_batch_l.place(x=1,y=100)
        mini_batch = ttk.Combobox(self,state="readonly")
        mini_batch['values'] = (32,64,128,256)
        mini_batch.current(0)
        mini_batch.place(x=111,y=100)

        if self.parent.network_edition.get() == netco.CYCLES:
            network_edition = self.parent.network_edition.get()
            features_list = netco.CYCLES_FEATURES

        if self.parent.network_edition.get() == netco.TREND:
            network_edition = self.parent.network_edition.get()
            features_list = netco.TREND_FEATURES

        train_button = tk.Button(self,text="Train",command=lambda:self.network.train(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),network_edition))
        train_button.place(x=111,y=140)
        self.bind('<Return>', lambda event:self.parent.network.train(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),network_edition))
        
        self.mainloop()

    def settingsGUI(self,title,width=400,heigth=200):
        self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg="#000000")

class inferenceToplevelGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.settingsGUI("Inference Setup")
        self.parent = parent
        self.root_path = os.getcwd()+'/models/'+self.parent.network_edition.get()+'/'+self.parent.model.get()

        networks = []
        for file in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,file)):
                networks.append(file)

        self.network_version = ttk.Combobox(self,state="readonly")
        self.network_version['values'] = sorted(networks)      
        self.network_version.current(0)
        self.network_version.place(x=111,y=60)
        
        train1_button = tk.Button(self,text="edition",command=lambda:self.load_trained_network())
        train1_button.place(x=111,y=80)
        
        self.mainloop()

    def load_trained_network(self):
        network_name = self.network_version.get()
        network_root_path = self.root_path
        self.network = net.Network(netco.LOAD,network_name,network_root_path)
        #self.network.layers_import(self.network.version_path+"/network_structure.json")
        window_settings = self.parent.model.get().split('_')
        window_settings = [30,2]
        #del window_settings[0]
        self.network.inference(window_settings)
        
    def settingsGUI(self,title,width=400,heigth=200):
        self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg="#000000")