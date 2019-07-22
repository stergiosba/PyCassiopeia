import os
import json
import codecs
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

import include.network.network as net
import include.network.net_constants as netco

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

        if self.parent.network_edition.get() == netco.TREND: 
            network_name = netco.NN_TREND
            features = netco.TREND_FEATURES

        if self.parent.network_edition.get() == netco.CYCLES: 
            network_name = netco.NN_CYCLES
            features = netco.CYCLES_FEATURES

        if self.parent.network_edition.get() == netco.SOC: 
            network_name == netco.NN_BATTERY
            features = netco.BATTERY_FEATURES

        if self.parent.network_edition.get() == netco.WENG: 
            network_name == netco.NN_WENG
            features = netco.WENG_FEATURES

        self.network = net.Network(self.parent.network_edition.get(),netco.CREATE,network_name,root_path,features)

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

        if self.parent.network_edition.get() == netco.CYCLES: features = netco.CYCLES_FEATURES
        if self.parent.network_edition.get() == netco.TREND: features = netco.TREND_FEATURES
        if self.parent.network_edition.get() == netco.SOC: features = netco.BATTERY_FEATURES
        if self.parent.network_edition.get() == netco.WENG: features = netco.WENG_FEATURES
        
        x_list_place = 121
        self.network_version = ttk.Combobox(self,state="readonly")
        self.network_version['values'] = sorted(networks)      
        self.network_version.current(0)
        self.network_version.place(x=x_list_place,y=40)
        self.network = net.Network(self.parent.network_edition.get(),netco.LOAD,self.network_version.get(),self.root_path,features)
        self.network.layers_import(self.network.version_path+"/network_structure.json")

        def callback():
            self.network = net.Network(self.parent.network_edition.get(),netco.LOAD,self.network_version.get(),self.root_path,features)
            self.network.layers_import(self.network.version_path+"/network_structure.json")

        self.network_version.bind("<<ComboboxSelected>>", lambda _ : callback())

        epochs_l = tk.Label(self,text="Epochs:",bg=self['bg'],fg = "#EFB509")
        epochs_l.place(x=1,y=60)
        epochs = ttk.Combobox(self,state="readonly")
        epochs['values'] = (1,5,10,20,50,80,100,200,500,1000,2000)
        epochs.current(0)
        epochs.place(x=x_list_place,y=60)

        learning_rate_l = tk.Label(self,text="Learning Rate:",bg=self['bg'],fg = "#EFB509")
        learning_rate_l.place(x=1,y=80)
        learning_rate = ttk.Combobox(self,state="readonly")
        learning_rate['values'] = (0.0001,0.001,0.01,0.1,1,10)
        learning_rate.current(0)
        learning_rate.place(x=x_list_place,y=80)    

        mini_batch_l = tk.Label(self,text="Minibatch Size:",bg=self['bg'],fg = "#EFB509")
        mini_batch_l.place(x=1,y=100)
        mini_batch = ttk.Combobox(self,state="readonly")
        mini_batch['values'] = (32,64,128,256)
        mini_batch.current(0)
        mini_batch.place(x=x_list_place,y=100)

        shuffle_l = tk.Label(self,text="Shuffle Data:",bg=self['bg'],fg = "#EFB509")
        shuffle_l.place(x=1,y=120)
        shuffle = ttk.Combobox(self,state="readonly")
        shuffle['values'] = (True,False)
        shuffle.current(0)
        shuffle.place(x=x_list_place,y=120)

        test_size_l = tk.Label(self,text="Train/Test Split %:",bg=self['bg'],fg = "#EFB509")
        test_size_l.place(x=1,y=140)
        test_size = ttk.Combobox(self,state="readonly")
        test_size['values'] = (0.3,0.25,0.2,0.15,0.1)
        test_size.current(0)
        test_size.place(x=x_list_place,y=140)

        train_button = tk.Button(self,text="Train",command=lambda:self.network.train(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),shuffle.get(),float(test_size.get())))
        train_button.place(x=x_list_place,y=180)
        self.bind('<Return>', lambda event:self.network.train(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),shuffle.get(),float(test_size.get())))
        
        self.mainloop()

    def settingsGUI(self,title,width=400,heigth=240):
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

        _label = tk.Label(self,text="Inference Parameters",bg=self['bg'],fg = "#EFB509")
        _label.grid(row=0,column=0,sticky="nsew")

        networks = []
        for file in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,file)):
                networks.append(file)

        if self.parent.network_edition.get() == netco.CYCLES: self.features = netco.CYCLES_FEATURES
        if self.parent.network_edition.get() == netco.TREND: self.features = netco.TREND_FEATURES
        if self.parent.network_edition.get() == netco.SOC: self.features = netco.BATTERY_FEATURES
        if self.parent.network_edition.get() == netco.WENG: self.features = netco.WENG_FEATURES

        self.network_version = ttk.Combobox(self,state="readonly")
        self.network_version['values'] = sorted(networks)      
        self.network_version.current(0)
        self.network_version.place(x=111,y=40)

        samples = []
        for file in os.listdir(self.root_path+"/samples"):
            samples.append(file)
        samples = sorted(samples)

        file = ttk.Combobox(self,state="readonly")
        file['values'] = samples
        file.current(0)
        file.place(x=111,y=60)
        
        load_inference_button = tk.Button(self,text="Inference",command=lambda:self.load_trained_network(file.get()))
        load_inference_button.place(x=111,y=80)

        self.bind('<Return>', lambda event:self.load_trained_network(file.get()))
        
        self.mainloop()

    def load_trained_network(self,sample):
        network_edition = self.parent.network_edition.get()
        network_name = self.network_version.get()
        network_root_path = self.root_path
        network_features = self.features
        self.network = net.Network(network_edition,netco.LOAD,network_name,network_root_path,network_features)
        window_settings = self.parent.model.get().split('_')
        del window_settings[0]
        self.network.inference(window_settings,sample)
        
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