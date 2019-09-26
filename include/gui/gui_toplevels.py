import os
import json
import codecs
import re
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import include.network.network as net
import include.network.net_constants as netco

class creationToplevelClassificationGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        
        self.parent = parent
        self.settingsGUI("Network Structure")

        settings_l = tk.Label(self,text="Network Structure Settings",bg=self['bg'],fg = self.parent.parent.theme.bg)
        settings_l.grid(row=0,column=1)

        empty_label = tk.Label(self,text="",bg=self['bg'],fg = self.parent.parent.theme.bg)
        empty_label.grid(row=1,column=0,sticky="nsew",pady=5)

        layers_l = tk.Label(self,text="Layers:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        layers_l.grid(row=2,column=0)

        self.layers = ttk.Combobox(self,state="readonly")#,postcommand = self.callback_trend)
        self.layers['values'] = list(range(1,6))
        self.layers.current(1)
        self.layers.grid(row=2,column=1)
        self.callback_layers_creation()
        self.layers.bind("<<ComboboxSelected>>", lambda _ : self.callback_layers_creation())

        self.mainloop()

    def callback_layers_creation(self):
        for count,i in enumerate(self.winfo_children()):
            if count>=6:
                self.winfo_children()[-1].destroy()
        
        layers = int(self.layers.get())
        empty_label = tk.Label(self,text="",bg=self['bg'],fg = self.parent.parent.theme.bg)
        empty_label.grid(row=3,column=0,sticky="nsew",pady=5)

        in_label = tk.Label(self,text="In",bg=self['bg'],fg = self.parent.parent.theme.bg)
        in_label.grid(row=4,column=0,sticky="nsew")
        out_label = tk.Label(self,text="Out",bg=self['bg'],fg = self.parent.parent.theme.bg)
        out_label.grid(row=4,column=1,sticky="nsew")
        activation_label = tk.Label(self,text="Activation",bg=self['bg'],fg = self.parent.parent.theme.bg)
        activation_label.grid(row=4,column=2,sticky="nsew")
        
        def limitSize(*args):
            values_in = []
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
        self.activationValues = []
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
            inEntry.grid(row=5+i,column=0)

            if i == layers-1:
                if self.parent.network_edition.get() == netco.TREND:
                    self.outValues.append(tk.StringVar(value=netco.TREND_OUTPUTS))
                elif self.parent.network_edition.get() == netco.CYCLES:
                    self.outValues.append(tk.StringVar(value=netco.CYCLES_OUTPUTS))
            else:
                self.outValues.append(tk.StringVar())
            self.outValues[i].trace('w',limitSize)
            outEntry = tk.Entry(self,textvariable=self.outValues[i])
            outEntry.grid(row=5+i,column=1)

            activationEntry = ttk.Combobox(self,state="readonly")
            activationEntry['values'] = (netco.TANH,netco.LINEAR,netco.SIGMOID,netco.RELU)
            activationEntry.current(0)
            activationEntry.grid(row=5+i,column=2)
            self.activationValues.append(activationEntry)
            
        save_button = tk.Button(self,text="Save",command=lambda:self.create_network())
        save_button['bg'] = self.parent.parent.theme.bg
        save_button['fg'] = self.parent.parent.theme.fg
        save_button.grid(row=layers+5,column=1,pady=25)

    def settingsGUI(self,title,width=500,heigth=400):
        self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg=self.parent.parent.theme.fg)

    def create_network(self):
        layers = int(self.layers.get())
        if self.parent.network_edition.get() == netco.TREND: 
            network_name = netco.NN_TREND
            features = netco.TREND_FEATURES

        if self.parent.network_edition.get() == netco.CYCLES: 
            network_name = netco.NN_CYCLES
            features = netco.CYCLES_FEATURES
        
        root_path = os.path.join(os.getcwd(),netco.CLASSIFIERS,self.parent.network_edition.get(),self.parent.model.get())

        self.network = net.Network(self.parent.network_edition.get(),netco.CREATE,network_name,root_path,features)

        structure_data = []
        for i in range(layers):
            structure_data.append([self.outValues[i].get(),self.inValues[i].get(),self.activationValues[i].get()])
        self.structure = np.array(structure_data)
        self.parent.structure = self.structure
        if not os.path.exists(self.network.version_path):
            os.makedirs(self.network.version_path)
        with open(self.network.version_path+"/info.txt","w") as file:
            file.write(30*"-"+"\n")
            file.write("Network Structure Information\n")
            file.write("Layer Format: [IN,OUT]\n")
            file.write(30*"#"+"\n")
            for layer_counter,layer in enumerate(self.structure,start=0):
                if layer_counter == 0:
                    file.write("First: ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                elif layer_counter == len(self.structure)-1:
                    file.write("Last:  ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                else:
                    file.write("Hidden ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                layer_counter+=1
            file.write(30*"-"+"\n")
        b = self.structure.tolist() # nested lists with same data, indices
        file_path = self.network.version_path+"/network_structure.json"
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)

class trainToplevelClassificationGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.parent = parent
        self.settingsGUI("Training Setup")
        self.root_path = os.path.join(os.getcwd(),netco.CLASSIFIERS,self.parent.network_edition.get(),self.parent.model.get())

        _label = tk.Label(self,text="Training Parameters",bg=self['bg'],fg = self.parent.parent.theme.bg)
        _label.grid(row=0,column=0,sticky="nsew")

        network_l = tk.Label(self,text="Network:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        network_l.place(x=1,y=40)

        if self.parent.network_edition.get() == netco.CYCLES: features = netco.CYCLES_FEATURES
        if self.parent.network_edition.get() == netco.TREND: features = netco.TREND_FEATURES
        
        networks = []
        for file in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,file)):
                networks.append(file)

        x_list_place = 151
        self.network_version = ttk.Combobox(self,state="readonly",postcommand = self.callback_network)
        self.network_version['values'] = sorted(networks)      
        self.network_version.current(0)
        self.network_version.place(x=x_list_place,y=40)
        self.network = net.Network(self.parent.network_edition.get(),netco.LOAD,self.network_version.get(),self.root_path,features)
        self.network.layers_import(self.network.version_path+"/network_structure.json")

        def callback():
            self.network = net.Network(self.parent.network_edition.get(),netco.LOAD,self.network_version.get(),self.root_path,features)
            self.network.layers_import(self.network.version_path+"/network_structure.json")

        self.network_version.bind("<<ComboboxSelected>>", lambda _ : callback())

        epochs_l = tk.Label(self,text="Epochs:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        epochs_l.place(x=1,y=60)
        epochs = ttk.Combobox(self,state="readonly")
        epochs['values'] = (1,100,150,200,500,1000,2000)
        epochs.current(0)
        epochs.place(x=x_list_place,y=60)

        learning_rate_l = tk.Label(self,text="Learning Rate:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        learning_rate_l.place(x=1,y=80)
        learning_rate = ttk.Combobox(self,state="readonly")
        learning_rate['values'] = (0.0001,0.001,0.01,0.1,1,10)
        learning_rate.current(0)
        learning_rate.place(x=x_list_place,y=80)    

        mini_batch_l = tk.Label(self,text="Minibatch Size:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        mini_batch_l.place(x=1,y=100)
        mini_batch = ttk.Combobox(self,state="readonly")
        mini_batch['values'] = (8,16,32,64,128,256)
        mini_batch.current(2)
        mini_batch.place(x=x_list_place,y=100)

        shuffle_l = tk.Label(self,text="Shuffle Data:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        shuffle_l.place(x=1,y=120)
        shuffle = ttk.Combobox(self,state="readonly")
        shuffle['values'] = (True,False)
        shuffle.current(0)
        shuffle.place(x=x_list_place,y=120)

        test_size_l = tk.Label(self,text="Train/Test Split %:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        test_size_l.place(x=1,y=140)
        test_size = ttk.Combobox(self,state="readonly")
        test_size['values'] = (0.3,0.25,0.2,0.15,0.1)
        test_size.current(0)
        test_size.place(x=x_list_place,y=140)

        train_button = tk.Button(self,text="Train",command=lambda:self.training(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),shuffle.get(),float(test_size.get())))
        train_button['bg'] = self.parent.parent.theme.bg
        train_button['fg'] = self.parent.parent.theme.fg
        train_button.place(x=x_list_place,y=180)
        self.bind('<Return>', lambda event:self.training(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),shuffle.get(),float(test_size.get())))
        
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
        self.configure(bg=self.parent.parent.theme.fg)

    def training(self,epochs,learning_rate,mini_batch,shuffle,test_size):
        if self.parent.network_edition.get() == netco.CYCLES:
            data = pd.read_csv(os.path.join(self.network.root_path,netco.TRAINING+".csv"),usecols=netco.CYCLES_FEATURES)
            self.network.train(data,epochs,learning_rate,mini_batch,shuffle,test_size,netco.CYCLES_OUTPUTS)
        if self.parent.network_edition.get() == netco.TREND:
            data = pd.read_csv(os.path.join(self.network.root_path,netco.TRAINING+".csv"),usecols=netco.TREND_FEATURES)
            self.network.train(data,epochs,learning_rate,mini_batch,shuffle,test_size,netco.TREND_OUTPUTS)
            
    def callback_network(self):
        networks = []
        for file in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,file)):
                networks.append(file)
        self.network_version['values'] = sorted(networks)

class fancyFrame(tk.Frame):
    def __init__(self,parent,inValues,outValues,activationValues,layers):
        tk.Frame.__init__(self,parent)
        self.config(bg=parent['bg'])
        self.inValues = inValues
        self.outValues = outValues
        self.activationValues = activationValues
        self.layers = layers

class creationToplevelControlGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.parent = parent
        self.settingsGUI("Control Networks Structures")
        settings_l = tk.Label(self,text="Network Structure Settings",bg=self['bg'],fg = self.parent.parent.theme.bg)
        settings_l.place(x=200,y=0)
        
        cycles_l = tk.Label(self,text="Cycle:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        cycles_l.place(x=150,y=20)
        self.cycle = ttk.Combobox(self,state="readonly")
        self.cycle['values'] = list(range(0,7))
        self.cycle.current(0)
        self.cycle.place(x=250,y=20)

        self.ice_frame = fancyFrame(self,[],[],[],2)
        self.ice_frame.place(x=100,y=40)

        ice_label = tk.Label(self.ice_frame,text="Engine Control Network",bg=self['bg'],fg = self.parent.parent.theme.bg)
        ice_label.grid(row=0,column=0,sticky="nsew")
        layers_ice_l = tk.Label(self.ice_frame,text="Layers:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        layers_ice_l.grid(row=1,column=0)

        self.layers_ice = ttk.Combobox(self.ice_frame,state="readonly")
        self.layers_ice['values'] = list(range(1,6))
        self.layers_ice.current(1)
        self.layers_ice.grid(row=1,column=1)
        self.layers_ice.bind("<<ComboboxSelected>>", lambda _ : self.callback_layers_creation(self.ice_frame))

        self.emot_frame = fancyFrame(self,[],[],[],2)
        self.emot_frame.place(x=100,y=240)

        empty_label = tk.Label(self.ice_frame,text="",bg=self['bg'],fg = self.parent.parent.theme.bg)
        empty_label.grid(row=2,column=0,sticky="nsew",pady=5)

        emot_label = tk.Label(self.emot_frame,text="Motor Control Network",bg=self['bg'],fg = self.parent.parent.theme.bg)
        emot_label.grid(row=0,column=0,sticky="nsew")

        layers_emot_l = tk.Label(self.emot_frame,text="Layers:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        layers_emot_l.grid(row=1,column=0)

        self.layers_emot = ttk.Combobox(self.emot_frame,state="readonly")
        self.layers_emot['values'] = list(range(1,6))
        self.layers_emot.current(1)
        self.layers_emot.grid(row=1,column=1)
        self.callback_layers_creation(self.ice_frame)
        self.callback_layers_creation(self.emot_frame)
        self.layers_emot.bind("<<ComboboxSelected>>", lambda _ : self.callback_layers_creation(self.emot_frame))

        if self.parent.controller.get()=='Add new Controller':
            self.version_control()
            self.nncontroller = netco.NNCONTROLLER+"_"+str(self.version)
            print("~$> Creating new Controller: |-> "+self.nncontroller)
            controller_l = tk.Label(self,text="Adding Controller : "+self.nncontroller,bg=self['bg'],fg = self.parent.parent.theme.selected_fg)
        else:
            self.version = int(self.parent.controller.get().split("_")[-1])
            self.nncontroller = self.parent.controller.get()
            print("~$> Editing Controller: |-> "+self.nncontroller)
            controller_l = tk.Label(self,text="Editing Controller: "+self.nncontroller,bg=self['bg'],fg = self.parent.parent.theme.selected_fg)
        
        controller_l.place(x=250,y=400)
        save_button = tk.Button(self,text="Save",command=lambda:self.create_nncontroller())
        save_button['bg'] = self.parent.parent.theme.bg
        save_button['fg'] = self.parent.parent.theme.fg
        save_button.place(x=300,y=500)
        self.mainloop()

    def callback_layers_creation(self,frame):
        for count,i in enumerate(frame.winfo_children()):
            if count>=3:
                frame.winfo_children()[-1].destroy()
        if frame == self.ice_frame:
            frame.layers = int(self.layers_ice.get())
            features = netco.ENG_FEATURES
            outputs = netco.ENG_OUTPUTS
        else:
            frame.layers = int(self.layers_emot.get())
            features = netco.EMOT_FEATURES
            outputs = netco.EMOT_OUTPUTS

        in_label = tk.Label(frame,text="In",bg=self['bg'],fg = self.parent.parent.theme.bg)
        in_label.grid(row=3,column=0,sticky="nsew")
        out_label = tk.Label(frame,text="Out",bg=self['bg'],fg = self.parent.parent.theme.bg)
        out_label.grid(row=3,column=1,sticky="nsew")
        activation_label = tk.Label(frame,text="Activation",bg=self['bg'],fg = self.parent.parent.theme.bg)
        activation_label.grid(row=3,column=2,sticky="nsew")
        
        def limitSize(*args):
            values_in = []
            values_out = []
            for i in range(len(frame.inValues)):
                values_in.append(frame.inValues[i].get())
                values_out.append(frame.outValues[i].get())
                if len(values_out[i]) > 3:
                    frame.outValues[i].set(values_out[i][:3])
                if i>0:
                    frame.inValues[i].set(values_out[i-1])
                if len(values_in[i]) > 3:
                    frame.inValues[i].set(values_in[i][:3])      
                
        frame.inValues = []
        frame.outValues = []
        frame.activationValues = []
        for i in range(frame.layers):
            if i == 0:
                frame.inValues.append(tk.StringVar(value=str(len(features)-1)))
            else:
                frame.inValues.append(tk.StringVar())
            frame.inValues[i].trace('w',limitSize)
            inEntry = tk.Entry(frame,textvariable=frame.inValues[i])
            inEntry.grid(row=5+i,column=0)

            if i == frame.layers-1:
                frame.outValues.append(tk.StringVar(value=outputs))
            else:
                frame.outValues.append(tk.StringVar())
            frame.outValues[i].trace('w',limitSize)
            outEntry = tk.Entry(frame,textvariable=frame.outValues[i])
            outEntry.grid(row=5+i,column=1)

            activationEntry = ttk.Combobox(frame,state="readonly")
            activationEntry['values'] = (netco.TANH,netco.LINEAR,netco.SIGMOID,netco.RELU)
            activationEntry.current(0)
            activationEntry.grid(row=5+i,column=2)
            frame.activationValues.append(activationEntry)

    def settingsGUI(self,title,width=600,heigth=600):
        self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg=self.parent.parent.theme.fg)

    def version_control(self):
        versions_dir = []
        if not os.path.exists(self.parent.controllers_root_path): os.makedirs(self.parent.controllers_root_path)
        for filename in os.listdir(self.parent.controllers_root_path):
            if os.path.isdir(os.path.join(self.parent.controllers_root_path,filename)):
                if re.match(re.escape(netco.NNCONTROLLER),filename):
                    versions_dir.append(filename)
        versions_dir = sorted(versions_dir,reverse=True)
        if versions_dir == []:
            self.version = 1
        else:
            self.version = int(versions_dir[0].split("_")[-1])+1
        self.version = str(self.version)

    def create_nncontroller(self):
        working_cycle = self.cycle.get()
        controller_cycle_path = os.path.join(self.parent.controllers_root_path,self.nncontroller,netco.CYCLE+'_'+working_cycle)
        if not os.path.exists(controller_cycle_path): os.makedirs(os.path.join(controller_cycle_path))
        
        self.network_ice = net.NNRegressor(netco.ENGINE,netco.CREATE,netco.NN_ENG,controller_cycle_path,netco.ENG_FEATURES)
        self.network_emot = net.NNRegressor(netco.MOTOR,netco.CREATE,netco.NN_EMOT,controller_cycle_path,netco.EMOT_FEATURES)
        structure_ice_data = []
        structure_emot_data = []
        
        for i in range(self.ice_frame.layers):
            structure_ice_data.append([self.ice_frame.outValues[i].get(),self.ice_frame.inValues[i].get(),self.ice_frame.activationValues[i].get()])
        for i in range(self.emot_frame.layers):
            structure_emot_data.append([self.emot_frame.outValues[i].get(),self.emot_frame.inValues[i].get(),self.emot_frame.activationValues[i].get()])
        self.structure_ice = np.array(structure_ice_data)
        self.structure_emot = np.array(structure_emot_data)
        if not os.path.exists(self.network_ice.version_path):
            os.makedirs(self.network_ice.version_path)
        if not os.path.exists(self.network_emot.version_path):
            os.makedirs(self.network_emot.version_path)
        
        with open(self.network_ice.version_path+"/info.txt","w") as file:
            file.write(30*"-"+"\n")
            file.write("Network Structure Information\n")
            file.write("Layer Format: [IN,OUT]\n")
            file.write(30*"#"+"\n")
            for layer_counter,layer in enumerate(self.structure_ice,start=0):
                if layer_counter == 0:
                    file.write("First: ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                elif layer_counter == len(self.structure_ice)-1:
                    file.write("Last:  ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                else:
                    file.write("Hidden ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                layer_counter+=1
            file.write(30*"-"+"\n")
        b = self.structure_ice.tolist() # nested lists with same data, indices
        file_path = self.network_ice.version_path+"/network_structure.json"
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)
    
        with open(self.network_emot.version_path+"/info.txt","w") as file:
            file.write(30*"-"+"\n")
            file.write("Network Structure Information\n")
            file.write("Layer Format: [IN,OUT]\n")
            file.write(30*"#"+"\n")
            for layer_counter,layer in enumerate(self.structure_emot,start=0):
                if layer_counter == 0:
                    file.write("First: ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                elif layer_counter == len(self.structure_emot)-1:
                    file.write("Last:  ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                else:
                    file.write("Hidden ["+str(layer_counter)+"]: ")
                    file.write("["+str(layer[0])+", "+str(layer[1])+", "+layer[2]+"]\n")
                layer_counter+=1
            file.write(30*"-"+"\n")
        b = self.structure_emot.tolist() # nested lists with same data, indices
        file_path = self.network_emot.version_path+"/network_structure.json"
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)
        
class trainToplevelControlGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.parent = parent
        self.settingsGUI("Training Setup")
        
        _label = tk.Label(self,text="Training Parameters",bg=self['bg'],fg = self.parent.parent.theme.bg)
        _label.grid(row=0,column=0,sticky="nsew")

        x_list_place_ice=151
        self.controllers_root_path = os.path.join(os.getcwd(),netco.CONTROLLERS)
        controller_l = tk.Label(self,text="Controller:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        controller_l.place(x=1,y=20)
        self.controller = ttk.Combobox(self,state="readonly",postcommand = self.callback_control)
        if not os.path.exists(self.controllers_root_path):
            self.controller['values'] = list([""])
        else:
            self.controller['values'] = list(os.listdir(self.controllers_root_path))
        self.controller.current(0)
        self.controller.place(x=x_list_place_ice,y=20)

        self.trend_root_path = os.path.join(os.getcwd(),netco.CLASSIFIERS,netco.TREND)
        model_trend_l = tk.Label(self,text="Trend Model:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        model_trend_l.place(x=1,y=40)
        self.model_trend = ttk.Combobox(self,state="readonly",postcommand = self.callback_trend)
        if not os.path.exists(self.trend_root_path):
            self.model_trend['values'] = list([""])
        else:
            self.model_trend['values'] = list(os.listdir(self.trend_root_path))
        self.model_trend.current(0)
        self.model_trend.place(x=x_list_place_ice,y=40)

        self.controller_path = os.path.join(self.controllers_root_path,self.controller.get())
        self.model_path = os.path.join(self.trend_root_path,self.model_trend.get())

        networks = []
        for file in os.listdir(self.model_path):
            if os.path.isdir(os.path.join(self.model_path,file)):
                networks.append(file)
        trend_network_l = tk.Label(self,text="Trend Network:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        trend_network_l.place(x=1,y=60)
        self.trend_network = ttk.Combobox(self,state="readonly")
        self.trend_network['values'] = sorted(networks)
        self.trend_network.current(0)
        self.trend_network.place(x=x_list_place_ice,y=60)

        self.networks_ice = []
        self.networks_emot = []
        self.trend_classifier = net.NNClassifier(netco.TREND,netco.LOAD,self.trend_network.get(),self.model_path,netco.TREND_FEATURES)
        self.trend_classifier_path = os.path.join(self.model_path,self.trend_network.get())
        for cycle in range(netco.CYCLES_OUTPUTS):
            cycle_path = os.path.join(self.controller_path,netco.CYCLE+"_"+str(cycle))
            network_ice = net.NNRegressor(netco.ENGINE,netco.LOAD,netco.NN_ENG+'_1',cycle_path,netco.ENG_FEATURES)
            network_ice.layers_import(network_ice.version_path+"/network_structure.json")
            self.networks_ice.append(network_ice)

            network_emot = net.NNRegressor(netco.MOTOR,netco.LOAD,netco.NN_EMOT+'_1',cycle_path,netco.EMOT_FEATURES)
            network_emot.layers_import(network_ice.version_path+"/network_structure.json")
            self.networks_emot.append(network_emot)

        def callback():
            self.controller_path = os.path.join(self.controllers_root_path,self.controller.get())
            self.model_path = os.path.join(self.trend_root_path,self.model_trend.get())

            networks = []
            for file in os.listdir(self.model_path):
                if os.path.isdir(os.path.join(self.model_path,file)):
                    networks.append(file)
            self.trend_network['values'] = sorted(networks)
            self.trend_classifier_path = os.path.join(self.model_path,self.trend_network.get())
            self.trend_classifier = net.NNClassifier(netco.TREND,netco.LOAD,self.trend_network.get(),self.model_path,netco.TREND_FEATURES)
            for cycle in range(netco.CYCLES_OUTPUTS):
                cycle_path = os.path.join(self.controller_path,netco.CYCLE+"_"+str(cycle))
                network_ice = net.NNRegressor(netco.ENGINE,netco.LOAD,netco.NN_ENG+'_1',cycle_path,netco.ENG_FEATURES)
                network_ice.layers_import(network_ice.version_path+"/network_structure.json")
                self.networks_ice.append(network_ice)

                network_emot = net.NNRegressor(netco.MOTOR,netco.LOAD,netco.NN_EMOT+'_1',cycle_path,netco.EMOT_FEATURES)
                network_emot.layers_import(network_ice.version_path+"/network_structure.json")
                self.networks_emot.append(network_emot)

        self.controller.bind("<<ComboboxSelected>>", lambda _ : callback())
        self.model_trend.bind("<<ComboboxSelected>>", lambda _ : callback())
        self.trend_network.bind("<<ComboboxSelected>>", lambda _ : callback())

        epochs_l = tk.Label(self,text="Epochs:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        epochs_l.place(x=1,y=80)
        epochs = ttk.Combobox(self,state="readonly")
        epochs['values'] = (1,5,10,20,50,80,100,200,500,1000,2000)
        epochs.current(0)
        epochs.place(x=x_list_place_ice,y=80)

        learning_rate_l = tk.Label(self,text="Learning Rate:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        learning_rate_l.place(x=1,y=100)
        learning_rate = ttk.Combobox(self,state="readonly")
        learning_rate['values'] = (0.0001,0.001,0.01,0.1,1,10)
        learning_rate.current(0)
        learning_rate.place(x=x_list_place_ice,y=100)    

        mini_batch_l = tk.Label(self,text="Minibatch Size:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        mini_batch_l.place(x=1,y=120)
        mini_batch = ttk.Combobox(self,state="readonly")
        mini_batch['values'] = (8,16,32,64,128,256)
        mini_batch.current(2)
        mini_batch.place(x=x_list_place_ice,y=120)

        shuffle_l = tk.Label(self,text="Shuffle Data:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        shuffle_l.place(x=1,y=140)
        shuffle = ttk.Combobox(self,state="readonly")
        shuffle['values'] = (True,False)
        shuffle.current(0)
        shuffle.place(x=x_list_place_ice,y=140)

        test_size_l = tk.Label(self,text="Train/Test Split %:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        test_size_l.place(x=1,y=160)
        test_size = ttk.Combobox(self,state="readonly")
        test_size['values'] = (0.3,0.25,0.2,0.15,0.1)
        test_size.current(0)
        test_size.place(x=x_list_place_ice,y=160)
        
        train_button = tk.Button(self,text="Train",command=lambda:self.training(int(epochs.get()),float(learning_rate.get()),int(mini_batch.get()),shuffle.get(),float(test_size.get())))
        train_button['bg'] = self.parent.parent.theme.bg
        train_button['fg'] = self.parent.parent.theme.fg
        train_button.place(x=x_list_place_ice,y=180)
        '''
        x_offset = 400
        network_l_emot = tk.Label(self,text="Motor Network:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        network_l_emot.place(x=1+x_offset,y=40)

        x_list_place_emot = x_list_place_ice + x_offset
        
        self.network_version_emot = ttk.Combobox(self,state="readonly")
        self.network_version_emot['values'] = sorted(networks_emot)      
        self.network_version_emot.current(0)
        self.network_version_emot.place(x=x_list_place_emot,y=40)
        self.network_emot = net.Network(netco.MOTOR,netco.LOAD,self.network_version_emot.get(),self.root_path_emot,netco.EMOT_FEATURES)
        self.network_emot.layers_import(self.network_emot.version_path+"/network_structure.json")

        def callback_emot():
            self.network_emot = net.Network(netco.MOTOR,netco.LOAD,self.network_version_emot.get(),self.root_path_emot,netco.EMOT_FEATURES)
            self.network_emot.layers_import(self.network_emot.version_path+"/network_structure.json")

        epochs_l_emot = tk.Label(self,text="Epochs:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        epochs_l_emot.place(x=1+x_offset,y=60)
        epochs_emot = ttk.Combobox(self,state="readonly")
        epochs_emot['values'] = (1,5,10,20,50,80,100,200,500,1000,2000)
        epochs_emot.current(0)
        epochs_emot.place(x=x_list_place_emot,y=60)

        learning_rate_l_emot = tk.Label(self,text="Learning Rate:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        learning_rate_l_emot.place(x=1+x_offset,y=80)
        learning_rate_emot = ttk.Combobox(self,state="readonly")
        learning_rate_emot['values'] = (0.0001,0.001,0.01,0.1,1,10)
        learning_rate_emot.current(0)
        learning_rate_emot.place(x=x_list_place_emot,y=80)    

        mini_batch_l_emot = tk.Label(self,text="Minibatch Size:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        mini_batch_l_emot.place(x=1+x_offset,y=100)
        mini_batch_emot = ttk.Combobox(self,state="readonly")
        mini_batch_emot['values'] = (8,16,32,64,128,256)
        mini_batch_emot.current(2)
        mini_batch_emot.place(x=x_list_place_emot,y=100)

        shuffle_l_emot = tk.Label(self,text="Shuffle Data:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        shuffle_l_emot.place(x=1+x_offset,y=120)
        shuffle_emot = ttk.Combobox(self,state="readonly")
        shuffle_emot['values'] = (True,False)
        shuffle_emot.current(0)
        shuffle_emot.place(x=x_list_place_emot,y=120)

        test_size_l_emot = tk.Label(self,text="Train/Test Split %:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        test_size_l_emot.place(x=1+x_offset,y=140)
        test_size_emot = ttk.Combobox(self,state="readonly")
        test_size_emot['values'] = (0.3,0.25,0.2,0.15,0.1)
        test_size_emot.current(0)
        test_size_emot.place(x=x_list_place_emot,y=140)
        

        train_button_emot = tk.Button(self,text="Train",command=lambda:self.network_emot.train(int(epochs_emot.get()),float(learning_rate_emot.get()),int(mini_batch_emot.get()),shuffle_emot.get(),float(test_size_emot.get())))
        train_button_emot['bg'] = self.parent.parent.theme.bg
        train_button_emot['fg'] = self.parent.parent.theme.fg
        train_button_emot.place(x=x_list_place_emot,y=180)
        '''
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
        self.configure(bg=self.parent.parent.theme.fg)
    
    def training(self,epochs,learning_rate,mini_batch,shuffle,test_size):
        window_settings = [int(self.model_trend.get().split('_')[1]),int(self.model_trend.get().split('_')[2])]
        #for cycle in range(netco.CYCLES_OUTPUTS):
        for cycle in range(2):
            #cycle_data = pd.read_csv(os.path.join(self.model_path,netco.INFERENCE+'_'+str(cycle)+'.csv'))
            #labels = cycle_data['LABEL']
            #cycle_data=cycle_data.drop('LABEL',axis=1)
            #cycle_data = self.trend_classifier.inference(cycle_data,window_settings)
            #cycle_data['LABEL']=labels

            data_ice = pd.read_csv(os.path.join(os.getcwd(),netco.SIMULATIONS,"simulation_"+str(cycle)+".csv"),usecols=netco.ENG_FEATURES)
            data_ice = data_ice.rename(columns={"TQ_ICE": "LABEL"})
            #cycle_data = cycle_data.head(len(data_ice))
            #data_ice['TREND']=cycle_data[netco.PREDICTION_LABEL]
            
            data_emot = pd.read_csv(os.path.join(os.getcwd(),netco.SIMULATIONS,"simulation_"+str(cycle)+".csv"),usecols=netco.EMOT_FEATURES)
            data_emot = data_emot.rename(columns={"TQ_EMOT": "LABEL"})
            #data_emot['TREND']=cycle_data[netco.PREDICTION_LABEL]
        
            self.networks_ice[cycle].train(data_ice,epochs,learning_rate,mini_batch,shuffle,test_size,netco.ENG_OUTPUTS)
            
            self.networks_emot[cycle].train(data_emot,epochs,learning_rate,mini_batch,shuffle,test_size,netco.EMOT_OUTPUTS)
            
        


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
        self.controller['values'] = list_values

class inferenceToplevelControlGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.parent = parent
        self.settingsGUI("Inference Setup")
        self.root_path = os.getcwd()+'/models/'+self.parent.network_edition.get()+'/'+self.parent.model.get()

        _label = tk.Label(self,text="Inference Parameters for "+self.parent.model.get(),bg=self['bg'],fg = self.parent.parent.theme.bg)
        _label.grid(row=0,column=0,sticky="nsew")

        networks = []
        for file in os.listdir(self.root_path):
            if os.path.isdir(os.path.join(self.root_path,file)):
                networks.append(file)

        if self.parent.network_edition.get() == netco.CYCLES: self.features = netco.CYCLES_FEATURES
        if self.parent.network_edition.get() == netco.TREND: self.features = netco.TREND_FEATURES

        network__l = tk.Label(self,text="Network:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        network__l.place(x=1,y=40)
        self.network_version = ttk.Combobox(self,state="readonly")
        self.network_version['values'] = sorted(networks)      
        self.network_version.current(0)
        self.network_version.place(x=111,y=40)
        
        samples = []
        for file in os.listdir(self.root_path+"/samples"):
            samples.append(file)
        samples = sorted(samples)

        sample_l = tk.Label(self,text="Sample:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        sample_l.place(x=1,y=60)
        sample_file = ttk.Combobox(self,state="readonly")
        sample_file['values'] = samples
        sample_file.current(0)
        sample_file.place(x=111,y=60)
        '''
        times_l = tk.Label(self,text="Timestamps:",bg=self['bg'],fg = self.parent.parent.theme.bg)
        times_l.place(x=1,y=80)
        times = tk.Entry(self)
        times.place(x=111,y=80)
        
        load_inference_button = tk.Button(self,text="Inference",command=lambda:self.load_trained_network(sample_file.get(),int(times.get())))
        '''
        load_inference_button = tk.Button(self,text="Inference",command=lambda:self.load_trained_network(sample_file.get()))
        load_inference_button['bg'] = self.parent.parent.theme.bg
        load_inference_button['fg'] = self.parent.parent.theme.fg
        load_inference_button.place(x=111,y=100)

        self.bind('<Return>', lambda event:self.load_trained_network(sample_file.get()))
        
        self.mainloop()

    def load_trained_network(self,sample):
        network_edition = self.parent.network_edition.get()
        network_name = self.network_version.get()
        network_root_path = self.root_path
        network_features = self.features
        self.network = net.NNClassifier(network_edition,netco.LOAD,network_name,network_root_path,network_features)
        window_settings = self.parent.model.get().split('_')
        del window_settings[0]
        self.network.inference(sample,window_settings)
        
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
        self.configure(bg=self.parent.parent.theme.fg)