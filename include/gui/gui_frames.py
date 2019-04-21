import numpy as np
import os
import json
import codecs
import tkinter as tk
from .main import cycles_windows_execute,trend_windows_execute
from tkinter import ttk
import include.network.network as net
import include.network.net_constants as netco
import pandas as pd

features_list = ['LABEL','N_MAX','N_MIN','N_AVE','N_IN','N_OUT','A_AVE']

class cycleFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        cycles_set_canvas = tk.Canvas(self,bg="#16253D",width = grandparent.width,height=grandparent.height)
        cycles_set_canvas.grid(row=0,column=0)
        label = tk.Label(cycles_set_canvas,text="Engine Cycles Window Settings",bg=cycles_set_canvas['bg'],fg = "#EFB509")
        label.place(x=1,y=10)

        cycles_window_size_l = tk.Label(cycles_set_canvas,text="Window Size:",bg=cycles_set_canvas['bg'],fg = "#EFB509")
        cycles_window_size_l.place(x=1,y=40)
        cycles_window_size = ttk.Combobox(cycles_set_canvas,state="readonly")
        cycles_window_size['values'] = (30,60,90,180)
        cycles_window_size.current(len(cycles_window_size['values'])-1)
        cycles_window_size.place(x=111,y=40)

        cycles_window_step_l = tk.Label(cycles_set_canvas,text="Window Step:",bg=cycles_set_canvas['bg'],fg = "#EFB509")
        cycles_window_step_l.place(x=1,y=60)
        cycles_window_step = ttk.Combobox(cycles_set_canvas,state="readonly")
        cycles_window_step['values'] = (3,5,9,10,20,30)
        cycles_window_step.current(0)
        cycles_window_step.place(x=111,y=60)

        cycle_ex_button = tk.Button(cycles_set_canvas,text="Execute",command=lambda:cycles_windows_execute(int(cycles_window_size.get()),int(cycles_window_step.get())))
        cycle_ex_button.place(x=251,y=200)

class trendFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        trend_set_canvas = tk.Canvas(self,bg="#16253D",width = grandparent.width,height=grandparent.height)
        trend_set_canvas.grid(row=0,column=0)
        label = tk.Label(trend_set_canvas,text="Speed Trend Window Settings",bg=trend_set_canvas['bg'],fg = "#EFB509")
        label.place(x=1,y=10)

        trend_window_size_l = tk.Label(trend_set_canvas,text="Window Size:",bg=trend_set_canvas['bg'],fg = "#EFB509")
        trend_window_size_l.place(x=1,y=40)
        trend_window_size = ttk.Combobox(trend_set_canvas,state="readonly")
        trend_window_size['values'] = (30,60,90,180)
        trend_window_size.current(len(trend_window_size['values'])-1)
        trend_window_size.place(x=111,y=40)

        trend_window_step_l = tk.Label(trend_set_canvas,text="Window Step:",bg=trend_set_canvas['bg'],fg = "#EFB509")
        trend_window_step_l.place(x=1,y=60)
        trend_window_step = ttk.Combobox(trend_set_canvas,state="readonly")
        trend_window_step['values'] = (3,5,9,10,20,30)
        trend_window_step.current(1)
        trend_window_step.place(x=111,y=60)

        trend_ex_button = tk.Button(trend_set_canvas,text="Execute",command=lambda:trend_windows_execute(int(trend_window_size.get()),int(trend_window_step.get())))
        trend_ex_button.place(x=251,y=200)

class networksFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        net_canvas = tk.Canvas(self,bg="#16253D",width = grandparent.width,height=grandparent.height)
        net_canvas.grid(row=0,column=0)
        label = tk.Label(net_canvas,text="Neural Network Settings",bg=net_canvas['bg'],fg = "#EFB509")
        label.place(x=1,y=10)

        network_edition_l = tk.Label(net_canvas,text="Edition:",bg=net_canvas['bg'],fg = "#EFB509")
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

        model_l = tk.Label(net_canvas,text="Model:",bg=net_canvas['bg'],fg = "#EFB509")
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

        layers_l = tk.Label(net_canvas,text="Layers:",bg=net_canvas['bg'],fg = "#EFB509")
        layers_l.place(x=1,y=80)
        layers = ttk.Combobox(net_canvas,state="readonly")
        layers['values'] = list(range(1,6))
        layers.current(0)
        layers.place(x=111,y=80)
        
        structure_l = tk.Label(net_canvas,text="Structure:",bg=net_canvas['bg'],fg = "#EFB509")
        structure_l.place(x=1,y=100)
        structure_button = tk.Button(net_canvas,text="Edit Structure",command=lambda:self.gen_structure_box(int(layers.get())))
        structure_button.place(x=111,y=100)

        create_button = tk.Button(net_canvas,text="Create",command=lambda:self.create_network())
        create_button.place(x=111,y=180)

        train_button = tk.Button(net_canvas,text="Training Setup",command=lambda:self.gen_train_box())
        train_button.place(x=111,y=220)

    def gen_structure_box(self,layers):
        _win = structureToplevelGUI(self,layers)

    def gen_train_box(self):
        _win = trainToplevelGUI(self)

    def create_network(self):
        root_path = os.getcwd()+"/models/"+self.network_edition.get()+"/"+self.model.get()
        #obj_text = codecs.open(json_path, 'r', encoding='utf-8').read()
        #b_new = json.loads(obj_text)
        #layers_structure = np.array(b_new["network_structure"])
        #print(layers_structure)
        #if self.network_edition == netco.TREND:
        self.network = net.Network("NN_DT", root_path)
        #elif self.network_edition == netco.CYCLES:
        #    self.network = net.Network("NN_EC", root_path)
        self.network.layers_import(root_path+"/exit.json")
        print(self.network.structure)

class structureToplevelGUI(tk.Toplevel):
    def __init__(self,parent,layers):
        tk.Toplevel.__init__(self)
        self.settingsGUI("Network Structure")
        self.parent = parent
        self.layers = layers
        print(self.layers)
        in_label = tk.Label(self,text="In",bg=self['bg'],fg = "#EFB509")
        in_label.grid(row=0,column=0,sticky="nsew")
        out_label = tk.Label(self,text="Out",bg=self['bg'],fg = "#EFB509")
        out_label.grid(row=0,column=1,sticky="nsew")


        def limitSize(*args):
            print("XAXAXXA")
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
            self.inValues.append(tk.StringVar())
            self.inValues[i].trace('w',limitSize)
            inEntry1 = tk.Entry(self,textvariable=self.inValues[i])
            inEntry1.grid(row=i+1,column=0)

        #for i in range(layers):
            self.outValues.append(tk.StringVar())
            self.outValues[i].trace('w',limitSize)
            outEntry1 = tk.Entry(self,textvariable=self.outValues[i])
            outEntry1.grid(row=i+1,column=1)
        
        
        save_button = tk.Button(self,text="Save",command=lambda:self.layers_export())
        save_button.grid(row=layers+1,column=0)
        
        self.mainloop()

    def settingsGUI(self,title,width=400,heigth=400):
        #self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg="#000000")
        #self.columnconfigure(0,weight=1)
        #self.rowconfigure(0,weight=1)

    def layers_export(self):
        lista = []
        for i in range(self.layers):
            lista.append([self.outValues[i].get(),self.inValues[i].get()])
        self.structure = np.array(lista)
        self.parent.structure = self.structure
        net_ed = self.parent.network_edition.get()
        model = self.parent.model.get()
        self.version_path = os.getcwd()+"/models/"+net_ed+"/"+model
        print(self.version_path)
        if not os.path.exists(self.version_path):
            os.makedirs(self.version_path)
        with open(self.version_path+"/info.txt","w") as file:
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
        file_path = self.version_path+"/exit.json" ## your path variable
        json.dump({"network_structure":b}, codecs.open(file_path, 'w', encoding='utf-8'),indent=4)


class trainToplevelGUI(tk.Toplevel):
    def __init__(self,parent):
        tk.Toplevel.__init__(self)
        self.settingsGUI("Training Setup")
        self.parent = parent
        _label = tk.Label(self,text="Training Parameters",bg=self['bg'],fg = "#EFB509")
        _label.grid(row=0,column=0,sticky="nsew")

        epochs_l = tk.Label(self,text="Epochs:",bg=self['bg'],fg = "#EFB509")
        epochs_l.place(x=1,y=60)
        epochs = ttk.Combobox(self,state="readonly")
        epochs['values'] = (1,5,10,15,20,30,50,80,100)
        epochs.current(5)
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
        mini_batch['values'] = (8,16,32,64)
        mini_batch.current(0)
        mini_batch.place(x=111,y=100)

        fit_df = pd.read_csv(self.parent.network.root_path+"/train_data.csv",usecols=features_list)
  
        train_button = tk.Button(self,text="Train",command=lambda:self.parent.network.train(fit_df,int(epochs.get()),float(learning_rate.get()),int(mini_batch.get())))
        train_button.place(x=111,y=140)
        
        self.mainloop()

    def settingsGUI(self,title,width=400,heigth=200):
        #self.resizable(False, False)
        self.title(title)
        self.width = width
        self.height = heigth
        width_sc = self.winfo_screenwidth()
        heigth_sc = self.winfo_screenheight()
        offset_x = (width_sc-width)/2
        offset_y = (heigth_sc-heigth)/2
        self.geometry("%dx%d+%d+%d" % (width,heigth,offset_x,offset_y))
        self.configure(bg="#000000")
        #self.columnconfigure(0,weight=1)
        #self.rowconfigure(0,weight=1)

    