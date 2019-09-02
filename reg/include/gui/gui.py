import tkinter as tk
from tkinter import ttk
from .gui_windows import windowsFrame
from .gui_networks import classficiationNetworksFrame,controlNetworksFrame
from .gui_simulation import simulationFrame
from .gui_themes import GuiTheme

class MainApplication(ttk.Notebook):
    def __init__(self,parent,*args,**kwargs):
        self.theme = GuiTheme('light-blue','white','white','black')
        ttk.Notebook.__init__(self,parent,*args,**kwargs)
        style = ttk.Style()
        style.theme_create( "yummy", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {
            "configure": {"padding": [50, 20], "background": self.theme.bg, "foreground": self.theme.fg},
            "map":       {"background": [("selected", self.theme.selected_bg)],"foreground": [("selected", self.theme.selected_fg)],
                          "expand": [("selected", [1, 1, 1, 0])] } } } )

        style.theme_use("yummy")
        self.parent = parent
        windows_tab = windowsFrame(self)
        class_networks_tab = classficiationNetworksFrame(self)
        control_network_tab = controlNetworksFrame(self)
        simulation_tab = simulationFrame(self)

        self.add(windows_tab, text = "Windows")
        self.add(class_networks_tab, text = "Classification NET")
        self.add(control_network_tab, text= "Control NET")
        self.add(simulation_tab, text= "Simulation")

class GUI(tk.Tk):
    def __init__(self,title="Cassiopeia Control"):
        tk.Tk.__init__(self)
        self.settingsGUI(title)
        MainApplication(self).pack(side="top", fill="both", expand=True)
        self.mainloop()

    def settingsGUI(self,title,width=800,heigth=400):
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
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
