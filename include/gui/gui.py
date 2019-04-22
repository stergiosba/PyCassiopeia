import tkinter as tk
from tkinter import ttk
from .gui_frames import cycleFrame,trendFrame,networksFrame

class MainApplication(ttk.Notebook):
    def __init__(self,parent,*args,**kwargs):
        ttk.Notebook.__init__(self,parent,*args,**kwargs)
        style = ttk.Style()
        style.theme_create( "yummy", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {
            "configure": {"padding": [50, 20], "background": "#16253D", "foreground": "#EFB509"},
            "map":       {"background": [("selected", "#EFB509")],"foreground": [("selected", "#16253D")],
                          "expand": [("selected", [1, 1, 1, 0])] } } } )

        style.theme_use("yummy")
        self.parent = parent
        
        trend_tab = trendFrame(self)
        cycle_tab = cycleFrame(self)

        network_tab = networksFrame(self)

        self.add(trend_tab, text = "Trend Window")
        self.add(cycle_tab, text = "Cycle Window")
        self.add(network_tab, text = "Network Window")

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
