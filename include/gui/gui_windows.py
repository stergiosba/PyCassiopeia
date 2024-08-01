import os
import json
import codecs
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from .gui_main import cycles_windows_execute, trend_windows_execute


class windowsFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        grandparent = parent.parent
        windows_set_canvas = tk.Canvas(
            self,
            bg=parent.theme.selected_bg,
            width=grandparent.width,
            height=grandparent.height,
        )
        windows_set_canvas.grid(row=0, column=0)
        label = tk.Label(
            windows_set_canvas,
            text="Cruising Trend Window Settings",
            bg=windows_set_canvas["bg"],
            fg=parent.theme.selected_fg,
        )
        label.place(x=1, y=10)

        trend_window_size_l = tk.Label(
            windows_set_canvas,
            text="Window Size:",
            bg=windows_set_canvas["bg"],
            fg=parent.theme.selected_fg,
        )
        trend_window_size_l.place(x=1, y=40)
        trend_window_size = ttk.Combobox(windows_set_canvas, state="readonly")
        trend_window_size["values"] = (3, 5, 9, 15, 30, 60, 150)
        trend_window_size.current(0)
        trend_window_size.place(x=111, y=40)

        trend_window_step_l = tk.Label(
            windows_set_canvas,
            text="Window Step:",
            bg=windows_set_canvas["bg"],
            fg=parent.theme.selected_fg,
        )
        trend_window_step_l.place(x=1, y=60)
        trend_window_step = ttk.Combobox(windows_set_canvas, state="readonly")
        trend_window_step["values"] = (1, 3, 5, 10, 15, 30, 100)
        trend_window_step.current(0)
        trend_window_step.place(x=111, y=60)

        trend_ex_button = tk.Button(
            windows_set_canvas,
            text="Create Trend Windows",
            command=lambda: trend_windows_execute(
                int(trend_window_size.get()), int(trend_window_step.get())
            ),
            height=2,
            width=20,
        )
        trend_ex_button["bg"] = parent.theme.bg
        trend_ex_button["fg"] = parent.theme.fg
        trend_ex_button.place(x=111, y=100)

        label = tk.Label(
            windows_set_canvas,
            text="Pitch Angle Cycles Window Settings",
            bg=windows_set_canvas["bg"],
            fg=parent.theme.selected_fg,
        )
        label.place(x=401, y=10)

        cycles_window_size_l = tk.Label(
            windows_set_canvas,
            text="Window Size:",
            bg=windows_set_canvas["bg"],
            fg=parent.theme.selected_fg,
        )
        cycles_window_size_l.place(x=401, y=40)
        cycles_window_size = ttk.Combobox(windows_set_canvas, state="readonly")
        cycles_window_size["values"] = (5, 10, 20, 30, 60, 90, 180)
        cycles_window_size.current(0)
        cycles_window_size.place(x=511, y=40)

        cycles_window_step_l = tk.Label(
            windows_set_canvas,
            text="Window Step:",
            bg=windows_set_canvas["bg"],
            fg=parent.theme.selected_fg,
        )
        cycles_window_step_l.place(x=401, y=60)
        cycles_window_step = ttk.Combobox(windows_set_canvas, state="readonly")
        cycles_window_step["values"] = (1, 3, 5, 9, 10, 20)
        cycles_window_step.current(0)
        cycles_window_step.place(x=511, y=60)

        cycle_ex_button = tk.Button(
            windows_set_canvas,
            text="Create Cycles Windows",
            command=lambda: cycles_windows_execute(
                int(cycles_window_size.get()), int(cycles_window_step.get())
            ),
            height=2,
            width=20,
        )
        cycle_ex_button["bg"] = parent.theme.bg
        cycle_ex_button["fg"] = parent.theme.fg
        cycle_ex_button.place(x=511, y=100)
