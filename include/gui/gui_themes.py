class GuiTheme:
    """This is a basic GUI theme class\n
    |-> Arguments(bg,fg,selected_bg,selected_fg)
    """

    def __init__(self, *colors):
        color_codes = {
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF",
            "orange": "#FE8C00",
            "white": "#FFFFFF",
            "black": "#000000",
            "light-blue": "#309fcf",
        }
        self.bg = color_codes[colors[0]]
        self.fg = color_codes[colors[1]]
        if len(colors) == 3:
            self.selected_bg = color_codes[colors[2]]
        else:
            self.selected_bg = color_codes[colors[1]]
        if len(colors) == 4:
            self.selected_bg = color_codes[colors[2]]
            self.selected_fg = color_codes[colors[3]]
        else:
            self.selected_fg = color_codes[colors[0]]
