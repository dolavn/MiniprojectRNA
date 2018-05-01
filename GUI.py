from tkinter import Tk, Canvas, Frame, BOTH

class Surface(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.master.title("Lines")
        self.pack(fill=BOTH, expand=1)

        canvas = Canvas(self)
        canvas.create_line(15, 25, 200, 25)

        canvas.pack(fill=BOTH, expand=1)


root = Tk()
ex = Surface()
root.geometry("400x250+300+300")
root.mainloop()
