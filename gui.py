from tkinter import *


class GUI:
    def __init__(self, tk):
        self.tk = tk
        self.menu = Menu(self.tk)
        self.text_widget = Text(self.tk, height=5, width=25)
        self.text_widget.pack(pady=10)
        self.tk.config(menu=self.menu)
        filemenu = Menu(self.menu)
        self.menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='New', command=self.update_text("new is clicked"))
        filemenu.add_command(label='Open...', command=self.update_text("open is clicked"))
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=tk.quit)
        helpmenu = Menu(self.menu)
        self.menu.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='About')

    def update_text(self, words):
        def helper():
            self.text_widget.delete("1.0", END)
            self.text_widget.insert(END, words)
        return helper

def main():
    root = Tk()
    gui = GUI(root)
    mainloop()

if __name__ == "__main__":
    main()