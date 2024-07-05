from tkinter import *
import tkinter
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
MAX_WIDTH = 1500
MAX_HEIGHT = 800

class GUI:
    def __init__(self, tk):
        self.h_ratio = 1
        self.w_ratio = 1
        self.x = 0
        self.y = 0
        self.tk = tk
        self.menu = Menu(self.tk)
        # self.text_widget = Text(self.tk, height=5, width=25)
        # self.text_widget.pack(pady=10)
        self.w_width = self.tk.winfo_width()
        self.w_height = self.tk.winfo_height()
        self.create_menu()
        self.canvas = Canvas(self.tk, width=self.w_width, height=self.w_height)
        # self.text_id = self.canvas.create_text(self.w_width, self.w_height, text="x={}, y={}, value=({}, {}, {})".format(self.x, self.y, 0, 0, 0))
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Motion>", self._move)
        self.tk.config(menu=self.menu)
        self.image = None
        self.tk.bind("<Configure>", self._resize)
        # self.pixel_label = Label(self.tk, text="x={}, y={}, value=({}, {}, {})".format(self.x, self.y, 0, 0, 0))

    def create_menu(self):
        filemenu = Menu(self.menu)
        self.menu.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Open...', command=self.open_image)
        filemenu.add_command(label='Save as...', command=self.save_image)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=self.tk.quit)

        toolmenu = Menu(self.menu)
        self.menu.add_cascade(label="Tool", menu=toolmenu)
        toolmenu.add_command(label="Luminance adjustment", command=self.luminance_adjust)
        toolmenu.add_command(label="Gamut clipping", command=self.gamut_clip)
        toolmenu.add_command(label="Rescale to original size", command=self.rescale)
        toolmenu.add_command(label="Color space", command=self.change_color_space)
        toolmenu.add_command(label="Get value", command=self.get_value)

        helpmenu = Menu(self.menu)
        self.menu.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='About')
        helpmenu.add_command(label="Help")

    def luminance_adjust(self):
        pass

    def gamut_clip(self):
        pass

    def change_color_space(self):
        pass

    def get_value(self):
        pass

    def rescale(self):
        if self.image:
            image = self.image
            self.tk.geometry("{}x{}".format(image.width + 50, image.height + 50))
            photo = ImageTk.PhotoImage(image)
            # If an image is already displayed, remove it
            self.canvas.delete("all")

            # Resize the canvas
            self.canvas.config(width=image.width + 20, height=image.height + 20)

            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=NW, image=photo)

            # Keep a reference to the image to prevent garbage collection
            self.canvas.image = photo

            self.text_id = self.canvas.create_text(100, image.height + 20,
                                                   text="x={}, y={}, value=({}, {}, {})".format(self.x, self.y, 0, 0,
                                                                                                0))
        else:
            pass

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Open Image File",
                                               filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if file_path:
            self.display_image(file_path)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            self.image.save(file_path)

    def display_image(self, path):
        image = Image.open(path)
        if image.width > MAX_WIDTH or image.height > MAX_HEIGHT:
            image = image.resize((min((MAX_WIDTH, image.width)), min((MAX_HEIGHT, image.height))))
        photo = ImageTk.PhotoImage(image)
        self.tk.geometry("{}x{}".format(image.width+50, image.height+50))
        self.image = image
        # If an image is already displayed, remove it
        self.canvas.delete("all")

        # Resize the canvas
        self.canvas.config(width=image.width+20, height=image.height+20)

        # Display the image on the canvas
        self.canvas.create_image(0, 0, anchor=NW, image=photo)

        # Keep a reference to the image to prevent garbage collection
        self.canvas.image = photo

        self.text_id = self.canvas.create_text(100, image.height+20,
                                text="x={}, y={}, value=({}, {}, {})".format(self.x, self.y, 0, 0, 0))

    def display_pixel(self):
        if self.image:
            value = np.array(self.image)[int(self.y * self.h_ratio)][int(self.x * self.w_ratio)]
            self.canvas.itemconfig(self.text_id,
                                   text="x={}, y={}, value=({}, {}, {})".format(int(self.x * self.w_ratio),
                                                                                int(self.y * self.h_ratio), value[0],
                                                                                value[1], value[2]))
    def auto_resize(self):
        if self.image:
            image = self.image.resize((self.w_width-10, self.w_height-50))
            photo = ImageTk.PhotoImage(image)
            # If an image is already displayed, remove it
            self.canvas.delete("all")

            # Resize the canvas
            # self.canvas.config(width=self.w_width, height=self.w_height)

            # Display the image on the canvas
            self.canvas.create_image(0, 0, anchor=NW, image=photo)

            # Keep a reference to the image to prevent garbage collection
            self.canvas.image = photo

            self.text_id = self.canvas.create_text(100, image.height + 20,
                                                   text="x={}, y={}, value=({}, {}, {})".format(self.x, self.y, 0, 0,
                                                                                                0))

            image_w, image_h = self.image.size
            w, h = image.size
            self.w_ratio, self.h_ratio = image_w / w, image_h / h
    def _resize(self, event):
        if event.width == self.w_width and event.height == self.w_height:
            pass
        else:
            self.w_width = event.width
            self.w_height = event.height
            print(self.w_width)
            print(self.w_height)
            self.auto_resize()

    def _move(self, event):
        self.x = event.x
        self.y = event.y
        self.display_pixel()




def main():
    root = Tk()
    root.geometry("400x300")
    gui = GUI(root)
    mainloop()


if __name__ == "__main__":
    main()
