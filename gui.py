from tkinter import *
import tkinter
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import colorsys
from adapt_luminance import adapt_luminance
from glare import add_glare
from gamut_clipping import GamutClipper
MAX_WIDTH = 1500
MAX_HEIGHT = 800
color_space = {}
color_space["RGB"] = 0
color_space["HSL"] = 1

class Stack:
    def __init__(self, capacity=20):
        self.items = [None] * capacity  # Preallocate space for simplicity
        self.capacity = capacity
        self.count = 0  # Track the number of items in the buffer
        self.current = 0  # Pointer to the next position to overwrite

    def is_empty(self):
        return self.count == 0

    def push(self, item):
        # Write item at the current position, then move the pointer
        self.items[self.current] = item
        self.current = (self.current + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def pop(self):
        if not self.is_empty():
            # Move the pointer back to the last added item and remove it
            self.current = (self.current - 1 + self.capacity) % self.capacity
            item = self.items[self.current]
            self.items[self.current] = None  # Optional: Clear the spot
            if self.count > 0:
                self.count -= 1
            return item
        raise IndexError("pop from empty stack")

    def peek(self):
        if not self.is_empty():
            # Peek at the last added item without removing it
            last_index = (self.current - 1 + self.capacity) % self.capacity
            return self.items[last_index]
        raise IndexError("peek from empty stack")

    def size(self):
        return self.count


class GUI:
    def __init__(self, tk):
        self.h_ratio = 1
        self.w_ratio = 1
        self.x = 0
        self.y = 0
        self.tk = tk
        self.menu = Menu(self.tk)
        self.color_mode = color_space["RGB"]
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
        self.luminance = None
        self.image_copy = None
        self.back_stack = Stack()
        self.forward_stack = Stack()
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
        toolmenu.add_command(label="Rescale to original size", command=self.rescale)
        toolmenu.add_command(label="Change color space", command=self.change_color_space)
        toolmenu.add_command(label="Get value", command=self.get_value)
        toolmenu.add_command(label="Glare", command=self.create_sub_window_glare)

        luminmenu = Menu(self.menu)
        self.menu.add_cascade(label="Luminance adjustment", menu=luminmenu)
        luminmenu.add_command(label="Luminance adjustment", command=self.luminance_adjust)

        gamutmenu = Menu(self.menu)
        self.menu.add_cascade(label="Gamut clipping", menu=gamutmenu)
        gamutmenu.add_command(label="Gamut clipping", command=self.gamut_clip)
        gamutmenu.add_command(label="Main hue", command=self.maintain_hue)
        gamutmenu.add_command(label="Maintain lightness and hue", command=self.maintain_lightness_hue)

        editmenu = Menu(self.menu)
        self.menu.add_cascade(label="Edit", menu=editmenu)
        editmenu.add_command(label="Undo", command=self.undo)
        editmenu.add_command(label="Redo", command=self.redo)

        helpmenu = Menu(self.menu)
        self.menu.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='About', command=self.create_sub_window_about)
        helpmenu.add_command(label="Help", command=self.create_sub_window_help)

    def undo(self):
        if self.back_stack.size() > 1:
            self.forward_stack.push(self.back_stack.peek().copy())
            self.back_stack.pop()
            self.image = self.back_stack.peek()
            self.refresh()

    def redo(self):
        if self.forward_stack.size() > 0:
            self.back_stack.push(self.forward_stack.peek().copy())
            self.image = self.forward_stack.peek()
            self.forward_stack.pop()
            self.refresh()

    def glare_adjust(self, intensity=1.0):
        self.image_copy = self.image.copy()
        self.image = Image.fromarray(add_glare(np.array(self.image), intensity))
        self.back_stack.push(self.image.copy())
        self.refresh()

    def glare_remove(self):
        self.image = self.image_copy
        self.back_stack.push(self.image.copy())
        self.refresh()

    def luminance_adjust(self):
        self.create_sub_window_luminance_adjust()

    def gamut_clip(self):
        if self.image is not None:
            output_icc_path = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
            clip_intent = GamutClipper.ClipIntent.GAMUT_CLIPPING
            clipped_image = GamutClipper.clip(self.image, clip_intent, output_icc_path)
            self.image = clipped_image
            self.back_stack.push(self.image.copy())
            self.refresh()

    def maintain_hue(self):
        if self.image is not None:
            output_icc_path = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
            clip_intent = GamutClipper.ClipIntent.MAINTAIN_H
            clipped_image = GamutClipper.clip(self.image, clip_intent, output_icc_path)
            self.image = clipped_image
            self.back_stack.push(self.image.copy())
            self.refresh()

    def maintain_lightness_hue(self):
        if self.image is not None:
            output_icc_path = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
            clip_intent = GamutClipper.ClipIntent.MAINTAIN_LH
            clipped_image = GamutClipper.clip(self.image, clip_intent, output_icc_path)
            self.image = clipped_image
            self.back_stack.push(self.image.copy())
            self.refresh()

    def change_color_space(self):
        self.color_mode = (self.color_mode + 1) % 2
        print(self.color_mode)

    def get_value(self):
        self.create_sub_window_get_value()

    def refresh(self):
        image = self.image
        photo = ImageTk.PhotoImage(image)
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

    def create_sub_window_get_value(self): # sub window to get value
        self.sub_window_get_value = Toplevel(self.tk)
        self.sub_window_get_value.title("Sub Window")
        x_label = Label(self.sub_window_get_value, text="x: ")
        x_label.grid(row=0, column=0)
        self.x_entry = Entry(self.sub_window_get_value, width=30)
        self.x_entry.grid(row=0, column=1)
        y_label = Label(self.sub_window_get_value, text="y: ")
        y_label.grid(row=0, column=2)
        self.y_entry = Entry(self.sub_window_get_value, width=30)
        self.y_entry.grid(row=0, column=3)
        result_label = Label(self.sub_window_get_value, text="value is:({}, {}, {})".format(0, 0, 0))
        result_label.grid(row=3, column=2)
        def submit():
            # Retrieve the data from the entry widget
            if self.image is not None:
                x_input = self.x_entry.get()
                y_input = self.y_entry.get()
                rgb = self.get_value_from_color_space(x_input, y_input)
                # Display or use the data (here we update a label as an example)
                result_label.config(text="value is:({}, {}, {})".format(rgb[0], rgb[1], rgb[2]))
        button_to_get = Button(self.sub_window_get_value, text="Enter", command=submit)
        button_to_get.grid(row=1, column=2)

    def create_sub_window_luminance_adjust(self): # sub window to get value
        self.sub_window_lum_adjust = Toplevel(self.tk)
        self.sub_window_lum_adjust.title("Sub Window")
        x_label = Label(self.sub_window_lum_adjust, text="adjusted luminance: ")
        x_label.grid(row=0, column=0)
        self.x_entry = Entry(self.sub_window_lum_adjust, width=30)
        self.x_entry.grid(row=0, column=1)
        def submit():
            # Retrieve the data from the entry widget
            if self.image is not None:
                x_input = self.x_entry.get()
                self.luminance = float(x_input)
                self.image = Image.fromarray(adapt_luminance(self.image, self.luminance))
                self.back_stack.push(self.image.copy())
                self.refresh()
        button_to_get = Button(self.sub_window_lum_adjust, text="Enter", command=submit)
        button_to_get.grid(row=1, column=1)

    def create_sub_window_glare(self):  # sub window to get value
        self.sub_window_glare = Toplevel(self.tk)
        self.sub_window_glare.title("Sub Window")
        x_label = Label(self.sub_window_glare, text="glare intensity (default value 1): ")
        x_label.grid(row=0, column=0)
        self.x_entry = Entry(self.sub_window_glare, width=30)
        self.x_entry.grid(row=0, column=1)

        def submit():
                # Retrieve the data from the entry widget
            if self.image is not None:
                x_input = self.x_entry.get()
                intensity = float(x_input)
                self.glare_adjust(intensity)

        button_to_get = Button(self.sub_window_glare, text="Enter", command=submit)
        button_to_get.grid(row=1, column=1)
        button_to_remove = Button(self.sub_window_glare, text="Remove glare", command=self.glare_remove)
        button_to_remove.grid(row=2, column=1)

    def create_sub_window_help(self): # sub window to get value
        self.sub_window_help = Toplevel(self.tk)
        self.sub_window_help.title("Help")
        message = "To use our program, you first need to open an image file through File->Open...\n" \
                  "After you successfully upload the image, we provide several functions: \n" \
                  "\t1. get the rgb value of specific position: Tool->Get value, then input the x and y of the point\n" \
                  "\t2. add glare effect: Tool->Glare, then input the intensity of glare, you can remove the glare\n" \
                  "\t\tby Remove button\n" \
                  "\t3. adjust the luminance: Luminance adjustment->Luminance adjustment\n" \
                  "\t\tthen input the environment luminance\n" \
                  "\t4. gamut clip: Gamut clipping->\n" \
                  "\t\tif you want the normal gamut clipping, then click Gamut clipping\n" \
                  "\t\tif you want to maintain lightness and hue, then click maintain lightness and hue\n" \
                  "\t\tif you want to maintain hue, then click maintain hue\n" \
                  "\t5. Redo and undo the operation: Edit->Redo/Edit->Undo\n" \
                  "If you finish processing the image, you can save it through File->Save as..."
        help_message = Label(self.sub_window_help, text=message, anchor='w',  justify=LEFT)
        help_message.pack()

    def create_sub_window_about(self): # sub window to get value
        self.sub_window_about = Toplevel(self.tk)
        self.sub_window_about.title("About")
        message = "This is the capstone design of Group 16 from VE450\n" \
                  "Team member include: \n" \
                  "Han Zhang\n" \
                  "Yaqing Zhou\n" \
                  "Boyuan Zhang\n" \
                  "Tingxi Li\n" \
                  "Zhen Xu\n" \
                  "All rights preserved."
        help_message = Label(self.sub_window_about, text=message, anchor='w',  justify=LEFT)
        help_message.pack()


    def rescale(self):
        if self.image is not None:
            self.refresh()
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
            scale = max(image.width/MAX_WIDTH, image.height/MAX_HEIGHT)
            image = image.resize((int(image.width/scale), int(image.height/scale)))
        photo = ImageTk.PhotoImage(image)
        self.tk.geometry("{}x{}".format(image.width+50, image.height+50))
        self.image = image
        self.back_stack.push(self.image.copy())
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
        if self.image is not None:
            value = self.get_value_from_color_space(self.x * self.w_ratio, self.y * self.h_ratio)
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


    def get_value_from_color_space(self, x, y):
        image = np.array(self.image)
        if self.color_mode == color_space["RGB"]:
            return image[int(y)][int(x)]
        elif self.color_mode == color_space["HSL"]:
            return self.rgb_to_hls(image[int(y)][int(x)])
        else:
            return [0, 0, 0]

    def rgb_to_hls(self, rgb):
        r_scaled, g_scaled, b_scaled = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

        # Convert the RGB values to HSL values
        h, l, s = colorsys.rgb_to_hls(r_scaled, g_scaled, b_scaled)

        return h, l, s

def main():
    root = Tk()
    root.geometry("600x400")
    gui = GUI(root)
    mainloop()


if __name__ == "__main__":
    main()
