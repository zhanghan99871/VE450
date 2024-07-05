import tkinter as tk

def resize_window():
    root.geometry("600x400")

root = tk.Tk()
root.geometry("400x300")  # Set initial size
root.resizable(True, True)  # Allow resizing in both directions
root.minsize(200, 150)  # Set minimum size
root.maxsize(800, 600)  # Set maximum size

button = tk.Button(root, text="Resize Window", command=resize_window)
button.pack(pady=20)

root.mainloop()
