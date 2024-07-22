import tkinter as tk

def on_scale(val):
    # This function is triggered when the slider is moved
    label.config(text=f"Current Value: {int(float(val))}")

def main():
    root = tk.Tk()
    root.title("Scale Example")
    root.geometry("300x200")

    # Create a Scale widget
    scale = tk.Scale(root, from_=0, to=100, orient='horizontal', command=on_scale)
    scale.pack(padx=20, pady=20)

    # Create a Label widget to display the current value of the scale
    global label
    label = tk.Label(root, text="Current Value: 0")
    label.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
