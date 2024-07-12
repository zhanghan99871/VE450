import tkinter as tk

def submit():
    # Retrieve the data from the entry widget
    user_input = entry.get()
    # Optional: Clear the entry widget after getting the input
    entry.delete(0, tk.END)
    # Display or use the data (here we update a label as an example)
    result_label.config(text="You entered: " + user_input)

# Create the main window
root = tk.Tk()
root.title("User Input Example")

# Create an Entry widget
entry = tk.Entry(root, width=50)
entry.pack(pady=10)

# Create a Button to submit the input
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack(pady=5)

# Label to display the results
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Start the main event loop
root.mainloop()