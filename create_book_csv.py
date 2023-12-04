import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def process_csv(progress_var):
    # Step 1: Read the contents of quotes.csv and non_quotes.csv into two lists.
    quotes = []
    with open('quotes.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip the header
        for row in csvreader:
            text = row[0]
            start_location = int(row[1])
            end_location = int(row[2])
            speaker = row[4]  # Assuming "Speaker" is in the 4th column
            quotes.append((text, start_location, end_location, speaker, 'True'))

    results = []
    with open('non_quotes.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip the header
        for row in csvreader:
            text = row[0]
            start_location = int(row[1])
            end_location = int(row[2])
            speaker = row[4]  # Assuming "Speaker" is in the 4th column
            results.append((text, start_location, end_location, speaker, 'False'))

    # Step 2: Merge and sort the two lists by start location.
    combined = quotes + results
    combined.sort(key=lambda x: x[1])  # sort based on start location

    # Step 3: Write the sorted list to book.csv.
    with open('book.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Text', 'Start Location', 'End Location', 'Speaker', 'Is Quote'])
        for row in combined:
            csvwriter.writerow(row)

    progress_var.set(100)
    messagebox.showinfo("Info", "Processing complete! book.csv has been created.")


def create_gui():
    root = tk.Tk()
    root.title("CSV Processor(create book.csv)")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    progress_var = tk.DoubleVar()
    start_button = ttk.Button(frame, text="Start Processing", command=lambda: process_csv(progress_var))
    start_button.grid(row=0, column=0, pady=10)

    progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=300)
    progress_bar.grid(row=1, column=0, pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_gui()

