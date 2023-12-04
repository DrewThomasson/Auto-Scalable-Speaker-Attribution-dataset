import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import openai
import random
import re
import csv

def load_csv_data():
    combined_data = []

    # Load quotes.csv
    with open('quotes.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            combined_data.append(row)
    
    # Load non_quotes.csv
    with open('non_quotes.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            combined_data.append(row)

    combined_data.sort(key=lambda x: int(x[1]))  # Sorting by Start Location

    for item in combined_data:
        csv_viewer.insert('', 'end', values=item)

def get_random_chunk(filename, chunk_size=1000):
    with open(filename, 'r') as file:
        content = file.read()
        start_index = random.randint(0, len(content) - chunk_size)
        return content[start_index: start_index + chunk_size]

def decipher_quote_symbols(filename, run_counter=0):
    if run_counter >= 5:
        messagebox.showerror("Error", "The text file might be incorrectly formatted.")
        return '', ''
    
    chunk = get_random_chunk(filename)
    query_for_gpt = {
        "messages": [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": f"Give me a single example of a quote in this chunk of text \n\nSample:\n{chunk} only respond with the quote by a character and nothing else."}
        ]
    }

    openai.api_key = api_key_entry.get()
    response = openai.ChatCompletion.create(model="gpt-4", messages=query_for_gpt["messages"])
    response_text = response['choices'][0]['message']['content']
    start_del, end_del = response_text[0], response_text[-1]

    if start_del in ["'", ".", " "] or end_del in ["'", ".", " "]:
        return decipher_quote_symbols(filename, run_counter + 1)
    
    determined_delimiters_label.config(text=f"Determined Delimiters: {start_del} and {end_del}")
    
    return start_del, end_del

def extract_quotes_and_save(filename):
    if manual_delimiters_var.get() == 1:
        start_del = manual_start_delimiter_entry.get()
        end_del = manual_end_delimiter_entry.get()
    else:
        start_del, end_del = decipher_quote_symbols(filename)

        if not start_del and not end_del:
            return

    with open(filename, 'r') as file:
        text = file.read()

    quotes = re.findall(f'{start_del}.*?{end_del}', text, re.DOTALL)
    csv_filename = 'quotes.csv'

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Quote', 'Start Location', 'End Location', 'Is Quote'])
        for quote in quotes:
            start = text.find(quote)
            end = start + len(quote)
            csv_writer.writerow([quote, start, end, 'True'])

    # Generating non_quotes.csv
    locations = []
    with open(csv_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            start_location = int(row[1])
            end_location = int(row[2])
            locations.append((start_location, end_location))

    locations.sort(key=lambda x: x[0])

    results = []
    prev_end_location = 0
    for start_location, end_location in locations:
        snippet = text[prev_end_location:start_location].strip()
        if snippet:
            results.append((snippet, prev_end_location, start_location, 'False', 'Narrator'))
        prev_end_location = end_location

    snippet = text[prev_end_location:].strip()
    if snippet:
        results.append((snippet, prev_end_location, len(text), 'False', 'Narrator'))

    with open('non_quotes.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker'])
        for row in results:
            csvwriter.writerow(row)

    progress_bar["value"] = 100
    messagebox.showinfo("Information", f"Quotes written to '{csv_filename}'.\nNon-quotes written to 'non_quotes.csv'.")

def select_file():
    file_path = filedialog.askopenfilename()
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

def start_extraction():
    progress_bar["value"] = 0
    extract_quotes_and_save(file_path_entry.get())
    load_csv_data()  # Load data into csv_viewer after extraction

app = tk.Tk()
app.title("Quote Extractor")

frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Select file
ttk.Label(frame, text="Select Text File:").grid(column=0, row=0, sticky=tk.W, pady=5)
file_path_entry = ttk.Entry(frame, width=40)
file_path_entry.grid(column=1, row=0, pady=5)
ttk.Button(frame, text="Browse", command=select_file).grid(column=2, row=0, pady=5, padx=5)

# OpenAI API Key Entry
ttk.Label(frame, text="Enter OpenAI API Key:").grid(column=0, row=1, sticky=tk.W, pady=5)
api_key_entry = ttk.Entry(frame, width=40, show="*")
api_key_entry.grid(column=1, row=1, pady=5)

# Manual Delimiters
manual_delimiters_var = tk.IntVar()
manual_delimiters_check = ttk.Checkbutton(frame, text="Manually enter delimiters", variable=manual_delimiters_var)
manual_delimiters_check.grid(columnspan=3, row=2, sticky=tk.W, pady=5)

ttk.Label(frame, text="Start Delimiter:").grid(column=0, row=3, sticky=tk.W, pady=5)
manual_start_delimiter_entry = ttk.Entry(frame, width=10)
manual_start_delimiter_entry.grid(column=1, row=3, pady=5)

ttk.Label(frame, text="End Delimiter:").grid(column=0, row=4, sticky=tk.W, pady=5)
manual_end_delimiter_entry = ttk.Entry(frame, width=10)
manual_end_delimiter_entry.grid(column=1, row=4, pady=5)

# Start Button
ttk.Button(frame, text="Start Extraction", command=start_extraction).grid(columnspan=3, row=5, pady=10)

# Progress Bar
progress_bar = ttk.Progressbar(frame, orient="horizontal", length=200, mode="determinate")
progress_bar.grid(columnspan=3, row=6, pady=5)

# Determined Delimiters Label
determined_delimiters_label = ttk.Label(frame, text="Determined Delimiters: N/A")
determined_delimiters_label.grid(columnspan=3, row=7, pady=5)

# CSV Viewer
ttk.Label(frame, text="CSV Viewer:").grid(columnspan=3, row=8, sticky=tk.W, pady=5)
csv_viewer = ttk.Treeview(frame, columns=('Text', 'Start Location', 'End Location', 'Is Quote', 'Speaker'), show='headings')
csv_viewer.heading('Text', text='Text')
csv_viewer.heading('Start Location', text='Start Location')
csv_viewer.heading('End Location', text='End Location')
csv_viewer.heading('Is Quote', text='Is Quote')
csv_viewer.heading('Speaker', text='Speaker')
csv_viewer.grid(columnspan=3, row=9, sticky=(tk.W, tk.E, tk.N, tk.S))


app.mainloop()

