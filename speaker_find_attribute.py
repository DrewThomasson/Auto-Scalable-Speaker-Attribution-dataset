import openai
import pandas as pd
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

import os
from openai import OpenAI


api_key = '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'
model = "gpt-4"

# Set your API key here
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)


def generate_outline(text):
    """
    Generates an outline based on the input text.
    """
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Create an outline for the following text:\n\n{text}",
        }
    ],
    model=model,
)




def extract_sentence(start, end, filename):
    with open(filename, 'r') as f:
        content = f.read()
    sentence = content[start:end]
    return sentence

def extract_context(sentence, filename):
    with open(filename, 'r') as f:
        content = f.read()
    start_index = content.find(sentence)
    if start_index == -1:
        return None
    left_index = max(0, start_index - 500)
    right_index = min(len(content), start_index + len(sentence) + 500)
    return content[left_index:right_index]

#def ask_openai(context, sentence, names):
#    message = {
#        "role": "system",
#        "content": "You are a helpful assistant that determines which character is speaking in a story. You answer with only the name and nothing else."
#    }
#    message_user = {
#        "role": "user",
#        "content": f"{context} :who said {sentence} in the text I gave, here is a list of names of characters you have identified so far{names}"
#    }
#    response = openai.ChatCompletion.create(
#        model="gpt-4",
#        messages=[message, message_user]
#    )
#    return response.choices[0].message['content'].strip()


def ask_openai(context, sentence, names):
    """
    Generates an outline based on the input text.
    """
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"""You are a helpful assistant that determines which character is speaking in a story. You answer with only the name and nothing else. {context} :who said {sentence} in the text I gave, here is a list of names of characters you have identified so far{names}""",
        }
    ],
    model=model,
)

    chat_completion = chat_completion.choices[0].message.content
    return chat_completion

def warn_about_costs(txt_file_path):
    file_size = os.path.getsize(txt_file_path)
    if file_size > 0: 
        result = messagebox.askyesno(
            "Cost Warning", 
            "Processing large files like an entire book can be costly. For example, processing the entirety of 'Alice in Wonderland' might cost around $10-13, "
            "though it depends on how many quotes from characters exist in the book. Do you wish to continue?"
        )
        if not result:
            return False
    return True

def main(api_key, txt_file_path, progress_var, status_label, num_of_wanted_requests, message_label):
    openai.api_key = api_key
    df = pd.read_csv('quotes.csv')
    responses = []
    names = []
    request_counter = 0

    for index, row in df.head(num_of_wanted_requests).iterrows():
        sentence = extract_sentence(row['Start Location'], row['End Location'], txt_file_path)
        context = extract_context(sentence, txt_file_path)
        response = ask_openai(context, sentence, names)
        responses.append(response)
        if response not in names:
            names.append(response)
        request_counter += 1
        progress = (request_counter / num_of_wanted_requests) * 100
        progress_var.set(progress)
        status_label['text'] = f"Processing: {request_counter}/{num_of_wanted_requests}"
        if request_counter % 20 == 0:
            message_label['text'] = "Waiting for 1 minute..."
            time.sleep(60)
            message_label['text'] = ""

    df.loc[df.head(num_of_wanted_requests).index, 'Speaker'] = responses
    df.to_csv('quotes.csv', index=False)
    status_label['text'] = "Done"

def on_start(api_key_entry, txt_file_entry, progress_var, status_label, manual_var, manual_entry, message_label):
    api_key = api_key_entry.get()
    txt_file_path = txt_file_entry.get()

    if not api_key:
        messagebox.showerror("Error", "Please enter your OpenAI API key.")
        return
    if not txt_file_path:
        messagebox.showerror("Error", "Please select the text file.")
        return

    if not warn_about_costs(txt_file_path):
        return

    num_of_wanted_requests = len(pd.read_csv('quotes.csv'))
    if manual_var.get():
        try:
            manual_value = int(manual_entry.get())
            if manual_value > num_of_wanted_requests:
                messagebox.showwarning("Warning", f"Number entered ({manual_value}) is greater than rows in CSV ({num_of_wanted_requests}). Using CSV row count.")
            else:
                num_of_wanted_requests = manual_value
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
            return

    status_label['text'] = "Processing..."
    thread = threading.Thread(target=main, args=(api_key, txt_file_path, progress_var, status_label, num_of_wanted_requests, message_label))
    thread.start()

def create_gui():
    root = tk.Tk()
    root.title("OpenAI Speaker Identifier")

    ttk.Label(root, text="OpenAI API Key:").pack(pady=10, padx=10)
    api_key_entry = ttk.Entry(root, width=50, show="*")
    api_key_entry.pack(pady=10, padx=10)

    ttk.Label(root, text="Text File:").pack(pady=10, padx=10)
    txt_file_entry = ttk.Entry(root, width=40)
    txt_file_entry.pack(pady=10, padx=10, side=tk.LEFT)
    txt_file_btn = ttk.Button(root, text="Browse", command=lambda: txt_file_entry.delete(0, tk.END) or txt_file_entry.insert(tk.END, filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])))
    txt_file_btn.pack(pady=10, padx=10, side=tk.RIGHT)

    manual_var = tk.BooleanVar(value=False)
    manual_check = ttk.Checkbutton(root, text="Manually set number of requests", variable=manual_var)
    manual_check.pack(pady=5)
    manual_entry = ttk.Entry(root, width=10)
    manual_entry.pack(pady=5)

    start_btn = ttk.Button(root, text="Start", command=lambda: on_start(api_key_entry, txt_file_entry, progress_var, status_label, manual_var, manual_entry, message_label))
    start_btn.pack(pady=10)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100, length=300)
    progress_bar.pack(pady=10)

    status_label = ttk.Label(root, text="")
    status_label.pack(pady=10)

    message_label = ttk.Label(root, text="")
    message_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
