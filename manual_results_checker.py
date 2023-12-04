import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

class QuoteReviewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Quote Reviewer")

        self.load_csv_button = ttk.Button(master, text="Load quotes.csv", command=self.load_csv)
        self.load_csv_button.pack()

        self.load_txt_button = ttk.Button(master, text="Load TXT File", command=self.load_txt)
        self.load_txt_button.pack()

        self.quote_frame = ttk.Frame(master)
        self.quote_frame.pack()

        self.quote_label = ttk.Label(self.quote_frame, text="", wraplength=300)
        self.quote_label.pack(side=tk.LEFT)

        self.speaker_label = ttk.Label(self.quote_frame, text="", wraplength=100)
        self.speaker_label.pack(side=tk.RIGHT)

        self.context_label = tk.Text(master, wrap='word', height=10, width=50)
        self.context_label.pack()

        self.correct_var = tk.BooleanVar()
        self.correct_check = ttk.Checkbutton(master, text="Correct Attribution", variable=self.correct_var)
        self.correct_check.pack()

        self.next_button = ttk.Button(master, text="Next Quote", command=self.next_quote)
        self.next_button.pack()

        self.save_button = ttk.Button(master, text="Save Changes", command=self.save_csv)
        self.save_button.pack()

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.pack()

        self.df = None
        self.txt_content = None
        self.current_index = 0

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.display_quote_context()
            self.progress_bar.config(maximum=len(self.df))

    def load_txt(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                self.txt_content = file.read()

    def display_quote_context(self):
        if self.df is not None and self.current_index < len(self.df):
            quote = self.df.iloc[self.current_index]['Quote']
            speaker = self.df.iloc[self.current_index]['Speaker']
            context = self.extract_context(quote)
            self.quote_label.config(text=quote)
            self.speaker_label.config(text=f"Speaker: {speaker}")
            self.context_label.delete(1.0, tk.END)
            self.context_label.insert(tk.END, context)
            start_index = self.context_label.search(quote, 1.0, tk.END)
            end_index = f"{start_index}+{len(quote)}c"
            self.context_label.tag_add("highlight", start_index, end_index)
            self.context_label.tag_config("highlight", background="yellow")
            self.progress_var.set(self.current_index)

    def next_quote(self):
        if self.df is not None and self.current_index < len(self.df):
            self.df.at[self.current_index, 'Correct Attribution'] = self.correct_var.get()
            self.current_index += 1
            self.display_quote_context()

    def save_csv(self):
        if self.df is not None:
            correct_count = self.df[self.df['Correct Attribution'] == True].shape[0]
            total_count = len(self.df)
            accuracy_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
            self.df.to_csv('quotes_updated.csv', index=False)
            messagebox.showinfo("Accuracy Rating", f"Final accuracy: {accuracy_percentage:.2f}% correct")


    def extract_context(self, quote):
        if self.txt_content:
            start_index = self.txt_content.find(quote)
            if start_index != -1:
                left_index = max(0, start_index - 500)
                right_index = min(len(self.txt_content), start_index + len(quote) + 500)
                return self.txt_content[left_index:right_index]
        return "Context not available"

def main():
    root = tk.Tk()
    app = QuoteReviewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
