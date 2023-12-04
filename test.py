import tkinter as tk
from tkinter import ttk
import pandas as pd

class QuoteReviewerApp:
    def __init__(self, master):
        self.master = master
        master.title("Quote Reviewer")

        self.load_button = ttk.Button(master, text="Load CSV", command=self.load_csv)
        self.load_button.pack()

        self.quote_label = ttk.Label(master, text="", wraplength=400)
        self.quote_label.pack()

        self.context_label = ttk.Label(master, text="", wraplength=400)
        self.context_label.pack()

        self.correct_var = tk.BooleanVar()
        self.correct_check = ttk.Checkbutton(master, text="Correct Attribution", variable=self.correct_var)
        self.correct_check.pack()

        self.next_button = ttk.Button(master, text="Next Quote", command=self.next_quote)
        self.next_button.pack()

        self.save_button = ttk.Button(master, text="Save Changes", command=self.save_csv)
        self.save_button.pack()

        self.df = None
        self.current_index = 0

    def load_csv(self):
        # Load the CSV file and display the first quote and context
        self.df = pd.read_csv('quotes.csv')
        self.display_quote_context()

    def display_quote_context(self):
        if self.df is not None and self.current_index < len(self.df):
            quote = self.df.iloc[self.current_index]['Quote']
            context = self.extract_context(quote)  # Implement this method to extract context
            self.quote_label.config(text=quote)
            self.context_label.config(text=context)

    def next_quote(self):
        # Update the dataframe with the current checkbox value
        if self.df is not None and self.current_index < len(self.df):
            self.df.at[self.current_index, 'Correct Attribution'] = self.correct_var.get()
            self.current_index += 1
            self.display_quote_context()

    def save_csv(self):
        # Save the updated dataframe to the CSV file
        if self.df is not None:
            self.df.to_csv('quotes_updated.csv', index=False)

    def extract_context(self, quote):
        # Placeholder for context extraction logic
        return "Context for the quote"

def main():
    root = tk.Tk()
    app = QuoteReviewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
