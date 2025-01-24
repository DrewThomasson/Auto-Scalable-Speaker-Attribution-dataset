import os
import re
import csv
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import threading
import ollama

class BookAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Book Analysis Suite (Ollama)")
        self.geometry("1000x800")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.model_name = "deepseek-r1:1.5b"
        self.current_file = ""
        self.running_process = None
        
        self.create_widgets()
        self.check_dependencies()

    def check_dependencies(self):
        try:
            subprocess.run(['ebook-convert', '--version'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            messagebox.showwarning("Warning", "Calibre's ebook-convert not found. TXT files will still work.")
            self.convert_btn.config(state=tk.DISABLED)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        
        # Conversion Tab
        self.conversion_frame = ttk.Frame(self.notebook)
        self.create_conversion_tab()
        self.notebook.add(self.conversion_frame, text="Ebook Conversion")
        
        # Quote Extraction Tab
        self.quote_frame = ttk.Frame(self.notebook)
        self.create_quote_tab()
        self.notebook.add(self.quote_frame, text="Quote Extraction")
        
        # Speaker Identification Tab
        self.speaker_frame = ttk.Frame(self.notebook)
        self.create_speaker_tab()
        self.notebook.add(self.speaker_frame, text="Speaker Identification")
        
        # CSV Merging Tab
        self.merge_frame = ttk.Frame(self.notebook)
        self.create_merge_tab()
        self.notebook.add(self.merge_frame, text="CSV Merging")
        
        self.notebook.pack(expand=1, fill="both")

    # Conversion Tab Components
    def create_conversion_tab(self):
        frame = self.conversion_frame
        ttk.Label(frame, text="Convert Ebook to Text or Use TXT File").pack(pady=10)
        
        self.convert_btn = ttk.Button(frame, text="Select File", 
                                    command=self.start_conversion)
        self.convert_btn.pack(pady=10)
        
        self.conv_progress = ttk.Progressbar(frame, orient="horizontal", mode="indeterminate")

    def start_conversion(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Ebook Files", "*.epub *.mobi *.pdf"), ("Text Files", "*.txt")]
        )
        if not file_path: return
        
        self.conv_progress.pack(pady=10)
        self.conv_progress.start()
        
        try:
            if file_path.lower().endswith('.txt'):
                self.current_file = file_path
                messagebox.showinfo("Info", "Using TXT file directly")
            else:
                self.current_file = self.convert_ebook(file_path)
                messagebox.showinfo("Success", f"Converted file saved at:\n{self.current_file}")
            
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, self.current_file)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.conv_progress.stop()
            self.conv_progress.pack_forget()

    def convert_ebook(self, file_path, output_folder="converted"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        base_name = os.path.basename(file_path)
        output_file = os.path.splitext(base_name)[0] + '.txt'
        output_path = os.path.join(output_folder, output_file)
        
        result = subprocess.run(
            ['ebook-convert', file_path, output_path],
            capture_output=True,
            text=True,
            check=True
        )
        return output_path

    # Quote Extraction Tab Components
    def create_quote_tab(self):
        frame = self.quote_frame
        ttk.Label(frame, text="Select Text File:").grid(row=0, column=0, padx=5, pady=5)
        
        self.file_entry = ttk.Entry(frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(frame, text="Browse", command=lambda: self.file_entry.insert(0, filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt")]))).grid(row=0, column=2, padx=5, pady=5)
        
        # Manual Delimiters
        ttk.Label(frame, text="Manual Delimiters:").grid(row=1, column=0, padx=5, pady=5)
        self.start_delim = ttk.Entry(frame, width=5)
        self.start_delim.grid(row=1, column=1, padx=5, pady=5)
        self.end_delim = ttk.Entry(frame, width=5)
        self.end_delim.grid(row=1, column=2, padx=5, pady=5)
        
        # Detect Button
        ttk.Button(frame, text="Detect Delimiters", command=self.detect_delimiters).grid(row=2, column=0, columnspan=3, pady=5)
        self.detected_label = ttk.Label(frame, text="Detected Delimiters: ")
        self.detected_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Extract Button
        ttk.Button(frame, text="Extract Quotes", command=self.start_quote_extraction).grid(row=4, column=0, columnspan=3, pady=10)
        self.quote_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.quote_progress.grid(row=5, column=0, columnspan=3, pady=5, sticky="ew")

    def detect_delimiters(self):
        file_path = self.file_entry.get()
        if not file_path:
            messagebox.showerror("Error", "Select a text file first")
            return
        
        try:
            with open(file_path, 'r') as f:
                sample = f.read(2000)
            
            response = ollama.generate(
                model=self.model_name,
                prompt=f"Identify quote delimiters in this text. Only respond with opening and closing symbols separated by a comma:\n\n{sample}"
            )
            delimiters = response['response'].strip().split(',')
            if len(delimiters) == 2:
                self.detected_label.config(text=f"Detected Delimiters: {delimiters[0]} and {delimiters[1]}")
            else:
                messagebox.showwarning("Warning", "Could not detect delimiters. Using default (\")")
                self.detected_label.config(text='Detected Delimiters: " and "')
        except Exception as e:
            messagebox.showerror("Ollama Error", str(e))

    def start_quote_extraction(self):
        if self.running_process and self.running_process.is_alive():
            messagebox.showwarning("Warning", "Another process is already running")
            return
        
        self.running_process = threading.Thread(target=self.extract_quotes)
        self.running_process.start()

    def extract_quotes(self):
        try:
            file_path = self.file_entry.get()
            if not file_path:
                messagebox.showerror("Error", "Select a text file first")
                return
            
            # Get delimiters
            start_d = self.start_delim.get().strip() or self.detected_label.cget("text").split(" ")[3]
            end_d = self.end_delim.get().strip() or self.detected_label.cget("text").split(" ")[5]
            
            with open(file_path, 'r') as f:
                text = f.read()
            
            # Find all quotes
            pattern = re.compile(f'{re.escape(start_d)}.*?{re.escape(end_d)}', re.DOTALL)
            matches = pattern.finditer(text)
            quotes = []
            
            # Progress calculation
            total = len(text)
            self.quote_progress['maximum'] = total
            self.quote_progress['value'] = 0
            
            for match in matches:
                start = match.start()
                end = match.end()
                quotes.append((match.group(), start, end))
                self.quote_progress['value'] = end  # Update progress
            
            # Write quotes.csv
            with open('quotes.csv', 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Quote', 'Start', 'End', 'Is Quote', 'Speaker', 'Reasoning'])
                for quote in quotes:
                    writer.writerow([quote[0], quote[1], quote[2], 'True', '', ''])
            
            # Generate non-quotes
            self.generate_non_quotes(text, quotes)
            messagebox.showinfo("Success", "Quote extraction completed")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.quote_progress['value'] = 0

    def generate_non_quotes(self, text, quotes):
        quotes.sort(key=lambda x: x[1])
        non_quotes = []
        prev_end = 0
        
        for quote in quotes:
            if quote[1] > prev_end:
                snippet = text[prev_end:quote[1]].strip()
                if snippet:
                    non_quotes.append([snippet, prev_end, quote[1], 'False', 'Narrator', ''])
            prev_end = quote[2]
        
        # Add remaining text
        if prev_end < len(text):
            snippet = text[prev_end:].strip()
            if snippet:
                non_quotes.append([snippet, prev_end, len(text), 'False', 'Narrator', ''])
        
        with open('non_quotes.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Text', 'Start', 'End', 'Is Quote', 'Speaker', 'Reasoning'])
            writer.writerows(non_quotes)

    # Speaker Identification Tab Components
    def create_speaker_tab(self):
        frame = self.speaker_frame
        ttk.Label(frame, text="Select Text File:").grid(row=0, column=0, padx=5, pady=5)
        self.txt_entry = ttk.Entry(frame, width=40)
        self.txt_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=lambda: self.txt_entry.insert(0, filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt")]))).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(frame, text="Identify Speakers", command=self.start_speaker_id).grid(row=1, column=0, columnspan=3, pady=10)
        self.speaker_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.speaker_progress.grid(row=2, column=0, columnspan=3, pady=5, sticky="ew")
        self.status_label = ttk.Label(frame, text="")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)

    def start_speaker_id(self):
        if self.running_process and self.running_process.is_alive():
            messagebox.showwarning("Warning", "Another process is already running")
            return
        
        self.running_process = threading.Thread(target=self.identify_speakers)
        self.running_process.start()

    def identify_speakers(self):
        try:
            df = pd.read_csv('quotes.csv')
            speakers = []
            txt_file = self.txt_entry.get()
            
            # Add Reasoning column if missing
            if 'Reasoning' not in df.columns:
                df['Reasoning'] = ''
            
            with open(txt_file, 'r') as f:
                full_text = f.read()
            
            total = len(df)
            self.speaker_progress['maximum'] = total
            self.speaker_progress['value'] = 0
            
            for idx, row in df.iterrows():
                if pd.isna(row['Quote']):
                    continue
                
                # Get context
                start = max(0, row['Start'] - 500)
                end = min(len(full_text), row['End'] + 500)
                context = full_text[start:end]
                
                # Generate prompt
                prompt = f"""Identify the speaker of this quote from the context.
                Known speakers: {', '.join(speakers) if speakers else 'None'}
                Context: {context}
                Quote: {row['Quote']}
                Respond only with the speaker's name."""
                
                # Get response
                speaker, reasoning = self.ollama_query(prompt)
                
                # Update speakers list
                if speaker and speaker not in speakers:
                    speakers.append(speaker)
                
                # Update CSV
                df.at[idx, 'Speaker'] = speaker
                df.at[idx, 'Reasoning'] = reasoning
                self.speaker_progress['value'] = idx + 1
                self.status_label.config(text=f"Processed {idx+1}/{total} quotes")
            
            df.to_csv('quotes.csv', index=False)
            messagebox.showinfo("Success", "Speaker identification completed")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.speaker_progress['value'] = 0

    # CSV Merging Tab Components
    def create_merge_tab(self):
        frame = self.merge_frame
        ttk.Button(frame, text="Merge CSVs", command=self.merge_csv).pack(pady=10)
        self.merge_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.merge_progress.pack(pady=5, fill="x")

    def merge_csv(self):
        try:
            quotes = pd.read_csv('quotes.csv')
            non_quotes = pd.read_csv('non_quotes.csv')
            
            combined = pd.concat([quotes, non_quotes]).sort_values(by='Start')
            combined.to_csv('book.csv', index=False)
            
            self.merge_progress['value'] = 100
            messagebox.showinfo("Success", "Merged file saved as book.csv")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.merge_progress['value'] = 0

    def ollama_query(self, prompt):
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            raw_response = response['response']
            
            # Extract thinking and answer
            thinking_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
            thinking = thinking_match.group(1).strip() if thinking_match else ""
            answer = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            
            return answer, thinking
            
        except Exception as e:
            messagebox.showerror("Ollama Error", f"Failed to get response: {str(e)}")
            raise

if __name__ == "__main__":
    app = BookAnalysisApp()
    app.mainloop()