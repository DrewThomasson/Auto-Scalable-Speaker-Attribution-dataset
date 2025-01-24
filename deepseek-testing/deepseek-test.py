import os
import re
import csv
import queue
import time
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import threading
import ollama

class BookAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Book Analyzer")
        self.geometry("800x600")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.model = "deepseek-r1:1.5b"
        self.current_file = ""
        self.token_queue = queue.Queue()
        self.generation_timeout = 120  # Seconds per response
        self.max_retries = 2  # Max retries per quote
        self.abort_flag = threading.Event()
        
        self.create_widgets()
        self.check_calibre()
        self.after(50, self.update_display)

    def check_calibre(self):
        try:
            subprocess.run(['ebook-convert', '--version'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            messagebox.showinfo("Info", "Ebook conversion requires Calibre")

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill='both', expand=True)

        # File Selection
        file_frame = ttk.LabelFrame(main_frame, text="1. Select Book", padding=10)
        file_frame.pack(fill='x', pady=5)
        
        self.file_entry = ttk.Entry(file_frame, width=40)
        self.file_entry.pack(side='left', padx=5)
        
        ttk.Button(file_frame, text="Browse", 
                 command=self.select_file).pack(side='left')

        # Delimiter Options
        delim_frame = ttk.LabelFrame(main_frame, text="2. Quote Settings (optional)", padding=10)
        delim_frame.pack(fill='x', pady=5)
        
        ttk.Label(delim_frame, text="Start:").pack(side='left')
        self.start_delim = ttk.Entry(delim_frame, width=4)
        self.start_delim.pack(side='left', padx=5)
        
        ttk.Label(delim_frame, text="End:").pack(side='left')
        self.end_delim = ttk.Entry(delim_frame, width=4)
        self.end_delim.pack(side='left', padx=5)

        # Processing
        ttk.Button(main_frame, text="Analyze Book", 
                 command=self.start_analysis).pack(pady=10)
        
        self.progress = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill='x')
        
        self.status = ttk.Label(main_frame, text="Ready")
        self.status.pack(pady=5)

        # Streaming Display
        stream_frame = ttk.LabelFrame(main_frame, text="Live Generation", padding=10)
        stream_frame.pack(fill='both', expand=True, pady=5)
        
        self.stream_text = tk.Text(stream_frame, wrap=tk.WORD, height=10)
        scrollbar = ttk.Scrollbar(stream_frame, orient=tk.VERTICAL, command=self.stream_text.yview)
        self.stream_text.configure(yscrollcommand=scrollbar.set)
        self.stream_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Books", "*.epub *.mobi *.pdf *.txt")]
        )
        if path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def start_analysis(self):
        if not self.file_entry.get():
            messagebox.showerror("Error", "Select a book file first")
            return
        
        self.abort_flag.clear()
        thread = threading.Thread(target=self.full_analysis)
        thread.start()

    def full_analysis(self):
        try:
            self.status.config(text="Starting analysis...")
            txt_path = self.convert_to_text(self.file_entry.get())
            self.process_quotes(txt_path)
            self.identify_speakers(txt_path)
            self.merge_results()
            messagebox.showinfo("Success", "Analysis complete!\nOutput: book.csv")
            self.status.config(text="Ready")
            self.progress['value'] = 0
        except Exception as e:
            if not self.abort_flag.is_set():
                messagebox.showerror("Error", str(e))
                self.status.config(text="Failed")

    def convert_to_text(self, path):
        self.status.config(text="Converting to text...")
        if path.endswith('.txt'):
            return path
        
        output_dir = "converted"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 
                                 os.path.basename(path).rsplit('.', 1)[0] + '.txt')
        
        subprocess.run(['ebook-convert', path, output_path], 
                      check=True, 
                      stdout=subprocess.DEVNULL)
        return output_path

    def process_quotes(self, txt_path):
        self.status.config(text="Extracting quotes...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        start_d = self.start_delim.get().strip() or self.detect_delimiters(text)
        end_d = self.end_delim.get().strip() or self.detect_delimiters(text, end=True)
        
        pattern = re.compile(f'{re.escape(start_d)}.*?{re.escape(end_d)}', re.DOTALL)
        quotes = [m for m in pattern.finditer(text)]
        
        self.write_csv('quotes.csv', quotes, text, is_quote=True)
        self.write_csv('non_quotes.csv', quotes, text, is_quote=False)

    def detect_delimiters(self, text, end=False):
        sample = text[:2000]
        response = ollama.generate(
            model=self.model,
            prompt=f"Identify {'closing' if end else 'opening'} quote symbol in:\n{sample}\nRespond only with the symbol."
        )
        return response['response'].strip()

    def write_csv(self, filename, quotes, text, is_quote):
        entries = []
        prev_end = 0
        
        for match in sorted(quotes, key=lambda x: x.start()):
            if is_quote:
                entries.append([match.group(), match.start(), match.end(), 'True', '', ''])
            else:
                if match.start() > prev_end:
                    snippet = text[prev_end:match.start()].strip()
                    if snippet:
                        entries.append([snippet, prev_end, match.start(), 'False', 'Narrator', ''])
                prev_end = match.end()
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Text', 'Start', 'End', 'Is Quote', 'Speaker', 'Reasoning'])
            writer.writerows(entries)

    def identify_speakers(self, txt_path):
        self.status.config(text="Identifying speakers...")
        quotes = pd.read_csv('quotes.csv')
        speakers = []
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        for idx, row in quotes.iterrows():
            if self.abort_flag.is_set():
                break
            
            start = max(0, row['Start'] - 500)
            end = min(len(full_text), row['End'] + 500)
            context = full_text[start:end]
            
            prompt = f"""Identify speaker from context. Known: {', '.join(speakers) or 'None'}
            Context: {context}
            Quote: {row['Text']}
            Respond only with name."""
            
            speaker, reasoning = self.ask_llm(prompt)
            
            if speaker and speaker not in speakers:
                speakers.append(speaker)
            
            quotes.at[idx, 'Speaker'] = speaker
            quotes.at[idx, 'Reasoning'] = reasoning
            self.progress['value'] = (idx+1)/len(quotes)*100
        
        if not self.abort_flag.is_set():
            quotes.to_csv('quotes.csv', index=False)

    def merge_results(self):
        if self.abort_flag.is_set():
            return
            
        self.status.config(text="Finalizing...")
        quotes = pd.read_csv('quotes.csv')
        non_quotes = pd.read_csv('non_quotes.csv')
        pd.concat([quotes, non_quotes]).sort_values('Start').to_csv('book.csv', index=False)

    def ask_llm(self, prompt):
        full_response = ""
        thinking_text = ""
        speaker = ""
        retries = 0
        success = False

        # Clear previous output
        self.token_queue.put("CLEAR")

        while not success and retries <= self.max_retries and not self.abort_flag.is_set():
            try:
                stop_event = threading.Event()
                timer = threading.Timer(self.generation_timeout, stop_event.set)
                timer.start()

                stream = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    stream=True,
                    options={'num_ctx': 4096}
                )

                start_time = time.time()
                token_count = 0
                
                for chunk in stream:
                    if stop_event.is_set() or token_count > 1000 or self.abort_flag.is_set():
                        raise TimeoutError("Generation aborted")
                    
                    token = chunk['response']
                    full_response += token
                    token_count += 1
                    
                    # Update displays
                    self.token_queue.put(token)
                    print(token, end='', flush=True)
                    
                    # Reset timeout timer
                    timer.cancel()
                    timer = threading.Timer(self.generation_timeout, stop_event.set)
                    timer.start()

                # Successful completion
                timer.cancel()
                success = True

            except (TimeoutError, Exception) as e:
                self.token_queue.put(f"\n⚠️ Error: {str(e)}. Retrying ({retries}/{self.max_retries})...\n")
                print(f"\nError: {str(e)}")
                retries += 1
                full_response = ""
                time.sleep(1)
                
                if retries > self.max_retries:
                    self.token_queue.put("\n❌ Failed after retries. Moving to next quote.\n")
                    return "Unknown", "Generation failed after multiple attempts"

            finally:
                try:
                    timer.cancel()
                except:
                    pass

        if self.abort_flag.is_set():
            return "Aborted", "Process cancelled"

        # Parse response
        thinking_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
        if thinking_match:
            thinking_text = thinking_match.group(1).strip()
            speaker = re.sub(r'<think>.*?</think>', '', full_response).strip()
        else:
            speaker = full_response.strip()

        return speaker, thinking_text

    def update_display(self):
        try:
            while True:
                content = self.token_queue.get_nowait()
                if content == "CLEAR":
                    self.stream_text.delete(1.0, tk.END)
                else:
                    self.stream_text.insert(tk.END, content)
                    self.stream_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(50, self.update_display)

if __name__ == "__main__":
    app = BookAnalyzer()
    app.mainloop()
