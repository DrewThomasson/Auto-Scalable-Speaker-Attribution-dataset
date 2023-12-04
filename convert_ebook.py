import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

def calibre_installed():
    """Check if Calibre's ebook-convert tool is available."""
    try:
        subprocess.run(['ebook-convert', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        messagebox.showerror("Error", "Calibre commandline tools are not installed. Please install them to use this feature.")
        return False

def convert_with_calibre(file_path, output_folder="ebooks", output_format="txt"):
    """Convert a file using Calibre's ebook-convert tool."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    base_name = os.path.basename(file_path)
    output_file_name = os.path.splitext(base_name)[0] + '.' + output_format
    output_path = os.path.join(output_folder, output_file_name)

    subprocess.run(['ebook-convert', file_path, output_path])
    return output_path

def process_file():
    if not calibre_installed():
        return

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title='Select Ebook File',
        filetypes=[('Ebook Files', 
                    ('*.cbz', '*.cbr', '*.cbc', '*.chm', '*.epub', '*.fb2', '*.html', '*.lit', '*.lrf', 
                     '*.mobi', '*.odt', '*.pdf', '*.prc', '*.pdb', '*.pml', '*.rb', '*.rtf', '*.snb', 
                     '*.tcr', '*.txt'))]
    )
    
    if not file_path:
        return

    converted_file_path = convert_with_calibre(file_path)
    messagebox.showinfo("Success", f"File converted successfully.\nSaved at: {converted_file_path}")

if __name__ == "__main__":
    process_file()
