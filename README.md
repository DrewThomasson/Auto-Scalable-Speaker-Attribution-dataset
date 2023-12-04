\# Auto-Scalable Speaker Attribution Dataset

This is my attempt at implementing a method for scaling speaker attribution datasets for literature using GPT or any other capable Large Language Model (LLM).

The hope for this is to use this methology to create a speaker attribution dataset for nearly all books found in project gutenburg

## How to run
- python run_gui.py


<img width="1019" alt="Screenshot 2023-12-04 at 5 06 55 PM" src="https://github.com/DrewThomasson/Auto-Scalable-Speaker-Attribution-dataset/assets/126999465/f93fb7b1-c741-4540-a647-65d1f8a49e61">

<img width="806" alt="Screenshot 2023-12-04 at 5 07 36 PM" src="https://github.com/DrewThomasson/Auto-Scalable-Speaker-Attribution-dataset/assets/126999465/eb2157c9-ff21-4676-8e93-916cba276149">


## manual_results_checker.py
<img width="386" alt="Screenshot 2023-12-04 at 5 03 13 PM" src="https://github.com/DrewThomasson/Auto-Scalable-Speaker-Attribution-dataset/assets/126999465/388cc144-b9df-4e10-b6ae-5ee9431eae53">

-once you click the "save changes" button itll show you the accuracy rating in a popup like so:
<img width="259" alt="Screenshot 2023-12-04 at 5 40 00 PM" src="https://github.com/DrewThomasson/Auto-Scalable-Speaker-Attribution-dataset/assets/126999465/84d87a71-6e9e-41c6-8856-54b240c1b8a6">


- This python script will give you a easy to use gui to manually check the output results of the speaker attribution via LLM
- give it the refrence txt file and then give it the quotes.csv file that was generated after running the speaker_find_attribute.py

## Test Results

For the first test, I used a snippet from "Guardians of Ga'Hoole" in the `ebooks` folder. The results are impressive:

- GPT-4 achieved a remarkable 98.33% accuracy rate for speaker attributio for the first run.
- On the second run of the same piece of text GPT achieved 96.67% accuracy rating.
- given this it appears to still have a arguable very high accuracy rate these could be imporived through improved prompting or increasing the context length given to the LLM
- the second reults can be found under the file "quotes_updated.csv" in the ebooks folder in this repo

## Speaker Attribution Results

I meticulously reviewed each speaker attribution assigned by GPT-4 and categorized the results for all 60 quotes found in the snippet from "Guardians of Ga'Hoole" as follows:

- **True:** Correct attribution
- **False:** Incorrect attribution
- **True/Incorrect Quotation:** The quote may not have been said by a character, but if assigned, this would be the correct answer.

### Attribution Breakdown

- True
- True/Incorrect Quotation
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- False
- True/Incorrect Quotation
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
- True
