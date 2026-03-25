import re
import os
import langdetect
from langdetect import detect, DetectorFactory

# Ensures consistent language detection results
DetectorFactory.seed = 0

def clean_txt_file(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found.")
        return

    with open(input_filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Split by the "===" separators to isolate pages
    entries = re.split(r'={10,}', content)
    
    cleaned_paragraphs = []

    for entry in entries:
        lines = entry.split('\n')
        page_text_parts = []
        
        for line in lines:
            line = line.strip()
            
            # 2. Skip Metadata, Redirects, and specific ID numbers
            # (Matches 'URL:', 'TITLE:', 'WORDS:', '_RedirectToLoginPage_', and long digits)
            if not line or any(meta in line for meta in ['URL:', 'TITLE:', 'WORDS:', '_RedirectToLoginPage_']):
                continue
            
            # 3. Remove all numbers and special characters (Keep only letters and basic punctuation)
            # This handles strings like '963258741!' and '147852369'
            line = re.sub(r'[^a-zA-Z\s.,!?]', '', line)
            
            # Clean up extra whitespace
            line = " ".join(line.split())
            
            if len(line) > 30:  # Threshold to filter out menu items/titles
                page_text_parts.append(line)

        # Combine lines into a single paragraph per page
        full_paragraph = " ".join(page_text_parts)

        # 4. English Language Filter
        if full_paragraph:
            try:
                # Detect language; only keep if English
                if detect(full_paragraph) == 'en':
                    cleaned_paragraphs.append(full_paragraph)
            except:
                continue

    # 5. Write to output file
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        for p in cleaned_paragraphs:
            out_f.write(p + "\n\n")
    
    print(f"Success! Cleaned {len(cleaned_paragraphs)} paragraphs into {output_filename}")

# --- SET YOUR FILENAMES HERE ---
input_file = "/Users/jahanvigajera/Desktop/Assignment/NLU_assignment/Assignment2/problem1/iitj_complete_data/all_text.txt"  # Change this to your actual filename
output_file = "cleaned_dataset.txt"

clean_txt_file(input_file, output_file)