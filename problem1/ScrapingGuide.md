# Complete IIT Jodhpur Website Crawler Guide
## Extract ALL Text Data from iitj.ac.in

---

## 🎯 Overview

This package provides two powerful crawlers to extract **all text data** from the entire IIT Jodhpur website:

1. **full_site_crawler.py** - Advanced crawler with resume capability and detailed reports
2. **quick_full_crawler.py** - Simple one-command crawler for quick extraction

Both crawlers will:
- Follow all internal links automatically
- Extract text from every accessible page
- Avoid duplicates
- Save progress periodically
- Generate comprehensive reports

---

## 🚀 Quick Start (Easiest Method)

### Option 1: Quick Crawler (Recommended for Beginners)

```bash
# Install requirements
pip install requests beautifulsoup4

# Run the crawler
python quick_full_crawler.py
```

**What happens:**
1. Asks for confirmation
2. Crawls up to 500 pages (you can change this)
3. Saves everything to `iitj_full_site/` directory
4. Takes approximately 15-30 minutes

**Output files:**
- `all_pages.json` - All pages with metadata
- `all_text.txt` - Pure text from all pages
- `summary.txt` - Statistics and summary

---

### Option 2: Full-Featured Crawler (Advanced)

```bash
python full_site_crawler.py
```

**Features:**
- Can resume if interrupted
- Unlimited pages (or set a limit)
- More detailed reports
- Progress tracking
- Better error handling

---

## 📖 Detailed Instructions

### Step 1: Install Dependencies

```bash
pip install requests beautifulsoup4 lxml
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Crawler

#### For Quick Extraction (500 pages):

```bash
python quick_full_crawler.py
# Choose 'y' to start
# Wait 15-30 minutes
# Done!
```

#### For Complete Site (unlimited):

```bash
python full_site_crawler.py
# Use default settings (y)
# Choose 'n' for resume (first time)
# Wait 1-3 hours depending on site size
# Press Ctrl+C anytime to save and stop
```

---

## ⚙️ Configuration Options

### Quick Crawler

Edit the script or provide parameters:

```python
# In quick_full_crawler.py, modify:
crawl_iitj_website(
    max_pages=1000,  # Change this number
    output_dir='my_custom_folder'
)
```

### Full Crawler

Interactive configuration:

```
Use default settings? (y/n): n

Output directory [iitj_complete_data]: my_data
Max pages (press Enter for unlimited): 1000
Delay between requests in seconds [1.5]: 2.0
Resume from previous crawl if exists? (y/n): n
```

---

## 📊 Understanding the Output

### Directory Structure

```
iitj_complete_data/
├── documents.json          # All scraped pages (structured)
├── all_text.txt           # Combined text from all pages
├── all_urls.txt           # List of all URLs crawled
├── summary.txt            # Detailed statistics
├── structured_data.json   # Metadata + documents
├── progress.json          # Crawling statistics
└── crawler_state.pkl      # State for resuming
```

### Example Output Files

#### all_text.txt
```
================================================================================
URL: https://iitj.ac.in/department/cse
TITLE: Computer Science & Engineering - IIT Jodhpur
WORDS: 1234
================================================================================

The Department of Computer Science and Engineering at IIT Jodhpur...
```

#### summary.txt
```
IIT JODHPUR COMPLETE WEBSITE CRAWL - SUMMARY
================================================================================

Crawl started: 2026-03-08T10:00:00
Crawl completed: 2026-03-08T12:30:00

Pages successfully crawled: 487
Pages failed: 13
Total words extracted: 234,567
Average words per page: 481.6
```

#### documents.json
```json
[
  {
    "url": "https://iitj.ac.in/department/cse",
    "title": "Computer Science & Engineering",
    "text": "Full text content here...",
    "text_length": 5432,
    "word_count": 987,
    "headings": [
      {"level": 1, "text": "About CSE"},
      {"level": 2, "text": "Research Areas"}
    ],
    "metadata": {
      "description": "CSE department at IIT Jodhpur",
      "keywords": "computer science, IIT Jodhpur"
    },
    "timestamp": "2026-03-08T10:15:23",
    "links_count": 45
  }
]
```

---

## 🔄 Resuming a Crawl

If your crawl gets interrupted, you can resume:

### With Full Crawler:

```bash
python full_site_crawler.py
# When asked "Resume from previous crawl?" 
# Choose: y
```

The crawler will:
- Load previously visited URLs
- Continue from where it stopped
- Avoid re-crawling pages

### With Quick Crawler:

The quick crawler doesn't support resume, but it saves progress every 50 pages, so you won't lose much data.

---

## 📈 Expected Results

### Typical IIT Jodhpur Website Statistics:

- **Total Pages**: 300-800 pages (varies)
- **Total Words**: 150,000 - 400,000 words
- **Crawl Time**: 1-3 hours (with 1.5s delay)
- **File Sizes**: 
  - JSON: 10-50 MB
  - Text: 5-20 MB

### Actual numbers will depend on:
- How many pages are publicly accessible
- How much content is on each page
- Server response times

---

## ⚡ Performance Tips

### Speed vs Respect

**Faster crawling** (use with caution):
```python
delay=0.5  # Half second delay
```

**More respectful** (recommended):
```python
delay=2.0  # Two second delay
```

### Limit Pages for Testing

Test first with a small number:
```python
max_pages=50  # Just 50 pages to test
```

### Parallel Crawling (Advanced)

Not recommended for this use case, but possible with ThreadPoolExecutor.

---

## 🛠️ Troubleshooting

### Problem: Crawler stops with timeout errors

**Solution:**
```python
# In the script, increase timeout:
response = session.get(url, timeout=30)  # Increase from 15 to 30
```

### Problem: Too many failed pages

**Solution:**
- Check internet connection
- Increase delay between requests
- Some pages may require authentication

### Problem: Out of memory

**Solution:**
- Set a `max_pages` limit
- Process in batches
- Run on a machine with more RAM

### Problem: Site blocks the crawler

**Solution:**
```python
# Change User-Agent to look more like a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...'
}
```

---

## 📊 Post-Processing the Data

### Extract Specific Information

```python
import json

# Load the data
with open('iitj_complete_data/documents.json', 'r') as f:
    docs = json.load(f)

# Find all department pages
dept_pages = [d for d in docs if '/department/' in d['url']]

# Get all page titles
titles = [d['title'] for d in docs]

# Calculate statistics
total_words = sum(d['word_count'] for d in docs)
print(f"Total words: {total_words:,}")
```

### Convert to DataFrame

```python
import pandas as pd
import json

with open('iitj_complete_data/documents.json', 'r') as f:
    docs = json.load(f)

df = pd.DataFrame(docs)
df.to_excel('iitj_data.xlsx', index=False)
```

### Search for Keywords

```python
import json

with open('iitj_complete_data/documents.json', 'r') as f:
    docs = json.load(f)

# Find pages mentioning "machine learning"
ml_pages = [d for d in docs if 'machine learning' in d['text'].lower()]
print(f"Found {len(ml_pages)} pages about machine learning")
```

---

## 🎯 Use Cases

### 1. Build a Search Engine
Use the extracted text to build a custom search for IIT Jodhpur content.

### 2. Create a Chatbot
Feed the text data to an LLM to create an IIT Jodhpur information chatbot.

### 3. Content Analysis
Analyze what topics are most covered on the website.

### 4. Research Dataset
Use for NLP research, text mining, or information extraction.

### 5. Archive
Keep a snapshot of the website content.

---

## ⚠️ Important Notes

### Ethical Considerations

1. **Respect robots.txt**: Check https://iitj.ac.in/robots.txt
2. **Rate limiting**: Don't overwhelm the server (use delays)
3. **Terms of service**: Ensure you're allowed to scrape
4. **Data usage**: Use responsibly and give credit

### Legal Considerations

- This is publicly accessible information
- Still, check terms of service
- Don't use for commercial purposes without permission
- Respect copyright on content

### Technical Considerations

- Some pages may require JavaScript
- Some content may be behind authentication
- Dynamic content may not be captured
- The crawler only gets HTML text, not images/videos

---

## 🔧 Advanced Customization

### Skip Certain Sections

```python
# In is_valid_url function, add:
if '/news/' in url or '/events/' in url:
    return False  # Skip news and events
```

### Extract Specific Elements Only

```python
# Instead of get_text(), extract specific divs:
main_content = soup.find('div', class_='main-content')
if main_content:
    text = main_content.get_text()
```

### Save in Different Format

```python
# Save as CSV
import csv
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['url', 'title', 'text', 'word_count'])
    writer.writeheader()
    writer.writerows(documents)
```

---

## 📞 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Test with `max_pages=10` first
3. Verify internet connection
4. Check if site is accessible in browser
5. Try the quick_full_crawler.py first

---

## 📝 Summary Commands

### Easiest Way:
```bash
pip install requests beautifulsoup4
python quick_full_crawler.py
# Press 'y' and wait
```

### Full Control:
```bash
pip install requests beautifulsoup4 lxml
python full_site_crawler.py
# Configure as needed
```

### Resume After Interruption:
```bash
python full_site_crawler.py
# Choose 'y' when asked to resume
```

---

**Happy Crawling! 🕷️📊**

---

## 📄 Sample Workflow

1. **Initial test run**:
   ```bash
   python quick_full_crawler.py
   # Set max_pages to 50
   ```

2. **Review output**:
   ```bash
   cat iitj_full_site/summary.txt
   ```

3. **Full crawl**:
   ```bash
   python full_site_crawler.py
   # Use unlimited pages
   ```

4. **Process data**:
   ```python
   import json
   with open('iitj_complete_data/documents.json') as f:
       data = json.load(f)
   # Do your analysis
   ```

5. **Create report**:
   - Use the extracted text for your project
   - Cite the source appropriately
   - Store safely for future use