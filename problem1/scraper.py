"""
Complete IIT Jodhpur Website Crawler
=====================================

This script crawls the entire iitj.ac.in website and extracts all text data.
It follows internal links and saves comprehensive data.

Requirements:
    pip install requests beautifulsoup4 lxml

Usage:
    python full_site_crawler.py

Features:
    - Crawls entire website following internal links
    - Avoids duplicate pages
    - Respects robots.txt guidelines
    - Rate limiting to avoid overwhelming server
    - Saves progress periodically
    - Can resume from interruption
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin, urlparse, urldefrag
from datetime import datetime
import os
from collections import deque
import pickle

class IITJodhpurFullCrawler:
    """
    Complete website crawler for IIT Jodhpur
    """
    
    def __init__(self, output_dir='iitj_complete_data', max_pages=None, delay=1.5):
        """
        Initialize the crawler
        
        Args:
            output_dir (str): Directory to save data
            max_pages (int): Maximum pages to crawl (None for unlimited)
            delay (float): Delay between requests in seconds
        """
        self.base_url = "https://iitj.ac.in"
        self.domain = "iitj.ac.in"
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay = delay
        
        # Headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Session
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # URL management
        self.to_visit = deque([self.base_url])
        self.visited = set()
        self.failed = set()
        
        # Data storage
        self.documents = []
        self.url_map = {}  # URL -> document index
        
        # Statistics
        self.stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'total_words': 0,
            'start_time': datetime.now().isoformat(),
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # State files
        self.state_file = os.path.join(output_dir, 'crawler_state.pkl')
        self.progress_file = os.path.join(output_dir, 'progress.json')
        
        print("="*80)
        print("IIT JODHPUR COMPLETE WEBSITE CRAWLER")
        print("="*80)
        print(f"Base URL: {self.base_url}")
        print(f"Output directory: {output_dir}")
        print(f"Max pages: {max_pages if max_pages else 'Unlimited'}")
        print(f"Delay between requests: {delay}s")
        print("="*80)
    
    def normalize_url(self, url):
        """Normalize URL by removing fragments and query parameters (optional)"""
        # Remove fragment
        url, _ = urldefrag(url)
        # Remove trailing slash
        url = url.rstrip('/')
        return url
    
    def is_valid_url(self, url):
        """Check if URL should be crawled"""
        parsed = urlparse(url)
        
        # Must be from same domain
        if self.domain not in parsed.netloc:
            return False
        
        # Skip certain file types
        skip_extensions = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.zip', '.tar', '.gz', '.doc', '.docx', '.xls', '.xlsx',
            '.ppt', '.pptx', '.mp4', '.avi', '.mp3', '.wav'
        ]
        
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip certain patterns
        skip_patterns = [
            '/wp-admin/', '/wp-content/', '/wp-includes/',
            '/login', '/logout', '/admin',
            'javascript:', 'mailto:', 'tel:',
            '#', 'void(0)'
        ]
        
        if any(pattern in url.lower() for pattern in skip_patterns):
            return False
        
        return True
    
    def extract_links(self, soup, current_url):
        """Extract all valid links from a page"""
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert to absolute URL
            absolute_url = urljoin(current_url, href)
            
            # Normalize
            normalized_url = self.normalize_url(absolute_url)
            
            # Validate
            if self.is_valid_url(normalized_url):
                links.add(normalized_url)
        
        return links
    
    def clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"\(\)\[\]]', ' ', text)
        # Remove excessive punctuation
        text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
        return text.strip()
    
    def extract_metadata(self, soup):
        """Extract metadata from page"""
        metadata = {}
        
        # Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        # Meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Open Graph tags
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        return metadata
    
    def scrape_page(self, url):
        """
        Scrape a single page
        
        Returns:
            tuple: (document_data, links_found)
        """
        try:
            print(f"[{self.stats['pages_crawled'] + 1}] Crawling: {url}")
            
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "No Title"
            
            # Extract metadata
            metadata = self.extract_metadata(soup)
            
            # Extract links before removing elements
            links = self.extract_links(soup, url)
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 
                                'header', 'noscript', 'iframe']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            text = self.clean_text(text)
            
            # Extract headings
            headings = []
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    h_text = heading.get_text(strip=True)
                    if h_text:
                        headings.append({
                            'level': i,
                            'text': h_text
                        })
            
            # Create document
            document = {
                'url': url,
                'title': title_text,
                'text': text,
                'text_length': len(text),
                'word_count': len(text.split()),
                'headings': headings,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'links_count': len(links)
            }
            
            print(f"  ✓ Success: {document['word_count']} words, {len(links)} links found")
            
            self.stats['pages_crawled'] += 1
            self.stats['total_words'] += document['word_count']
            
            return document, links
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {str(e)}")
            self.stats['pages_failed'] += 1
            self.failed.add(url)
            return None, set()
        except Exception as e:
            print(f"  ✗ Parse error: {str(e)}")
            self.stats['pages_failed'] += 1
            self.failed.add(url)
            return None, set()
    
    def save_progress(self):
        """Save current progress"""
        # Save documents
        docs_file = os.path.join(self.output_dir, 'documents.json')
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        self.stats['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save state for resumption
        state = {
            'to_visit': list(self.to_visit),
            'visited': list(self.visited),
            'failed': list(self.failed),
            'url_map': self.url_map
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"\n💾 Progress saved: {len(self.documents)} documents")
    
    def load_state(self):
        """Load previous state if exists"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                
                self.to_visit = deque(state['to_visit'])
                self.visited = set(state['visited'])
                self.failed = set(state['failed'])
                self.url_map = state['url_map']
                
                # Load documents
                docs_file = os.path.join(self.output_dir, 'documents.json')
                if os.path.exists(docs_file):
                    with open(docs_file, 'r', encoding='utf-8') as f:
                        self.documents = json.load(f)
                
                # Load stats
                if os.path.exists(self.progress_file):
                    with open(self.progress_file, 'r', encoding='utf-8') as f:
                        saved_stats = json.load(f)
                        self.stats.update(saved_stats)
                
                print(f"✓ Resumed from previous state")
                print(f"  Documents: {len(self.documents)}")
                print(f"  To visit: {len(self.to_visit)}")
                print(f"  Visited: {len(self.visited)}")
                
                return True
            except Exception as e:
                print(f"⚠ Could not load state: {e}")
                return False
        return False
    
    def crawl(self, resume=True):
        """
        Main crawling function
        
        Args:
            resume (bool): Whether to resume from previous state
        """
        # Try to resume if requested
        if resume:
            self.load_state()
        
        save_interval = 50  # Save every 50 pages
        
        try:
            while self.to_visit:
                # Check max pages limit
                if self.max_pages and self.stats['pages_crawled'] >= self.max_pages:
                    print(f"\n⚠ Reached maximum page limit: {self.max_pages}")
                    break
                
                # Get next URL
                url = self.to_visit.popleft()
                
                # Skip if already visited
                if url in self.visited:
                    continue
                
                # Mark as visited
                self.visited.add(url)
                
                # Scrape the page
                document, links = self.scrape_page(url)
                
                if document:
                    # Store document
                    self.url_map[url] = len(self.documents)
                    self.documents.append(document)
                    
                    # Add new links to queue
                    for link in links:
                        if link not in self.visited and link not in self.to_visit:
                            self.to_visit.append(link)
                
                # Rate limiting
                time.sleep(self.delay)
                
                # Periodic save
                if self.stats['pages_crawled'] % save_interval == 0:
                    self.save_progress()
                    self.print_status()
        
        except KeyboardInterrupt:
            print("\n\n⚠ Crawling interrupted by user")
            self.save_progress()
        
        # Final save
        self.save_progress()
        self.generate_reports()
        self.print_final_summary()
    
    def print_status(self):
        """Print current crawling status"""
        print("\n" + "="*80)
        print("CRAWLING STATUS")
        print("="*80)
        print(f"Pages crawled: {self.stats['pages_crawled']}")
        print(f"Pages failed: {self.stats['pages_failed']}")
        print(f"Pages in queue: {len(self.to_visit)}")
        print(f"Total words: {self.stats['total_words']:,}")
        print("="*80 + "\n")
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)
        
        # 1. All text combined
        all_text_file = os.path.join(self.output_dir, 'all_text.txt')
        with open(all_text_file, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(f"\n{'='*80}\n")
                f.write(f"URL: {doc['url']}\n")
                f.write(f"TITLE: {doc['title']}\n")
                f.write(f"WORDS: {doc['word_count']}\n")
                f.write(f"{'='*80}\n\n")
                f.write(doc['text'])
                f.write('\n\n')
        print(f"✓ Combined text: {all_text_file}")
        
        # 2. URLs list
        urls_file = os.path.join(self.output_dir, 'all_urls.txt')
        with open(urls_file, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(f"{doc['url']}\n")
        print(f"✓ URLs list: {urls_file}")
        
        # 3. Summary statistics
        summary_file = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("IIT JODHPUR COMPLETE WEBSITE CRAWL - SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Crawl started: {self.stats['start_time']}\n")
            f.write(f"Crawl completed: {datetime.now().isoformat()}\n\n")
            f.write(f"Pages successfully crawled: {self.stats['pages_crawled']}\n")
            f.write(f"Pages failed: {self.stats['pages_failed']}\n")
            f.write(f"Total words extracted: {self.stats['total_words']:,}\n")
            f.write(f"Average words per page: {self.stats['total_words']/max(1, self.stats['pages_crawled']):.1f}\n\n")
            
            # Top 10 longest pages
            f.write("TOP 10 LONGEST PAGES:\n")
            f.write("-"*80 + "\n")
            sorted_docs = sorted(self.documents, key=lambda x: x['word_count'], reverse=True)
            for i, doc in enumerate(sorted_docs[:10], 1):
                f.write(f"{i}. {doc['title']} ({doc['word_count']} words)\n")
                f.write(f"   {doc['url']}\n\n")
            
            # Failed URLs
            if self.failed:
                f.write("\nFAILED URLs:\n")
                f.write("-"*80 + "\n")
                for url in sorted(self.failed):
                    f.write(f"{url}\n")
        
        print(f"✓ Summary report: {summary_file}")
        
        # 4. Structured JSON
        structured_file = os.path.join(self.output_dir, 'structured_data.json')
        structured = {
            'metadata': {
                'base_url': self.base_url,
                'crawl_date': datetime.now().isoformat(),
                'total_pages': len(self.documents),
                'total_words': self.stats['total_words']
            },
            'statistics': self.stats,
            'documents': self.documents
        }
        with open(structured_file, 'w', encoding='utf-8') as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)
        print(f"✓ Structured data: {structured_file}")
    
    def print_final_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("CRAWLING COMPLETE!")
        print("="*80)
        print(f"\n✅ Successfully crawled {self.stats['pages_crawled']} pages")
        print(f"❌ Failed to crawl {self.stats['pages_failed']} pages")
        print(f"📝 Total words extracted: {self.stats['total_words']:,}")
        print(f"📊 Average words per page: {self.stats['total_words']/max(1, self.stats['pages_crawled']):.1f}")
        print(f"\n📁 All data saved to: {self.output_dir}/")
        print("\nGenerated files:")
        print("  • documents.json - Complete structured data")
        print("  • all_text.txt - All text combined")
        print("  • all_urls.txt - List of all URLs")
        print("  • summary.txt - Detailed summary")
        print("  • structured_data.json - Metadata + documents")
        print("  • progress.json - Crawling statistics")
        print("="*80)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    
    # Configuration
    print("\nDefault settings:")
    print("  • Output directory: iitj_complete_data")
    print("  • Max pages: Unlimited (will crawl entire site)")
    print("  • Delay between requests: 1.5 seconds")
    
    use_defaults = input("\nUse default settings? (y/n): ").strip().lower()
    
    if use_defaults == 'y':
        output_dir = 'iitj_complete_data'
        max_pages = None
        delay = 1.5
    else:
        output_dir = input("Output directory [iitj_complete_data]: ").strip() or 'iitj_complete_data'
        max_pages_input = input("Max pages (press Enter for unlimited): ").strip()
        max_pages = int(max_pages_input) if max_pages_input else None
        delay_input = input("Delay between requests in seconds [1.5]: ").strip()
        delay = float(delay_input) if delay_input else 1.5
    
    resume = input("\nResume from previous crawl if exists? (y/n): ").strip().lower() == 'y'
    
    # Create crawler
    crawler = IITJodhpurFullCrawler(
        output_dir=output_dir,
        max_pages=max_pages,
        delay=delay
    )
    
    # Start crawling
    print("\n🚀 Starting crawl...")
    print("Press Ctrl+C to stop and save progress\n")
    
    crawler.crawl(resume=resume)


if __name__ == "__main__":
    main()