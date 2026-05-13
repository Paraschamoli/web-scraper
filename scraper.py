"""
Single URL Web Scraper - Extract company information from any URL

Usage:
    python scraper.py https://example.com
    python scraper.py https://example.com --output json
"""

import asyncio
import sys
import os
import json
import csv
import argparse
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from playwright.async_api import async_playwright, Page
    from openai import OpenAI
    import pdfplumber
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install playwright openai pdfplumber requests")
    print("Then run: playwright install chromium")
    sys.exit(1)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable not set")
    sys.exit(1)

class WebScraper:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
    
    def is_pdf(self, url: str) -> bool:
        """Check if URL points to a PDF."""
        parsed = urlparse(url)
        return parsed.path.lower().endswith('.pdf')
    
    def extract_pdf_text(self, url: str) -> str:
        """Extract text from PDF URL."""
        try:
            # Download PDF
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract text using pdfplumber
            import io
            with io.BytesIO(response.content) as pdf_file:
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    async def crawl_url(self, url: str, follow_links: bool = False, max_pages: int = 50) -> str:
        """Crawl a URL and extract text content (HTML or PDF).
        
        Args:
            url: URL to crawl
            follow_links: If True, extract links to company pages and crawl them too
            max_pages: Maximum number of company pages to crawl (default: 50)
        """
        # Check if it's a PDF
        if self.is_pdf(url):
            print("[PDF detected] Extracting text from PDF...")
            return self.extract_pdf_text(url)
        
        # HTML page - use Playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait for content to load
                await asyncio.sleep(2)
                
                # Extract all text from page
                text = await page.evaluate("""() => {
                    return document.body.innerText;
                }""")
                
                # If follow_links is enabled, extract and crawl company pages
                if follow_links:
                    print("[Link extraction] Finding company page links...")
                    links = await page.evaluate("""() => {
                        const links = [];
                        document.querySelectorAll('a[href]').forEach(a => {
                            const href = a.href;
                            // Filter for likely company profile links
                            if (href && !href.includes('#') && 
                                !href.includes('mailto:') &&
                                !href.includes('tel:') &&
                                (href.includes('exhibitor') || 
                                 href.includes('company') ||
                                 href.includes('profile') ||
                                 href.includes('detail') ||
                                 !href.includes('anuga.com'))) {  // External links likely company pages
                                links.push(href);
                            }
                        });
                        return [...new Set(links)];  // Deduplicate
                    }""")
                    
                    print(f"Found {len(links)} potential company page links")
                    
                    # Crawl up to max_pages company pages
                    for i, link in enumerate(links[:max_pages], 1):
                        try:
                            print(f"[{i}/{min(max_pages, len(links))}] Crawling: {link}")
                            new_page = await browser.new_page()
                            await new_page.goto(link, wait_until="networkidle", timeout=20000)
                            await asyncio.sleep(1)
                            
                            page_text = await new_page.evaluate("""() => {
                                return document.body.innerText;
                            }""")
                            
                            text += f"\n\n--- COMPANY PAGE: {link} ---\n{page_text}"
                            await new_page.close()
                            
                        except Exception as e:
                            print(f"  Error crawling {link}: {e}")
                            continue
                
                return text
            finally:
                await browser.close()
    
    def extract_company_info(self, text: str, url: str) -> list:
        """Extract company information using LLM - returns list of all companies."""
        # Process text in chunks to handle large PDFs
        chunk_size = 12000  # characters per chunk
        all_companies = []
        total_chunks = (len(text) + chunk_size - 1) // chunk_size
        
        print(f"Processing {len(text)} characters in {total_chunks} chunks...")
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = text[start_idx:end_idx]
            
            print(f"[Chunk {i+1}/{total_chunks}] Extracting companies...")
            
            prompt = f"""Extract ALL company information from the following content. This is a database/document containing multiple companies.

URL: {url}

CONTENT:
{chunk}

Extract ALL companies found in this chunk. Return a JSON ARRAY of company objects.
Each company object should have:
{{
  "company_name": "string",
  "website": "string or null",
  "location": "string or null",
  "country": "string or null",
  "contact_person": "string or null",
  "direct_emails": ["email1@company.com"] or null,
  "phone_numbers": ["+91-123-456-7890"] or null,
  "product_category": "string or null",
  "business_description": "string or null",
  "linkedin_profile": "string or null",
  "social_links": ["https://facebook.com/company"] or null,
  "export_region": "string or null",
  "eu_destinations": ["Germany", "France"] or null,
  "export_details": ["Product 1", "Product 2"] or null,
  "certifications": ["ISO 9001", "CE"] or null
}}

Return ONLY a JSON ARRAY of company objects. If no companies found in this chunk, return empty array []."""

            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b:nitro",
                    messages=[
                        {"role": "system", "content": "You are a data extraction specialist. Extract ALL companies and return a JSON ARRAY."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                content = response.choices[0].message.content
                
                # Clean up response
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                result = json.loads(content)
                
                # Ensure result is a list
                if isinstance(result, dict):
                    result = [result]
                elif not isinstance(result, list):
                    result = []
                
                print(f"  Found {len(result)} companies in chunk {i+1}")
                all_companies.extend(result)
                
            except Exception as e:
                print(f"  Error in chunk {i+1}: {e}")
                continue
        
        # Remove duplicates based on company_name
        seen = set()
        unique_companies = []
        for company in all_companies:
            name = company.get("company_name", "").strip().lower()
            if name and name not in seen:
                seen.add(name)
                unique_companies.append(company)
        
        print(f"Total unique companies after deduplication: {len(unique_companies)}")
        return unique_companies
    
    def write_to_csv(self, companies: list, output_file: str = None):
        """Write companies to CSV file."""
        if not companies:
            print("No companies to write to CSV")
            return
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"companies_{timestamp}.csv"
        
        # Flatten nested lists to semicolon-separated strings
        def flatten_value(value):
            if isinstance(value, list):
                return "; ".join(str(v) for v in value)
            return str(value) if value is not None else ""
        
        # Get all unique keys from all companies
        all_keys = set()
        for company in companies:
            all_keys.update(company.keys())
        
        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            for company in companies:
                # Flatten values for CSV
                flattened = {k: flatten_value(v) for k, v in company.items()}
                writer.writerow(flattened)
        
        print(f"\nSaved {len(companies)} companies to: {output_file}")
        return output_file

async def main():
    parser = argparse.ArgumentParser(description="Scrape a URL and extract ALL company information")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--output", "-o", choices=["csv", "json", "table"], default="csv", help="Output format")
    parser.add_argument("--file", "-f", help="Output CSV filename (for csv output)")
    parser.add_argument("--follow-links", action="store_true", help="Follow links to individual company pages (for list pages)")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum number of company pages to crawl (default: 50)")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Scraping: {args.url}")
    print(f"{'='*60}\n")
    
    scraper = WebScraper()
    
    # Crawl the URL
    print("[1/2] Crawling website...")
    text = await scraper.crawl_url(args.url, follow_links=args.follow_links, max_pages=args.max_pages)
    
    if not text or len(text) < 100:
        print("Error: Could not extract meaningful content")
        sys.exit(1)
    
    print(f"Extracted {len(text)} characters\n")
    
    # Extract company info
    print("[2/2] Extracting ALL company information...")
    companies = scraper.extract_company_info(text, args.url)
    
    if not companies:
        print("No companies found")
        sys.exit(1)
    
    print(f"Found {len(companies)} companies\n")
    
    # Display/Save results
    if args.output == "csv":
        csv_file = scraper.write_to_csv(companies, args.file)
    elif args.output == "json":
        print(json.dumps(companies, indent=2, ensure_ascii=False))
    else:
        # Table display
        print("\n" + "="*60)
        print(f"COMPANIES FOUND: {len(companies)}")
        print("="*60)
        for i, company in enumerate(companies, 1):
            print(f"\n--- Company {i} ---")
            for key, value in company.items():
                if value:
                    if isinstance(value, list):
                        value = "; ".join(str(v) for v in value)
                    print(f"{key.replace('_', ' ').title()}: {value}")
        print("\n" + "="*60)

if __name__ == "__main__":
    import os
    asyncio.run(main())
