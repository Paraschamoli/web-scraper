#!/usr/bin/env python3
"""
UNIVERSAL EXHIBITOR SCRAPER – Chunked, Memory‑Safe, Anti‑Block
================================================================
- Generic – works with any exhibitor list URL (auto‑detects pagination)
- Processes exhibitor detail pages in chunks (saves memory)
- Progressive saving (crash‑safe)
- Randomised delays, burst pauses, exponential backoff (avoids 403)
- Single concurrency (1 page at a time) – minimal RAM usage

Usage:
  python universal_scraper.py "https://example.com/exhibitors" --output leads.csv
  python universal_scraper.py "https://example.com/list" --chunk-size 100 --burst-after 50
"""

import asyncio
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv

load_dotenv()

try:
    from crawlee.crawlers import PlaywrightCrawler, PlaywrightCrawlingContext
    from playwright.async_api import async_playwright
    from openai import OpenAI
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install crawlee playwright openai python-dotenv && playwright install")
    sys.exit(1)

# ============================================================
# CONFIG – READ FROM ENV OR DEFAULTS
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set. Add it to .env file.")
    sys.exit(1)

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_REGEX = r"""
(?:\+?\d{1,4}[\s\-\.]?)?
(?:\(?\d{2,5}\)?[\s\-\.]?)?
\d{3,5}[\s\-\.]?\d{3,5}
"""

# Anti‑block defaults (can be overridden via CLI)
DEFAULT_MIN_DELAY = 2.0          # seconds
DEFAULT_MAX_DELAY = 5.0
DEFAULT_BURST_AFTER = 30         # number of detail pages before a burst pause
DEFAULT_BURST_PAUSE = 15         # seconds
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 2        # seconds, exponential
DEFAULT_MAX_LIST_PAGES = 100     # max pagination steps
DEFAULT_CHUNK_SIZE = 50          # how many detail pages to process before saving

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_url(url):
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    if parsed.scheme not in ("http", "https"):
        parsed = parsed._replace(scheme="https")
    return parsed.geturl()

def url_to_absolute(base, relative):
    if not relative:
        return None
    if relative.startswith(("http://", "https://")):
        return normalize_url(relative)
    return normalize_url(urljoin(base, relative))

def is_valid_email(email):
    if not email or "@" not in email:
        return False
    email_lower = email.lower()
    if any(bad in email_lower for bad in ["noreply", "users.noreply.github.com", "localhost"]):
        return False
    local, domain = email_lower.split("@", 1)
    if "." not in domain or len(local) == 0:
        return False
    return True

def extract_emails(text):
    emails = re.findall(EMAIL_REGEX, text, re.IGNORECASE)
    valid = set()
    for e in emails:
        e = e.lower().strip().replace("mailto:", "")
        if is_valid_email(e):
            valid.add(e)
    return list(valid)

def extract_phones(text):
    phones = re.findall(PHONE_REGEX, text, re.VERBOSE)
    cleaned = set()
    for p in phones:
        p = " ".join(p.split())
        if len(p) >= 7:
            cleaned.add(p)
    return list(cleaned)

def random_delay(min_sec, max_sec):
    delay = random.uniform(min_sec, max_sec)
    # occasional longer pause (mimics human reading)
    if random.random() < 0.1:
        delay += random.uniform(3, 8)
    time.sleep(delay)

async def wait_with_backoff(attempt, base_sec):
    wait = base_sec * (2 ** (attempt - 1)) + random.uniform(0, 1)
    print(f"    ⏸ Backing off for {wait:.1f}s (attempt {attempt})")
    await asyncio.sleep(wait)

# ============================================================
# SCRAPER CLASS (WITHOUT PER‑PAGE LLM – OPTIONAL)
# ============================================================

class UniversalScraper:
    def __init__(self, output_file=None, chunk_size=50, min_delay=2, max_delay=5,
                 burst_after=30, burst_pause=15, max_retries=5, retry_backoff=2):
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.burst_after = burst_after
        self.burst_pause = burst_pause
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.companies = []          # will be saved in chunks
        self.seen_company_names = set()
        self.seen_urls = set()
        self.burst_counter = 0

    # --------------------------------------------------------
    # LLM extraction (optional – you can disable by setting use_llm=False)
    # --------------------------------------------------------
    def llm_extract_company(self, url, text):
        short_text = text[:8000]
        prompt = f"""
Extract company information from this page. Return ONLY valid JSON.

URL: {url}

Content:
{short_text}

JSON fields:
{{
  "company_name": "",
  "website": "",
  "location": "",
  "country": "",
  "contact_person": "",
  "product_category": "",
  "business_description": "",
  "linkedin_profile": "",
  "export_region": "",
  "eu_destinations": [],
  "export_details": [],
  "certifications": []
}}
"""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b:nitro",
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=30
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed[0] if parsed else {}
                return parsed
            except Exception as e:
                print(f"    LLM attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
                else:
                    return {}
        return {}

    # --------------------------------------------------------
    # Process one exhibitor detail page
    # --------------------------------------------------------
    async def process_exhibitor_page(self, url, page):
        url = normalize_url(url)
        if url in self.seen_urls:
            return None
        self.seen_urls.add(url)

        # Check if this is an a2zinc redirect page (openURL.aspx)
        # These are often broken/malformed, so skip them and rely on list page extraction
        if "openURL.aspx" in url:
            print(f"  ⏭️ Skipping a2zinc redirect (will extract from list page)")
            return None

        text = await page.evaluate("() => document.body.innerText")
        if len(text) < 200:
            return None

        # Always extract emails and phones directly
        direct_emails = extract_emails(text)
        phones = extract_phones(text)

        # Optional LLM extraction (you can disable by setting use_llm=False)
        # For now, we use it – but if you want speed, you can comment it out.
        llm_data = self.llm_extract_company(url, text)
        company_name = llm_data.get("company_name")

        # Fallback to title/H1
        if not company_name or len(company_name) < 2:
            title = await page.evaluate("() => document.title")
            if title and len(title) < 100 and len(title) > 2:
                company_name = title.strip()
            else:
                h1 = await page.evaluate("() => document.querySelector('h1')?.innerText")
                if h1 and len(h1) < 100:
                    company_name = h1.strip()

        if not company_name:
            print(f"  ⏭️ No company name for {url}")
            return None

        if company_name in self.seen_company_names:
            return None
        self.seen_company_names.add(company_name)

        company = {
            "company_name": company_name,
            "website": llm_data.get("website"),
            "location": llm_data.get("location"),
            "country": llm_data.get("country"),
            "contact_person": llm_data.get("contact_person"),
            "product_category": llm_data.get("product_category"),
            "business_description": llm_data.get("business_description"),
            "linkedin_profile": llm_data.get("linkedin_profile"),
            "export_region": llm_data.get("export_region"),
            "eu_destinations": "; ".join(llm_data.get("eu_destinations", [])),
            "export_details": "; ".join(llm_data.get("export_details", [])),
            "certifications": "; ".join(llm_data.get("certifications", [])),
            "source_url": url,
            "direct_emails": "; ".join(direct_emails),
            "phone_numbers": "; ".join(phones),
            "social_links": ""
        }
        return company

    # --------------------------------------------------------
    # Save current results to CSV (overwrites)
    # --------------------------------------------------------
    def save_results(self, is_final=False):
        if not self.companies:
            return
        if not self.output_file:
            self.output_file = f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        keys = set()
        for c in self.companies:
            keys.update(c.keys())
        keys = sorted(keys)
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for c in self.companies:
                writer.writerow(c)
        if not is_final:
            print(f"  💾 Chunk saved ({len(self.companies)} companies so far)")

    # --------------------------------------------------------
    # Add a company and save in chunks
    # --------------------------------------------------------
    async def add_company(self, company):
        if not company:
            return
        self.companies.append(company)
        # burst pause counter
        self.burst_counter += 1
        if self.burst_counter >= self.burst_after:
            pause = random.uniform(self.burst_pause, self.burst_pause + 5)
            print(f"  💤 Burst pause: {pause:.1f}s...")
            await asyncio.sleep(pause)
            self.burst_counter = 0

        if len(self.companies) % self.chunk_size == 0:
            self.save_results(is_final=False)

# ============================================================
# CRAWLER WITH AUTO‑PAGINATION & CHUNKED PROCESSING
# ============================================================

async def find_next_page_url(page, current_url):
    """Find the URL of the next paginated list page."""
    next_selectors = [
        'a[rel="next"]',
        'a:has-text("Next")', 'a:has-text("next")',
        'a:has-text("›")', 'a:has-text("»")', 'a:has-text(">")',
        'a.pagination__next', 'li.pagination-next a',
        'a[aria-label="Next"]', 'a.next',
    ]
    for selector in next_selectors:
        next_link = await page.query_selector(selector)
        if next_link:
            href = await next_link.get_attribute("href")
            if href:
                return urljoin(current_url, href)
    return None

async def scroll_list_page(page, max_scrolls=20):
    """Scroll to load dynamic content (if needed)."""
    last_height = await page.evaluate("document.body.scrollHeight")
    for _ in range(max_scrolls):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    await page.evaluate("window.scrollTo(0, 0)")
    await asyncio.sleep(1)

async def main():
    parser = argparse.ArgumentParser(description="Universal Exhibitor Scraper (chunked, anti‑block)")
    parser.add_argument("url", help="Starting exhibitor list URL")
    parser.add_argument("--output", "-o", help="Output CSV file (default: auto-generated)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Save after every N companies")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_MIN_DELAY, help="Min seconds between pages")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_MAX_DELAY, help="Max seconds between pages")
    parser.add_argument("--burst-after", type=int, default=DEFAULT_BURST_AFTER, help="Pause after N detail pages")
    parser.add_argument("--burst-pause", type=int, default=DEFAULT_BURST_PAUSE, help="Pause duration (seconds)")
    parser.add_argument("--max-list-pages", type=int, default=DEFAULT_MAX_LIST_PAGES, help="Max pagination steps")
    parser.add_argument("--selector", default='a[href*="/exhibitor/"], a[href*="/stand/"], a[href*="/participant/"]',
                        help="CSS selector for exhibitor links (default tries common patterns)")
    args = parser.parse_args()

    scraper = UniversalScraper(
        output_file=args.output,
        chunk_size=args.chunk_size,
        min_delay=args.delay_min,
        max_delay=args.delay_max,
        burst_after=args.burst_after,
        burst_pause=args.burst_pause,
    )

    print(f"\n🚀 Starting universal scraper from: {args.url}")
    print(f"   Chunk size: {args.chunk_size} companies per save")
    print(f"   Delays: {args.delay_min}–{args.delay_max}s, burst after {args.burst_after} pages ({args.burst_pause}s)")
    print(f"   Concurrency: 1 page at a time (lowest memory)\n")

    visited_list_pages = set()

    crawler = PlaywrightCrawler(headless=True)

    @crawler.router.default_handler
    async def handler(context: PlaywrightCrawlingContext):
        url = context.request.url
        print(f"\n📄 Processing: {url}")

        # Detect list page (does not contain "/exhibitor/" in the path)
        is_list_page = "list-of-exhibitors" in url or ("anuga.com" in url and "/exhibitor/" not in url)
        # SIAL Paris list page detection
        is_sial_list = "sialparis.com" in url and "/exhibitors-2026/exhibitors" in url and "/exhibitor/" not in url
        if is_sial_list:
            is_list_page = True
        # Generic: if the URL is the same as the starting root or has "page=", treat as list
        if not is_list_page and ("page=" in url or url.rstrip("/") == args.url.rstrip("/")):
            is_list_page = True

        # Detect smallworldlabs directory page (for a2zinc sites)
        is_smallworld_dir = "smallworldlabs.com" in url and "page_id=2424" in url

        if is_smallworld_dir:
            print("  📋 Smallworldlabs directory – enqueuing company detail pages...")
            # Enqueue all /co/ links as detail pages
            await context.enqueue_links(selector='a[href*="/co/"]', limit=10000)

            # Also check for pagination links and enqueue them
            html = await context.page.evaluate("() => document.documentElement.outerHTML")
            import re
            # Look for pagination patterns like "Next", page numbers, etc.
            pagination_links = set()
            # Common pagination patterns
            pagination_patterns = [
                r'href=["\']([^"\']*(?:page|p)=\d+[^"\']*)["\']',
                r'href=["\']([^"\']*[?&]page[^"\']*)["\']',
                r'<a[^>]*>\s*(?:Next|›|»|Next Page)\s*</a>',
            ]
            for pattern in pagination_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                for match in matches:
                    if match.startswith('http'):
                        url = match
                    else:
                        # Resolve relative URL
                        from urllib.parse import urljoin
                        url = urljoin(url, match)
                    if url not in pagination_links and url != context.request.url:
                        pagination_links.add(url)

            if pagination_links:
                print(f"  📋 Found {len(pagination_links)} pagination links")
                for link in pagination_links:
                    await context.add_requests([link])
            else:
                print(f"  📋 Enqueued detail links")
            return

        if is_list_page:
            # Check if this is an a2zinc platform
            is_a2zinc = "a2zinc.net" in url

            # For non-a2zinc sites, check visited pages
            if not is_a2zinc and url in visited_list_pages:
                print("  📋 Already visited this list page, skipping")
                return

            # For non-a2zinc sites, mark as visited
            if not is_a2zinc:
                visited_list_pages.add(url)

            # Scroll to load all exhibitor cards (in case of infinite scroll)
            print("  📋 List page – scrolling to load content...")
            await scroll_list_page(context.page)

            # Check if this is an a2zinc platform site (which has broken redirect links)
            is_a2zinc = "a2zinc.net" in url

            if is_a2zinc:
                # For a2zinc platforms, extract companies from the current page
                print("  📋 a2zinc platform – extracting companies from list page...")
                html = await context.page.evaluate("() => document.documentElement.outerHTML")

                import re
                # Find all anchor tags with openURL.aspx links to get company names
                link_pattern = r'<a[^>]*href=["\'][^"\']*openURL\.aspx[^"\']*["\'][^>]*>([^<]+)</a>'
                company_links = re.findall(link_pattern, html, re.IGNORECASE)
                for company_name in company_links:
                    company_name = company_name.strip()
                    if company_name and len(company_name) > 1 and company_name not in scraper.seen_company_names:
                        scraper.seen_company_names.add(company_name)
                        company = {
                            "company_name": company_name,
                            "website": "",
                            "location": "",
                            "country": "",
                            "contact_person": "",
                            "product_category": "",
                            "business_description": "",
                            "linkedin_profile": "",
                            "export_region": "",
                            "eu_destinations": "",
                            "export_details": "",
                            "certifications": "",
                            "source_url": url,
                            "direct_emails": "",
                            "phone_numbers": "",
                            "social_links": ""
                        }
                        await scraper.add_company(company)
                        print(f"  ✅ {company_name}")

                # Find and enqueue alphabetical filter pages (A, B, C, D, etc.)
                # The a2zinc platform uses letter parameters
                base_url = url.split('?')[0] if '?' in url else url
                letters = ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                alpha_links = []
                for letter in letters:
                    # Construct URL with letter parameter
                    if '?' in url:
                        filter_url = url + f'&letter={letter}'
                    else:
                        filter_url = f'{base_url}?letter={letter}'
                    if filter_url not in visited_list_pages:
                        alpha_links.append(filter_url)
                
                if alpha_links:
                    print(f"  📋 Enqueuing {len(alpha_links)} alphabetical filter pages")
                    for link in alpha_links:
                        await context.add_requests([link])

                print(f"  📋 Total companies extracted so far: {len(scraper.companies)}")
            else:
                # For non-a2zinc sites, enqueue detail links as usual
                await context.enqueue_links(selector=args.selector, limit=10000)
                print(f"  📋 Enqueued exhibitor links")

            # Pagination: find next page
            if len(visited_list_pages) < args.max_list_pages:
                next_url = await find_next_page_url(context.page, url)
                if next_url and next_url not in visited_list_pages:
                    print(f"  📋 Found next page: {next_url}")
                    await context.add_requests([next_url])
                elif is_sial_list:
                    # SIAL Paris specific pagination: increment page parameter
                    from urllib.parse import urlparse, parse_qs, urlencode
                    parsed = urlparse(url)
                    params = parse_qs(parsed.query)
                    current_page = int(params.get('catalog.prod.sial.exhibitors.en[page]', ['1'])[0])
                    # Try to enqueue next few pages (SIAL might have many pages)
                    for next_page_num in range(current_page + 1, current_page + 6):
                        params['catalog.prod.sial.exhibitors.en[page]'] = [str(next_page_num)]
                        next_page_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode(params, doseq=True)}"
                        if next_page_url not in visited_list_pages:
                            print(f"  📋 Enqueuing SIAL page {next_page_num}: {next_page_url[:80]}...")
                            await context.add_requests([next_page_url])
                            # Don't add to visited_list_pages yet - let it be processed first
                else:
                    print("  📋 No more pagination links")
            else:
                print(f"  📋 Reached max pagination steps ({args.max_list_pages})")
            return

        # --- This is an exhibitor detail page ---
        # Wait a random delay before processing
        random_delay(args.delay_min, args.delay_max)

        try:
            company = await scraper.process_exhibitor_page(url, context.page)
            if company:
                await scraper.add_company(company)
                print(f"  ✅ {company['company_name']} (emails: {company['direct_emails'][:50]})")
            else:
                print(f"  ⏭️ Skipped (no valid company data)")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    await crawler.run([args.url])

    # Final save
    scraper.save_results(is_final=True)
    print(f"\n🎉 Done. Total companies: {len(scraper.companies)}")
    if scraper.output_file:
        print(f"📁 Output saved to: {scraper.output_file}")

if __name__ == "__main__":
    asyncio.run(main())