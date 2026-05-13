#!/usr/bin/env python3
"""
CONTACT ENRICHER – Crawls company websites from leads.csv to find missing emails, phones, LinkedIn, etc.
=============================================================================================
Reads an existing leads.csv, visits each company's website (and contact/about pages),
extracts contact info via HTML + regex + LLM, and writes an enriched CSV.

Usage:
  python enrich_contacts.py --input leads.csv --output leads_enriched.csv
  python enrich_contacts.py --input leads.csv --delay-min 3 --delay-max 6
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
import socket
import requests
from typing import Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv

load_dotenv()

try:
    from playwright.async_api import async_playwright
    from openai import OpenAI
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install playwright openai python-dotenv && playwright install")
    sys.exit(1)

# ============================================================
# CONFIG
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not set. Add it to .env file.")
    sys.exit(1)

EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_REGEX = r"""
(?:(?:\+|00)\d{1,3}[\s\-\.]?)?          # country code
(?:\(?\d{1,5}\)?[\s\-\.]?)?              # area code
\d{2,5}[\s\-\.]?\d{2,5}[\s\-\.]?\d{0,5}  # main number
"""
# Stricter phone: must start with + or have 7+ digits total
STRICT_PHONE_REGEX = r"""
(?:\+\d{1,3}[\s\-\.\?]?)?                  # optional +country
(?:\(?\d{1,5}\)?[\s\-\.\?]?)?              # optional area code
\d{2,5}[\s\-\.\?]\d{2,5}                  # main number parts
(?:[\s\-\.\?]\d{2,5})?                    # optional extension
"""

LINKEDIN_REGEX = r'https?://(?:www\.)?linkedin\.com/(?:company|in)/[A-Za-z0-9_-]+'
SOCIAL_PATTERNS = {
    "facebook": r'https?://(?:www\.)?facebook\.com/(?!p/|sharer/|share\?)[A-Za-z0-9._-]+',
    "twitter": r'https?://(?:www\.)?(?:twitter\.com|x\.com)/(?!status/|i/|home|search|hashtag)[A-Za-z0-9_-]+',
    "instagram": r'https?://(?:www\.)?instagram\.com/(?!p/|reel/|reels/|explore/|stories/)[A-Za-z0-9._]+',
    "youtube": r'https?://(?:www\.)?youtube\.com/(?:channel|c|user)/[A-Za-z0-9_-]+',
}

# Path fragments that are NOT social profile pages
SOCIAL_FALSE_POSITIVES = {
    'instagram.com/p/', 'instagram.com/reel/', 'instagram.com/reels/',
    'instagram.com/explore/', 'instagram.com/stories/',
    'facebook.com/sharer/', 'facebook.com/share',
    'x.com/status/', 'x.com/home', 'x.com/search', 'x.com/hashtag',
    'twitter.com/status/', 'twitter.com/home',
}

CONTACT_PAGE_PATTERNS = [
    "/contact", "/contact-us", "/about", "/about-us", "/impressum",
    "/imprint", "/kontakt", "/reach-us", "/connect", "/en/contact",
    "/en/about", "/company", "/team",
]

BLOCKED_EMAIL_DOMAINS = {
    "sentry.io", "wixpress.com", "w3.org", "example.com", "localhost",
    "github.com", "github.io", "npmjs.com", "jsdelivr.net", "cloudflare.com",
    "googleapis.com", "google.com", "facebook.com", "twitter.com", "linkedin.com",
    "microsoft.com", "apple.com", "amazonaws.com", "wordpress.org",
    "gravatar.com", "pinterest.com", "sharethis.com", "addthis.com",
    "hotjar.com", "hubspot.com", "mailchimp.com", "mcusercontent.com",
}

DEFAULT_MIN_DELAY = 3.0
DEFAULT_MAX_DELAY = 6.0
DEFAULT_BURST_AFTER = 20
DEFAULT_BURST_PAUSE = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 25000  # ms


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_valid_email(email):
    if not email or "@" not in email:
        return False
    email_lower = email.lower().strip()
    if any(bad in email_lower for bad in ["noreply", "users.noreply.github.com", "localhost",
                                           "example.com", "test.com", "sentry.io",
                                           "wixpress.com", "email.com"]):
        return False
    local, domain = email_lower.split("@", 1)
    if "." not in domain or len(local) == 0:
        return False
    # Block known platform/CDN domains
    domain_root = domain.split(".")[-2] + "." + domain.split(".")[-1] if "." in domain else domain
    if domain in BLOCKED_EMAIL_DOMAINS or domain_root in BLOCKED_EMAIL_DOMAINS:
        return False
    return True


def extract_emails_from_text(text):
    emails = re.findall(EMAIL_REGEX, text, re.IGNORECASE)
    valid = set()
    for e in emails:
        e = e.lower().strip().replace("mailto:", "")
        if is_valid_email(e):
            valid.add(e)
    return list(valid)


def extract_emails_from_html(html):
    """Extract emails from mailto: links and data attributes in raw HTML."""
    emails = set()
    # mailto: links
    for match in re.findall(r'href=["\']mailto:([^"\']+)', html, re.IGNORECASE):
        email = match.split("?")[0].lower().strip()
        if is_valid_email(email):
            emails.add(email)
    # data-email or data-mail attributes
    for match in re.findall(r'data-(?:email|mail)=["\']([^"\']+)["\']', html, re.IGNORECASE):
        email = match.lower().strip()
        if is_valid_email(email):
            emails.add(email)
    # Emails in onclick handlers (e.g., location.href='mailto:...')
    for match in re.findall(r"mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", html):
        email = match.lower().strip()
        if is_valid_email(email):
            emails.add(email)
    # Obfuscated emails: [at] / [dot] patterns
    for match in re.findall(r'([A-Za-z0-9._%+-]+)\s*(?:\[at\]|\(@\)|\s+at\s+)\s*([A-Za-z0-9.-]+)\s*(?:\[dot\]|\(\.\)|\s+dot\s+)\s*([A-Za-z.]+)', html):
        email = f"{match[0]}@{match[1]}.{match[2]}".lower().strip()
        if is_valid_email(email):
            emails.add(email)
    return list(emails)


def _is_plausible_phone(phone_str):
    """Filter out dates, year ranges, IDs, and other non-phone digit sequences."""
    stripped = phone_str.strip()
    digits_only = re.sub(r'\D', '', stripped)
    # Too few or too many digits
    if len(digits_only) < 7 or len(digits_only) > 15:
        return False
    # Reject year ranges like (2019-2024) or 1997-2012
    if re.search(r'\(?\d{4}\s*[-–]\s*\d{4}\)?', stripped):
        return False
    # Reject patterns that look like dates: DD.MM.YYYY or DD/MM/YYYY
    if re.match(r'^\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}$', stripped):
        return False
    # Reject standalone 4-digit numbers that look like years
    if re.match(r'^\d{4}$', digits_only) and 1990 <= int(digits_only) <= 2030:
        return False
    # Reject 4-digit-4-digit patterns (year ranges, postal-like)
    if re.match(r'^\d{4}[\-\.]\d{4}$', stripped):
        return False
    # Numbers without country code and <10 digits are often false positives
    # (dates, IDs, zip codes, etc.) — only allow if clearly structured
    if not stripped.startswith('+') and len(digits_only) < 10:
        # Must have parentheses with area code like (030) or (089)
        if not re.search(r'\(\d{2,5}\)', stripped):
            return False
    # Must contain at least one non-digit separator or start with +
    if not stripped.startswith('+') and not re.search(r'[\s\-\.\(\)]', stripped):
        if len(digits_only) > 10:
            return False
    return True


def extract_phones_from_text(text):
    phones = re.findall(STRICT_PHONE_REGEX, text, re.VERBOSE)
    cleaned = set()
    for p in phones:
        p = " ".join(p.split())
        if _is_plausible_phone(p):
            cleaned.add(p)
    return list(cleaned)


def extract_phones_from_html(html):
    """Extract phones from tel: links in raw HTML."""
    phones = set()
    for match in re.findall(r'href=["\']tel:([^"\']+)', html, re.IGNORECASE):
        phone = match.strip()
        if phone and _is_plausible_phone(phone):
            phones.add(phone)
    return list(phones)


def extract_linkedin(text):
    matches = re.findall(LINKEDIN_REGEX, text, re.IGNORECASE)
    return list(set(matches))


def extract_social_links(html, text):
    links = set()
    combined = html + "\n" + text
    for platform, pattern in SOCIAL_PATTERNS.items():
        for match in re.findall(pattern, combined, re.IGNORECASE):
            links.add(match)
    # Also check href attributes in HTML
    for match in re.findall(r'href=["\'](' + "|".join(SOCIAL_PATTERNS.values()) + r')["\']', html, re.IGNORECASE):
        links.add(match)
    # Filter out false positives (path fragments, not profile pages)
    filtered = set()
    for link in links:
        lower = link.lower()
        if not any(fp in lower for fp in SOCIAL_FALSE_POSITIVES):
            # Also filter very short path segments (e.g., instagram.com/p)
            parsed = urlparse(link)
            path = parsed.path.strip("/")
            if path and len(path) < 2:
                continue
            filtered.add(link)
    return list(filtered)


def normalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()

def is_valid_domain(url: str) -> bool:
    """Check if URL has valid domain structure."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Basic domain validation
        if not domain or '.' not in domain:
            return False
        # Exclude file extensions that shouldn't be domains
        if domain.endswith(('.css', '.js', '.png', '.jpg', '.gif', '.pdf')):
            return False
        # Exclude common non-website patterns
        if any(x in domain for x in ['localhost', '127.0.0.1', '0.0.0.0']):
            return False
        return True
    except:
        return False

async def validate_url_accessibility(url: str) -> tuple[bool, str]:
    """Quick check if URL is accessible and returns actual content."""
    try:
        import socket
        parsed = urlparse(url)
        domain = parsed.netloc
        # Quick DNS check
        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            return False, "DNS resolution failed"
        
        # Quick HEAD request to check content type
        import requests
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        
        # Check if it's HTML content (not CSS, JS, etc.)
        if 'text/html' not in content_type:
            return False, f"Non-HTML content: {content_type}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

def find_contact_page_urls(base_url, html):
    """Find links to contact/about pages from the homepage HTML."""
    contact_urls = []
    for pattern in CONTACT_PAGE_PATTERNS:
        # Check in href attributes
        for match in re.findall(r'href=["\']([^"\']*' + re.escape(pattern) + r'[^"\']*)["\']', html, re.IGNORECASE):
            full_url = urljoin(base_url, match)
            contact_urls.append(full_url)
    return list(set(contact_urls))[:3]  # Limit to 3 contact pages per site


# ============================================================
# ENRICHER CLASS
# ============================================================

class ContactEnricher:
    def __init__(self, min_delay=3, max_delay=6, burst_after=20, burst_pause=20,
                 max_retries=3, use_llm=True):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.burst_after = burst_after
        self.burst_pause = burst_pause
        self.max_retries = max_retries
        self.use_llm = use_llm

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        self.burst_counter = 0
        self.stats = {"crawled": 0, "skipped": 0, "errors": 0, "enriched": 0}

    async def random_delay(self):
        delay = random.uniform(self.min_delay, self.max_delay)
        if random.random() < 0.1:
            delay += random.uniform(3, 8)
        await asyncio.sleep(delay)

    async def burst_check(self):
        self.burst_counter += 1
        if self.burst_counter >= self.burst_after:
            pause = random.uniform(self.burst_pause, self.burst_pause + 10)
            print(f"  💤 Burst pause: {pause:.1f}s...")
            await asyncio.sleep(pause)
            self.burst_counter = 0

    def llm_extract_contacts(self, url, text):
        """Use LLM to extract contact info from page text."""
        if not self.use_llm:
            return {}
        short_text = text[:6000]
        prompt = f"""Extract contact information from this company website page. Return ONLY valid JSON.

URL: {url}

Content:
{short_text}

JSON fields:
{{
  "contact_person": "",
  "direct_emails": [],
  "phone_numbers": [],
  "linkedin_profile": "",
  "social_links": []
}}

Only include information you are confident about. Leave fields empty if not found."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b:nitro",
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON. Extract only factual contact information."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=30
                )
                content = response.choices[0].message.content.strip()
                # Strip markdown fences
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed[0] if parsed else {}
                return parsed
            except Exception as e:
                print(f"    LLM attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
                else:
                    return {}
        return {}

    async def crawl_page(self, page, url):
        """Crawl a single page and return (text, html)."""
        # Add overall timeout to prevent hanging
        try:
            # Use asyncio.wait_for to prevent indefinite hanging
            result = await asyncio.wait_for(self._crawl_page_internal(page, url), timeout=120)
            return result
        except asyncio.TimeoutError:
            print(f"    ⏱️  Page crawl timeout for {url} (120s exceeded)")
            return None, None
        except Exception as e:
            error_msg = str(e).lower()
            if "execution context was destroyed" in error_msg or "navigation" in error_msg:
                print(f"    ⚠️  Page navigation error for {url}: {str(e)[:100]}")
                return None, None
            elif "timeout" in error_msg or "timed out" in error_msg:
                print(f"    ⏱️  Timeout for {url} - will retry with longer timeout")
                try:
                    # Retry with networkidle and longer timeout
                    await page.goto(url, wait_until="networkidle", timeout=90000)
                    await asyncio.sleep(5)
                    text = await page.evaluate("() => document.body.innerText")
                    html = await page.evaluate("() => document.documentElement.outerHTML")
                    if text and len(text) > 100:
                        print(f"    ✅ Retry successful for {url}")
                        return text, html
                except Exception as retry_exc:
                    print(f"    ❌ Retry failed for {url}: {retry_exc}")
            else:
                print(f"    ⚠️  Error loading {url}: {e}")
            return None, None

    async def _crawl_page_internal(self, page, url):
        """Internal crawl logic with detailed error handling."""
        try:
            # Enhanced timeout and retry strategy for connection issues
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)  # Longer wait for dynamic content
            
            # Check if page actually loaded - with context error handling
            try:
                page_title = await page.title()
                if not page_title or "error" in page_title.lower() or "not found" in page_title.lower():
                    print(f"    ⚠️  Page error detected for {url}: {page_title}")
                    return None, None
            except Exception as title_error:
                if "execution context" in str(title_error).lower():
                    print(f"    ⚠️  Context destroyed getting title for {url}")
                    return None, None
                raise
            
            # Try to scroll and wait for content
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2)
                await page.evaluate("window.scrollTo(0, 0)")
            except:
                pass  # Ignore scroll errors
            
            # Get content with fallback and redirect handling
            try:
                # Check if we were redirected to a different URL (like CSS files)
                current_url = page.url
                if current_url != url:
                    # Check if redirected to non-HTML content
                    if any(current_url.lower().endswith(ext) for ext in ['.css', '.js', '.png', '.jpg', '.gif', '.pdf']):
                        print(f"    ⚠️  Redirected to non-HTML content: {current_url}")
                        return None, None
                
                text = await page.evaluate("() => document.body.innerText")
                html = await page.evaluate("() => document.documentElement.outerHTML")
            except Exception as eval_error:
                # Handle context destruction during evaluation
                if "execution context" in str(eval_error).lower():
                    print(f"    ⚠️  Context destroyed during evaluation for {url}")
                    # Try fallback extraction
                    try:
                        text = await page.inner_text("body")
                        html = await page.content()
                    except:
                        text = ""
                        html = ""
                else:
                    # Fallback to simpler extraction for other errors
                    try:
                        text = await page.inner_text("body")
                        html = await page.content()
                    except:
                        text = ""
                        html = ""
            
            # Validate we got meaningful content
            if not text or len(text) < 100:
                print(f"    ⚠️  Minimal content from {url}: {len(text) if text else 0} chars")
                return text, html
                
            return text, html
        except Exception as e:
            # Re-raise to be handled by outer try-catch
            raise

    async def find_website_with_llm(self, company_name):
        """Use LLM to find the official website for a company."""
        # Clean company name for better search
        clean_name = company_name.strip()
        # Remove common suffixes that might confuse search
        clean_name = re.sub(r'\s+(S\.A\.|S\.L\.|S\.p\.A\.|GmbH|Ltd|LLC|Pte|Ltd|Co|Corp|Inc)\.?$', '', clean_name, flags=re.IGNORECASE)
        
        prompt = f"""You are a web search expert. Find the official website for the company "{clean_name}".

Search strategy:
1. Look for the company's official .com website
2. If the company has a country-specific domain (e.g., .fr, .it, .de), include that
3. Avoid social media pages (facebook, instagram, linkedin)
4. Avoid directory/listing pages
5. Return the main homepage URL, not a subpage

Return ONLY the website URL as plain text. For example:
https://www.company.com
https://company.com

If you cannot find a confident official website, return the word "NONE"."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b:nitro",
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a web search expert that returns only official company website URLs or 'NONE'. Never include explanations or extra text."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=30
                )
                content = response.choices[0].message.content.strip()
                # Clean up any markdown or extra text
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\s*', '', content)
                    content = re.sub(r'\s*```$', '', content)
                content = content.strip()
                
                # Check if it's a valid URL
                if content and content.startswith("http") and not content.startswith("NONE"):
                    # Ensure it has a proper domain
                    if '.' in content.split('//')[-1]:
                        print(f"    🔍 LLM found website: {content}")
                        return content
                else:
                    return None
            except Exception as e:
                print(f"    LLM website search attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
        return None

    async def enrich_company(self, browser, row):
        """Crawl a company's website and extract contact info."""
        try:
            # Add overall timeout to prevent hanging
            result = await asyncio.wait_for(self._enrich_company_internal(browser, row), timeout=180)
            return result
        except asyncio.TimeoutError:
            print(f"    ⏱️  Enrichment timeout for {row.get('company_name', 'Unknown')} (180s exceeded)")
            self.stats["errors"] += 1
            return row
        except Exception as e:
            print(f"    ❌ Enrichment error for {row.get('company_name', 'Unknown')}: {e}")
            self.stats["errors"] += 1
            return row

    async def _enrich_company_internal(self, browser, row):
        """Internal enrichment logic with page recreation on context errors."""
        website = row.get("website", "").strip()
        company_name = row.get("company_name", "").strip()

        # If no website and LLM is enabled, try to find one
        if not website and self.use_llm and company_name:
            print(f"    🔍 Searching for website: {company_name}")
            website = await self.find_website_with_llm(company_name)
            if website:
                row["website"] = website

        if not website:
            self.stats["skipped"] += 1
            return row

        url = normalize_url(website)
        if not url:
            self.stats["skipped"] += 1
            return row

        # Validate URL before crawling
        if not is_valid_domain(url):
            print(f"    ❌ Invalid URL detected: {url}")
            self.stats["skipped"] += 1
            return row

        # Quick accessibility check
        is_accessible, reason = await validate_url_accessibility(url)
        if not is_accessible:
            print(f"    ❌ URL not accessible: {url} - {reason}")
            self.stats["skipped"] += 1
            return row

        print(f"  🌐 Crawling: {url}")

        all_emails = set()
        all_phones = set()
        all_linkedin = set()
        all_social = set()
        contact_person = ""
        crawled_pages = []

        page = await browser.new_page()
        page_needs_close = True
        try:
            # Phase 1: Crawl homepage
            text, html = await self.crawl_page(page, url)
            if not text:
                self.stats["errors"] += 1
                return row

            crawled_pages.append(url)

            # Extract from homepage
            all_emails.update(extract_emails_from_text(text))
            all_emails.update(extract_emails_from_html(html))
            all_phones.update(extract_phones_from_text(text))
            all_phones.update(extract_phones_from_html(html))
            all_linkedin.update(extract_linkedin(text + "\n" + html))
            all_social.update(extract_social_links(html, text))

            # Phase 2: Find and crawl contact/about pages
            contact_urls = find_contact_page_urls(url, html)
            for contact_url in contact_urls:
                await self.random_delay()
                try:
                    c_text, c_html = await self.crawl_page(page, contact_url)
                    if c_text:
                        crawled_pages.append(contact_url)
                        all_emails.update(extract_emails_from_text(c_text))
                        all_emails.update(extract_emails_from_html(c_html))
                        all_phones.update(extract_phones_from_text(c_text))
                        all_phones.update(extract_phones_from_html(c_html))
                        all_linkedin.update(extract_linkedin(c_text + "\n" + c_html))
                        all_social.update(extract_social_links(c_html, c_text))
                except Exception as contact_error:
                    error_msg = str(contact_error).lower()
                    if "execution context" in error_msg or "navigation" in error_msg:
                        print(f"    ⚠️  Context error on contact page, recreating page")
                        # Close current page and create new one
                        try:
                            await asyncio.wait_for(page.close(), timeout=10)
                        except:
                            pass
                        page = await browser.new_page()
                        page_needs_close = True
                    else:
                        print(f"    ⚠️  Error crawling contact page {contact_url}: {contact_error}")

            # Phase 3: If still no emails, try LLM on combined text
            if not all_emails and self.use_llm and text:
                combined_text = text[:6000]
                llm_data = await self.extract_with_llm(combined_text, company_name)
                if llm_data:
                    for e in llm_data.get("emails", []):
                        all_emails.add(e)
                    for p in llm_data.get("phones", []):
                        all_phones.add(p)
                    for l in llm_data.get("linkedin", []):
                        all_linkedin.add(l)
                    for s in llm_data.get("social", []):
                        all_social.add(s)
                    if llm_data.get("contact_person") and not contact_person:
                        contact_person = llm_data["contact_person"]

        finally:
            if page_needs_close:
                try:
                    await asyncio.wait_for(page.close(), timeout=10)
                except:
                    pass

        # Update the row with enriched data (only fill empty fields)
        enriched = False

        if all_emails and not row.get("direct_emails", "").strip():
            row["direct_emails"] = "; ".join(sorted(all_emails))
            enriched = True

        if all_phones and not row.get("phone_numbers", "").strip():
            row["phone_numbers"] = "; ".join(sorted(all_phones))
            enriched = True

        if all_linkedin and not row.get("linkedin_profile", "").strip():
            row["linkedin_profile"] = "; ".join(sorted(all_linkedin))
            enriched = True

        if all_social and not row.get("social_links", "").strip():
            row["social_links"] = "; ".join(sorted(all_social))
            enriched = True

        if contact_person and not row.get("contact_person", "").strip():
            row["contact_person"] = contact_person
            enriched = True

        if enriched:
            self.stats["enriched"] += 1
            print(f"  ✅ Enriched: {row.get('company_name', 'Unknown')} | emails={len(all_emails)} phones={len(all_phones)} linkedin={len(all_linkedin)}")
        else:
            print(f"  ⏭️ No new contacts found for {row.get('company_name', 'Unknown')}")

        self.stats["crawled"] += 1
        return row


# ============================================================
# MAIN
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Enrich leads.csv with contact info from company websites")
    parser.add_argument("--input", "-i", default="leads.csv", help="Input CSV file")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file (default: <input>_enriched.csv)")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_MIN_DELAY, help="Min seconds between page loads")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_MAX_DELAY, help="Max seconds between page loads")
    parser.add_argument("--burst-after", type=int, default=DEFAULT_BURST_AFTER, help="Pause after N sites")
    parser.add_argument("--burst-pause", type=int, default=DEFAULT_BURST_PAUSE, help="Burst pause duration (seconds)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback extraction")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of companies to process (0=all)")
    parser.add_argument("--skip-has-emails", action="store_true", help="Skip companies that already have emails")
    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_enriched{ext}"

    # Read input CSV
    print(f"\n📖 Reading: {args.input}")
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(dict(row))

    print(f"   Found {len(rows)} companies")

    # Filter rows
    targets = []
    for row in rows:
        if args.skip_has_emails and row.get("direct_emails", "").strip():
            continue
        website = row.get("website", "").strip()
        # Include companies with websites OR companies without websites when LLM is enabled
        if website and website not in ("", "N/A", "-"):
            targets.append(row)
        elif not args.no_llm and row.get("company_name", "").strip():
            # Include companies without websites if LLM is enabled to discover them
            targets.append(row)

    if args.limit > 0:
        targets = targets[:args.limit]

    print(f"   Companies to process: {len(targets)} (includes LLM website discovery for companies without websites)")
    print(f"   Delays: {args.delay_min}–{args.delay_max}s, burst after {args.burst_after} sites ({args.burst_pause}s)")
    print(f"   LLM fallback: {'disabled' if args.no_llm else 'enabled'}\n")

    enricher = ContactEnricher(
        min_delay=args.delay_min,
        max_delay=args.delay_max,
        burst_after=args.burst_after,
        burst_pause=args.burst_pause,
        use_llm=not args.no_llm,
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        for i, row in enumerate(targets, 1):
            company_name = row.get("company_name", "Unknown")
            print(f"\n[{i}/{len(targets)}] {company_name}")

            try:
                row = await enricher.enrich_company(browser, row)
            except Exception as e:
                print(f"  ❌ Fatal error: {e}")
                enricher.stats["errors"] += 1

            await enricher.burst_check()
            await enricher.random_delay()

        await browser.close()

    # Write output CSV
    # Merge fieldnames (in case enricher added new fields)
    all_keys = set(fieldnames) if fieldnames else set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys = sorted(all_keys)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            # Fill missing keys
            for key in all_keys:
                if key not in row:
                    row[key] = ""
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"🎉 Enrichment complete!")
    print(f"   Crawled: {enricher.stats['crawled']}")
    print(f"   Enriched: {enricher.stats['enriched']}")
    print(f"   Skipped: {enricher.stats['skipped']}")
    print(f"   Errors: {enricher.stats['errors']}")
    print(f"📁 Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())