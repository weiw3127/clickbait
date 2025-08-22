import csv 
import time
import logging
from dataclasses import dataclass 
from typing import Iterable, List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

@dataclass
class ScrapeConfig:
    base_url: str = "https://www.dailymail.co.uk"
    delay_secs: float=1.0
    timeout: float = 15.0
    max_articles: Optional[int] = None
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None: 
            self.headers = DEFAULT_HEADERS

class DailyMailScraper:
    '''
    Simple DailyMail scraper: 
    - Reach providing URLS
    - extract article URLs; fetch each artile and parse title/headline + article text 
    - write rows {headlines, article, url} into CSV 
    '''
    def __init__(self, config: ScrapeConfig = ScrapeConfig()): 
        self.cfg = config
        self.session = requests.Session()
        self.session.headers.update(self.cfg.headers)
        self.log = logging.getLogger(self.__class__.__name__)
    
    def scrape_to_csv(self, start_urls: Iterable[str], out_csv_path: str="data/raw/dailymail.csv", append: bool=False):
        urls = self._collect_article_urls(start_urls)
        if self.cfg.max_articles:
            urls = urls[: self.cfg.max_articles]
        
        mode = "a" if append else "w"
        with open(out_csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["headline", "article", "url"])
            if not append: 
                writer.writeheader()

            for i, url in enumerate(urls, 1):
                try:
                    row = self._scrape_article(url)
                    if row:
                        writer.writerow(row)
                        self.log.info("Saved [%d/%d]: %s", i, len(urls), url)
                except Exception as e:
                    self.log.warning("Failed article %s: %s", url, e)
                time.sleep(self.cfg.delay_secs)

    def _collect_article_urls(self, start_urls: Iterable[str]) -> List[str]:
        """
        From listing/section pages collect article <a> links.
        This uses multiple selectors because DailyMail markup varies.
        """
        seen = set()
        results: List[str] = []

        for page_url in start_urls:
            html = self._get(page_url)
            if html is None:
                continue
            soup = BeautifulSoup(html, "lxml")

            # Common patterns for article anchors
            selectors = [
                "a.article a",         # nested anchors
                "a.linkro-darkred",    # headline link
                "a.js-headline-text",  # sometimes present
                "a.gotham-bold",       # alt style
                "h2 a",                # fallback
            ]

            for sel in selectors:
                for a in soup.select(sel):
                    href = (a.get("href") or "").strip()
                    if not href:
                        continue
                    url = urljoin(self.cfg.base_url, href)
                    if self._is_article_url(url) and url not in seen:
                        seen.add(url)
                        results.append(url)

        self.log.info("Collected %d candidate article URLs.", len(results))
        return results

        def _is_article_url(self, url: str) -> bool:
        """
        Heuristic: DailyMail articles usually contain '/news/' or '/tvshowbiz/' etc,
        and end with .html. We also ensure same domain.
        """
        parsed = urlparse(url)
        if parsed.netloc and "dailymail.co.uk" not in parsed.netloc:
            return False
        return url.endswith(".html")

    def _scrape_article(self, url: str) -> Optional[Dict[str, str]]:
        html = self._get(url)
        if html is None:
            return None
        soup = BeautifulSoup(html, "lxml")

        # Title/headline
        title_candidates = [
            ("h2", {"class": "linkro-darkred"}),
            ("h2", {"itemprop": "headline"}),
            ("h1", {"itemprop": "headline"}),
            ("h1", {}),
            ("title", {}),
        ]
        headline = None
        for tag, attrs in title_candidates:
            node = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
            if node and node.get_text(strip=True):
                headline = node.get_text(strip=True)
                break

        # Body paragraphs within article container
        # Common container ids/classes observed on DailyMail
        containers = soup.select(
            "#js-article-text, .article-text, .mol-para-with-font"
        )
        text_parts: List[str] = []

        if containers:
            for c in containers:
                for p in c.find_all("p"):
                    txt = p.get_text(" ", strip=True)
                    if txt:
                        text_parts.append(txt)
        else:
            # fallback: gather all <p>, filter boilerplate
            for p in soup.find_all("p"):
                txt = p.get_text(" ", strip=True)
                if txt and "Follow Daily Mail" not in txt and "e-mail" not in txt:
                    text_parts.append(txt)

        article_text = " ".join(text_parts).strip()

        if not headline or not article_text:
            self.log.debug("Missing content for %s", url)
            return None

        return {"headline": headline, "article": article_text, "url": url}

    # ---------------- HTTP ---------------- #

    def _get(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=self.cfg.timeout)
            if resp.status_code != 200:
                self.log.debug("GET %s -> %s", url, resp.status_code)
                return None
            return resp.text
        except requests.RequestException as e:
            self.log.debug("Request error: %s", e)
            return None

