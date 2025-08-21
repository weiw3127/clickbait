import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class DailyMailScraper:
    def __init__(self, base_url="https://www.dailymail.co.uk", delay=1):
        self.base_url = base_url
        self.delay = delay

    def scrape_article(self, url: str):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h2", {"class": "linkro-darkred"}).get_text(strip=True)
        paragraphs = soup.find_all("p")
        article = " ".join(p.get_text(strip=True) for p in paragraphs)

        return {"title": title, "article": article}

    def scrape_multiple(self, urls, out_path="data/raw/dailymail.csv"):
        data = []
        for url in urls:
            try:
                data.append(self.scrape_article(url))
                time.sleep(self.delay)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        pd.DataFrame(data).to_csv(out_path, index=False)
