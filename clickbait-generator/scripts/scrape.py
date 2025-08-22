import argparse
import logging
from clickbait.scraper import DailyMailScraper, ScrapeConfig


def main():
    parser = argparse.ArgumentParser(description="Scrape DailyMail articles to CSV.")
    parser.add_argument(
        "--start-urls",
        nargs="+",
        required=True,
        help="Listing or section URLs to start from.",
    )
    parser.add_argument("--out", default="data/raw/dailymail.csv", help="Output CSV path.")
    parser.add_argument("--max-articles", type=int, default=None, help="Limit total articles.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds).")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout (seconds).")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = ScrapeConfig(
        delay_secs=args.delay,
        timeout=args.timeout,
        max_articles=args.max_articles,
    )
    scraper = DailyMailScraper(cfg)
    scraper.scrape_to_csv(args.start_urls, out_csv_path=args.out, append=args.append)


if __name__ == "__main__":
    main()
