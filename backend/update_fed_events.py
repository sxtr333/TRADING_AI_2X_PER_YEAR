#!/usr/bin/env python3
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import List, Optional
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


FEEDS = [
    {
        "category": "Press",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
    },
    {
        "category": "Speeches",
        "url": "https://www.federalreserve.gov/feeds/speeches.xml",
    },
    {
        "category": "FOMC",
        "url": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    },
    {
        "category": "Calendar",
        "url": "https://www.federalreserve.gov/feeds/fomccalendars.xml",
    },
]

CALENDAR_HTML = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def fetch_url(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "ALADIN-FED/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_rss(xml_bytes: bytes, category: str) -> List[dict]:
    items = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return items

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title or not pub:
            continue
        try:
            dt = parsedate_to_datetime(pub)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except Exception:
            continue
        items.append(
            {
                "timestamp": dt.isoformat(),
                "title": title,
                "url": link,
                "category": category,
                "source": "fed_rss",
            }
        )
    return items


class TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: List[str] = []

    def handle_data(self, data: str) -> None:
        clean = data.strip()
        if clean:
            self.text.append(clean)


def parse_calendar_dates(html_bytes: bytes) -> List[dict]:
    parser = TextCollector()
    parser.feed(html_bytes.decode("utf-8", errors="ignore"))
    joined = " ".join(parser.text)

    pattern = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2})(?:-\d{1,2}|–\d{1,2}|—\d{1,2})?,\s*(\d{4})"
    )

    events = []
    for match in pattern.finditer(joined):
        month_name, day_str, year_str = match.groups()
        month = MONTHS.get(month_name)
        if not month:
            continue
        day = int(day_str)
        year = int(year_str)
        dt = datetime(year, month, day, 18, 0, tzinfo=timezone.utc)
        events.append(
            {
                "timestamp": dt.isoformat(),
                "title": f"FOMC Meeting ({month_name} {day}, {year})",
                "url": CALENDAR_HTML,
                "category": "FOMC",
                "source": "fed_calendar_html",
            }
        )
    return events


def dedupe(events: List[dict]) -> List[dict]:
    seen = set()
    unique = []
    for evt in events:
        key = (evt.get("timestamp"), evt.get("title"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(evt)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="html/fed_events.json")
    parser.add_argument("--max-items", type=int, default=600)
    args = parser.parse_args()

    events: List[dict] = []
    for feed in FEEDS:
        try:
            xml = fetch_url(feed["url"])
            events.extend(parse_rss(xml, feed["category"]))
        except Exception:
            continue

    try:
        html = fetch_url(CALENDAR_HTML)
        events.extend(parse_calendar_dates(html))
    except Exception:
        pass

    events = dedupe(events)
    events.sort(key=lambda x: x["timestamp"], reverse=True)
    events = events[: args.max_items]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "events": events,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
