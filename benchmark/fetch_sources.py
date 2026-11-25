"""Fetch rewrite sources into a local, ignored cache; never alter the URL list."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

SOURCE_PATH = Path("benchmark/source_articles.json")
CACHE_PATH = Path("benchmark/source_articles.local.json")
AGENT = "VEX-Bench-source-fetcher"
USER_AGENT = f"{AGENT}/1.0 (+https://github.com/HanxunH/VEX-Bench)"
TIMEOUT = (10, 30)


class _ArticleParser(HTMLParser):
    """Collect article/main content and paragraphs outside unwanted elements."""

    _ignored = {"script", "style", "nav", "header", "footer"}
    _void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ignored_depth = 0
        self.depth = {"article": 0, "main": 0}
        self.regions = {"article": [], "main": []}
        self.paragraph = None
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if self.ignored_depth:
            if tag not in self._void:
                self.ignored_depth += 1
        elif tag in self._ignored:
            self.ignored_depth = 1
        elif tag in self.depth:
            self.depth[tag] += 1
        elif tag == "p":
            self.paragraph = []

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in self.depth and self.depth[tag]:
            self.depth[tag] -= 1
        elif tag == "p" and self.paragraph is not None:
            self.paragraphs.append(" ".join(self.paragraph))
            self.paragraph = None

    def handle_data(self, data):
        if self.ignored_depth:
            return
        for tag in self.depth:
            if self.depth[tag]:
                self.regions[tag].append(data)
        if self.paragraph is not None:
            self.paragraph.append(data)


def extract_article_text(html):
    """Prefer article/main text, otherwise the longest paragraph; collapse whitespace."""
    parser = _ArticleParser()
    parser.feed(html)
    for tag in ("article", "main"):
        text = " ".join(" ".join(parser.regions[tag]).split())
        if text:
            return text
    paragraphs = [" ".join(p.split()) for p in parser.paragraphs]
    return max(paragraphs, key=len, default="")


def _load_entries(source_path, cache_path):
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    cache_path = Path(cache_path)
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    previous = {(domain, item["topic"]): item for domain, items in cached.items() for item in items}
    entries = {}
    for domain, articles in source.items():
        entries[domain] = []
        for article in articles:
            old = previous.get((domain, article["topic"]), {})
            entry = dict(article)
            entry["original_text"] = old.get("original_text") or article.get("original_text") or ""
            entry["fetched_at"] = old.get("fetched_at", "")
            entry["status"] = old.get("status", "ok" if entry["original_text"].strip() else "")
            entries[domain].append(entry)
    return entries


def _has_text(entry):
    return bool((entry.get("original_text") or "").strip())


def report_coverage(entries):
    """Print totals, domain/publisher coverage, and prior unsuccessful URLs."""
    rows = [(domain, entry) for domain, articles in entries.items() for entry in articles]
    fetched = sum(_has_text(entry) for _, entry in rows)
    print(f"Coverage: {len(rows)} total, {fetched} fetched, {len(rows) - fetched} missing")
    domains = defaultdict(list)
    publishers = defaultdict(list)
    for domain, entry in rows:
        domains[domain].append(entry)
        publishers[urlsplit(entry["source_url"]).hostname or "unknown"].append(entry)
    for label, groups in (("Domains", domains), ("Publishers", publishers)):
        print(f"{label}:")
        for name, group in sorted(groups.items()):
            count = sum(_has_text(entry) for entry in group)
            statuses = Counter(entry.get("status") for entry in group)
            print(f"  {name}: {len(group)} total, {count} fetched, {len(group) - count} missing; "
                  + ", ".join(f"{status}={statuses[status]}" for status in ("ok", "blocked", "failed", "empty")))
    failures = [entry for _, entry in rows if entry.get("status") in ("blocked", "failed", "empty")]
    print("Previously blocked/failed/empty URLs:")
    if not failures:
        print("  none")
    for entry in failures:
        print(f"  {entry['status']}: {entry['source_url']}")
    return len(rows) - fetched


class _Fetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.robots = {}
        self.last_request = None

    def get(self, url):
        if self.last_request is not None:
            remaining = 1.0 - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)
        try:
            return self.session.get(url, timeout=TIMEOUT, allow_redirects=False)
        finally:
            self.last_request = time.monotonic()

    def allowed(self, url):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            return False
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            try:
                response = self.get(origin + "/robots.txt")
                if not 200 <= response.status_code < 300:
                    self.robots[origin] = None
                else:
                    parser = RobotFileParser()
                    parser.set_url(origin + "/robots.txt")
                    parser.parse(response.text.splitlines())
                    self.robots[origin] = parser
            except requests.RequestException:
                self.robots[origin] = None
        parser = self.robots[origin]
        return parser is not None and parser.can_fetch(AGENT, url)

    def fetch(self, url):
        # Redirect destinations need their own robots check before any request.
        for _ in range(6):
            if not self.allowed(url):
                return "blocked", ""
            try:
                response = self.get(url)
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        return "failed", ""
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                text = extract_article_text(response.text)
                return ("ok", text) if text else ("empty", "")
            except requests.RequestException:
                return "failed", ""
        return "failed", ""


def fetch_sources(source_path=SOURCE_PATH, cache_path=CACHE_PATH, force=False):
    """Fetch uncached entries, printing status and saving progress locally."""
    source_path, cache_path = Path(source_path), Path(cache_path)
    if source_path.resolve() == cache_path.resolve():
        raise ValueError("The local cache must be separate from the shipped source file")
    entries = _load_entries(source_path, cache_path)
    fetcher = _Fetcher()
    for articles in entries.values():
        for entry in articles:
            if _has_text(entry) and not force:
                print(f"[ok] {entry['source_url']} (cached)")
                continue
            status, text = fetcher.fetch(entry["source_url"])
            entry["status"] = status
            entry["fetched_at"] = datetime.now(timezone.utc).isoformat()
            if text:
                entry["original_text"] = text
            print(f"[{status}] {entry['source_url']}", flush=True)
            cache_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Also write when every entry was already cached or the list was empty.
    cache_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report coverage without fetching")
    mode.add_argument("--fetch", action="store_true", help="fetch sources missing local text")
    parser.add_argument("--force", action="store_true", help="re-fetch all sources")
    args = parser.parse_args(argv)
    if args.force and not args.fetch:
        parser.error("--force requires --fetch")
    if args.fetch:
        entries = fetch_sources(SOURCE_PATH, CACHE_PATH, force=args.force)
        return 1 if report_coverage(entries) else 0
    report_coverage(_load_entries(SOURCE_PATH, CACHE_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
