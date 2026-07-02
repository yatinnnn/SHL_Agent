"""Scrape SHL Individual Test Solutions catalog into data/catalog.json.

The SHL catalog page uses a paginated table with a `type=2` param for
Individual Test Solutions. Each product row links to a detail page whose
top table exposes: Description, Job levels, Languages, Assessment length,
Remote Testing, and a "Test Type" cell whose colored letters (A/B/C/D/E/K/P/S)
tell us the test_type code.

Usage:
    python scripts/scrape_catalog.py            # scrape all Individual solutions
    python scripts/scrape_catalog.py --limit 20 # smoke test

Notes:
- Uses requests + BeautifulSoup only. No JS rendering needed.
- Polite: 1s between requests, retries on 5xx, custom UA.
- Writes data/catalog.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LISTING = "https://www.shl.com/solutions/products/product-catalog/"
UA = "Mozilla/5.0 (compatible; SHL-Catalog-Scraper/1.0; +intern-assignment)"

TYPE_CODES = {"A", "B", "C", "D", "E", "K", "P", "S"}


def _get(url: str, tries: int = 4, sleep: float = 1.0) -> str:
    for i in range(tries):
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        if r.status_code == 200:
            return r.text
        if 500 <= r.status_code < 600:
            time.sleep(1.5 * (i + 1))
            continue
        r.raise_for_status()
    r.raise_for_status()
    return ""


def parse_listing_page(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    rows: List[Dict[str, str]] = []
    # Product rows have a link in the product name cell.
    for a in soup.select("a[href*='/solutions/products/product-catalog/view/']"):
        href = a.get("href", "").strip()
        name = a.get_text(strip=True)
        if not name or not href:
            continue
        rows.append({"name": name, "url": urljoin(BASE, href)})
    # dedupe
    seen, out = set(), []
    for r in rows:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        out.append(r)
    return out


def list_all_individual(limit: Optional[int] = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    start = 0
    step = 12  # SHL paginates by 12
    while True:
        url = f"{LISTING}?start={start}&type=2"
        html = _get(url)
        page = parse_listing_page(html)
        if not page:
            break
        new = [r for r in page if r["url"] not in {o["url"] for o in out}]
        if not new:
            break
        out.extend(new)
        print(f"  page start={start}: +{len(new)} (total={len(out)})")
        if limit and len(out) >= limit:
            return out[:limit]
        start += step
        time.sleep(1.0)
    return out


def _cell_text(td) -> str:
    return re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()


def parse_detail(url: str, html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")
    data: Dict = {"url": url}
    # Name
    h1 = soup.find(["h1", "h2"])
    if h1:
        data["name"] = h1.get_text(strip=True)

    # Info rows: <div class="product-catalogue-training-calendar__row"> ... label/value pairs
    for row in soup.select(".product-catalogue-training-calendar__row, .product-catalogue__row, tr"):
        text = _cell_text(row)
        low = text.lower()
        if low.startswith("description"):
            data["description"] = text.split(":", 1)[-1].strip() if ":" in text else text.replace("Description", "", 1).strip()
        elif "job levels" in low:
            v = text.split(":", 1)[-1] if ":" in text else text.replace("Job levels", "", 1)
            data["job_levels"] = [x.strip() for x in re.split(r"[,;]", v) if x.strip()]
        elif "languages" in low:
            v = text.split(":", 1)[-1] if ":" in text else text.replace("Languages", "", 1)
            data["languages"] = [x.strip() for x in re.split(r"[,;]", v) if x.strip()]
        elif "assessment length" in low or "completion time" in low:
            m = re.search(r"(\d+)", text)
            if m:
                data["duration_minutes"] = int(m.group(1))
        elif "remote testing" in low:
            data["remote_testing"] = "yes" in low
        elif "adaptive" in low:
            data["adaptive_irt"] = "yes" in low

    # Fallback description = first meaningful paragraph
    if not data.get("description"):
        p = soup.find("p")
        if p:
            data["description"] = p.get_text(" ", strip=True)

    # Test type letters — SHL renders them as small colored badges. Look for
    # elements containing single-letter text from the type set inside the
    # "Test Type" section.
    tt_codes: List[str] = []
    for badge in soup.select(".product-catalogue__key, .badge, span, li"):
        t = badge.get_text(strip=True)
        if t in TYPE_CODES and t not in tt_codes:
            tt_codes.append(t)
    # Prefer the first badge found near a "Test Type" label.
    data["test_type"] = tt_codes[0] if tt_codes else "K"
    data["test_types_all"] = tt_codes

    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max products (smoke test)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json"))
    args = ap.parse_args()

    print("Listing Individual Test Solutions...")
    products = list_all_individual(limit=args.limit)
    print(f"Found {len(products)} products. Fetching detail pages...")

    results: List[Dict] = []
    for i, p in enumerate(products, 1):
        try:
            html = _get(p["url"])
            detail = parse_detail(p["url"], html)
            # Prefer listing name if detail name missing
            detail.setdefault("name", p["name"])
            results.append(detail)
            print(f"  [{i}/{len(products)}] {detail.get('name','?')[:60]} — type={detail.get('test_type')}")
        except Exception as e:
            print(f"  [{i}/{len(products)}] FAILED {p['url']}: {e}")
        time.sleep(1.0)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} items -> {out_path}")


if __name__ == "__main__":
    main()
