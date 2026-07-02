"""Load the SHL Individual Test Solutions catalog from data/catalog.json.

The scraper (scripts/scrape_catalog.py) is responsible for producing this file.
This module only reads, validates, and exposes helpers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

CATALOG_PATH = os.environ.get(
    "SHL_CATALOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.json"),
)


@dataclass
class Assessment:
    name: str
    url: str
    test_type: str  # 'A','B','C','D','E','K','P','S'
    description: str = ""
    job_levels: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    duration_minutes: Optional[int] = None
    remote_testing: Optional[bool] = None
    adaptive_irt: Optional[bool] = None

    def searchable_text(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.job_levels),
            f"test type {self.test_type}",
            TEST_TYPE_NAMES.get(self.test_type, ""),
        ]
        return " \n ".join(p for p in parts if p)


TEST_TYPE_NAMES: Dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
   
}

KEY_TO_TYPE: Dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

@lru_cache(maxsize=1)
def load_catalog(path: Optional[str] = None) -> List[Assessment]:
    p = path or CATALOG_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"catalog.json not found at {p}. Run: python scripts/scrape_catalog.py"
        )

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: List[Assessment] = []

    for row in raw:

        keys = row.get("keys") or []
        tt = "K"

        for k in keys:
            if k in KEY_TO_TYPE:
                tt = KEY_TO_TYPE[k]
                break

        if tt not in TEST_TYPE_NAMES:
            tt = "K"

        out.append(
            Assessment(
                name=row["name"].strip(),
                url=row["link"].strip(),
                test_type=tt,
                description=(row.get("description") or "").strip(),
                job_levels=row.get("job_levels") or [],
                languages=row.get("languages") or [],
                duration_minutes=(
                    int(row["duration"].split()[0])
                    if row.get("duration")
                    and row["duration"].split()[0].isdigit()
                    else None
                ),
                remote_testing=row.get("remote") == "yes",
                adaptive_irt=row.get("adaptive") == "yes",
            )
        )

    # dedupe by URL
    seen, dedup = set(), []
    for a in out:
        if a.url in seen:
            continue
        seen.add(a.url)
        dedup.append(a)

    return dedup

def by_name(name: str) -> Optional[Assessment]:
    n = name.strip().lower()
    for a in load_catalog():
        if a.name.lower() == n:
            return a
    return None
