"""Hybrid retrieval: BM25 (lexical) + TF-IDF cosine (semantic-ish).

Rationale: the assignment restricts final URLs to catalog items. We need a
retriever that (a) works offline, (b) has no heavy model dependency so it fits
free-tier hosting cold-starts, (c) reliably surfaces items when the user mentions
skills / job families / SHL product names (e.g. "OPQ", "GSA").

Hybrid ranking robustly handles both keyword queries (BM25) and paraphrased
descriptions (TF-IDF cosine). Reciprocal Rank Fusion combines the two.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import Assessment, load_catalog


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


@dataclass
class RetrievalHit:
    assessment: Assessment
    score: float


class Retriever:
    def __init__(self, assessments: List[Assessment]):
        self.assessments = assessments
        self._docs = [a.searchable_text() for a in assessments]
        self._tokens = [_tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(self._tokens)
        self._vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True
        )
        self._mat = self._vec.fit_transform(self._docs)

    def _bm25_ranks(self, query: str) -> np.ndarray:
        scores = self._bm25.get_scores(_tokenize(query))
        return scores

    def _tfidf_ranks(self, query: str) -> np.ndarray:
        qv = self._vec.transform([query])
        return cosine_similarity(qv, self._mat).ravel()

    def search(
        self,
        query: str,
        k: int = 10,
        test_types: Optional[List[str]] = None,
        max_duration: Optional[int] = None,
    ) -> List[RetrievalHit]:
        if not query.strip():
            return []
        # bm = self._bm25_ranks(query)
        # tf = self._tfidf_ranks(query)
        # # rank fusion (RRF)
        # bm_order = np.argsort(-bm)
        # tf_order = np.argsort(-tf)
        # rank_bm = np.empty_like(bm_order); rank_bm[bm_order] = np.arange(len(bm))
        # rank_tf = np.empty_like(tf_order); rank_tf[tf_order] = np.arange(len(tf))
        # fused = 1.0 / (60 + rank_bm) + 1.0 / (60 + rank_tf)

        bm = self._bm25_ranks(query)
        tf = self._tfidf_ranks(query)

        # normalize both scores
        if bm.max() > 0:
         bm = bm / bm.max()

        if tf.max() > 0:
         tf = tf / tf.max()

        # weighted fusion
        fused = 0.65 * bm + 0.35 * tf

        idx = np.argsort(-fused)
        hits: List[RetrievalHit] = []
        for i in idx:
            a = self.assessments[i]
            if test_types and a.test_type not in test_types:
                continue
            # if max_duration and a.duration_minutes and a.duration_minutes > max_duration:
            #     continue
            if (
               max_duration is not None
               and a.duration_minutes is not None
               and a.duration_minutes > max_duration
             ):
              continue            
            hits.append(RetrievalHit(a, float(fused[i])))
            if len(hits) >= k:
                break
        return hits


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(load_catalog())
