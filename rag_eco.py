"""
RAG Retrieval Pipeline — Faculty of Economics (Farhat Abbas University Sétif 1)
ChromaDB version — drop-in replacement for the Neo4j RAG pipeline.

Pairs with: ingest_economics_vectordb.py

Install:
    pip install chromadb sentence-transformers rank-bm25 rapidfuzz loguru
    pip install BAAI/bge-reranker-v2-m3   # pulled automatically by sentence-transformers
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    raise ImportError("chromadb required: pip install chromadb")

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False
    logging.warning("rank_bm25 not installed — BM25 disabled. pip install rank-bm25")

try:
    from rapidfuzz import fuzz as _fuzz
    _FUZZ_OK = True
except ImportError:
    _FUZZ_OK = False
    logging.warning("rapidfuzz not installed — fuzzy matching disabled.")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG  (must match ingest_economics_vectordb.py)
# ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_economics_db")
CHROMA_COLLECTION  = "economics_chunks"
CHROMA_URLS_COLL   = "economics_chunks_urls"

EMBED_MODEL_NAME = "sentence-transformers/LaBSE"
RERANK_MODEL     = "BAAI/bge-reranker-v2-m3"
VECTOR_DIMENSIONS = 768

TOP_K_VECTOR  = 60
TOP_K_BM25    = 30
TOP_K_RERANK  = 20
TOP_K_FINAL   = 8

W_SEM    = 0.65
W_BM25   = 0.15
W_FUSED  = 0.45
W_RERANK = 0.55

RERANK_POWER   = 0.75
RERANK_MIN_CAL = 0.20

DYN_TOP_K_MIN        = 3
DYN_TOP_K_MAX        = 8
DYNAMIC_SCORE_MARGIN = 0.28

LANG_EXACT_BOOST  = 0.10
LANG_MISMATCH_PEN = 0.05

# Graph-style soft boosts using metadata classification depth
BOOST_EXACT      = 0.25   # chunk's classification_id matches a detected node slug
BOOST_PARENT     = 0.12   # parent level match
BOOST_GENERAL    = 0.04   # classification_label == "General"
BOOST_SIBLING    = 0.06

ANSWER_THRESHOLD_STRICT = 0.38
ANSWER_THRESHOLD_LOOSE  = 0.20

DEDUP_CHARS      = 200
EMBED_CACHE_SIZE = 256

_STOPWORDS: Set[str] = {
    "le","la","les","de","du","des","et","en","un","une","pour","avec",
    "dans","sur","par","est","ce","qui","que","quoi","comment","quel",
    "je","tu","il","elle","nous","vous","ils","elles","mon","ton","son",
    "mes","tes","ses","nos","vos","leurs","au","aux","ou","si","ne","pas",
    "the","a","an","and","or","of","in","to","for","is","are","was","were",
    "it","its","this","that","with","by","on","at","from","be","been","has",
    "have","had","do","does","did","will","would","could","should","may",
    "في","من","إلى","على","عن","مع","هو","هي","هم","هن","أنا","نحن",
    "أنت","أنتم","كان","كانت","يكون","التي","الذي","الذين","اللتي","ما",
    "لا","هذا","هذه","ذلك","تلك","قد","لقد","إن","أن","لكن","أو",
    "who","what","where","when","why","how","which",
    "qui","que","quoi","comment","où","quand","quel","quelle",
    "من","ماذا","أين","متى","كيف","أي",
}

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    text:        str
    score:       float
    metadata:    Dict = field(default_factory=dict)
    sem_score:   float = 0.0
    bm25_score:  float = 0.0
    fused_score: float = 0.0
    rerank_raw:  float = 0.0
    rerank_cal:  float = 0.0
    graph_boost: float = 0.0
    node_level:  int   = -1   # -1 = no boost, 0 = exact, 1 = parent, 99 = general


@dataclass
class AcademicNode:
    """Lightweight node built from chunk metadata (no external graph DB needed)."""
    node_id:   str
    name:      str
    label:     str
    path:      str
    aliases:   List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# CHROMA CLIENT — singleton
# ══════════════════════════════════════════════════════════════

_chroma_client = None
_chroma_lock   = threading.Lock()


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                _chroma_client = chromadb.PersistentClient(
                    path=CHROMA_PERSIST_DIR,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                log.info(f"✓ ChromaDB connected at '{CHROMA_PERSIST_DIR}'")
    return _chroma_client


def _get_collection(name: str):
    return _get_chroma().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ══════════════════════════════════════════════════════════════
# QUERY ANALYZER
# ══════════════════════════════════════════════════════════════

class QueryAnalyzer:
    _RE_AR = re.compile(r"[\u0600-\u06FF]")
    _RE_FR = re.compile(
        r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
        r"comment|quoi|qui|quel|quelle|où|quand|pourquoi)\b",
        re.IGNORECASE,
    )

    _INTENT_RULES: List[Tuple[str, List[str]]] = [
        ("node_query",   ["spécialité","specialization","filière","département","department",
                          "faculté","faculty","niveau","level","تخصص","قسم","كلية","شعبة"]),
        ("course_query", ["cours","course","module","matière","td","tp","semestre","semester",
                          "programme","مقياس","مادة","برنامج"]),
        ("admin_query",  ["inscription","registration","examen","exam","résultat","result",
                          "calendrier","emploi du temps","تسجيل","امتحان","نتيجة"]),
        ("person_lookup",["prof","dr ","docteur","enseignant","teacher","professeur",
                          "مدرس","أستاذ","دكتور","email","contact","responsable"]),
        ("general_info", []),
    ]

    def analyze(self, query: str) -> Dict:
        return {
            "query":      query,
            "language":   self._detect_language(query),
            "intent":     self._detect_intent(query),
            "keywords":   self._extract_keywords(query),
            "normalized": self._normalize(query),
        }

    def _detect_language(self, text: str) -> str:
        s = text[:300]
        ar_chars = len(self._RE_AR.findall(s))
        fr_words = len(self._RE_FR.findall(s))
        total    = max(len(s.split()), 1)
        if ar_chars >= max(1, len(s.strip()) // 8):
            return "ar"
        if fr_words / total >= 0.15:
            return "fr"
        return "en"

    def _detect_intent(self, query: str) -> str:
        q = query.lower()
        for intent, kws in self._INTENT_RULES:
            if any(kw in q for kw in kws):
                return intent
        return "general_info"

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = self._normalize(query).split()
        kws    = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
        return kws if kws else tokens

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return re.sub(r"\s+", " ", text).strip().lower()


# ══════════════════════════════════════════════════════════════
# ACADEMIC NODE INDEX — built from ChromaDB metadata
# ══════════════════════════════════════════════════════════════

class AcademicNodeIndex:
    """
    Replaces AcademicGraph.  Builds an in-memory node catalog from the
    classification metadata stored in every chunk during ingestion.

    No Neo4j, no external file needed — the node tree is reconstructed
    from the 'hierarchy_path', 'classification_label', 'classification_name',
    and 'classification_id' fields that the ingestion pipeline wrote.
    """

    _ABBR_MAP: Dict[str, List[str]] = {
        "économie":                    ["eco","economy","economics","macroéconomie","microéconomie"],
        "finance":                     ["fin","finances","financial"],
        "comptabilité":                ["compta","accounting","cca"],
        "gestion":                     ["management","mgmt","administration des affaires"],
        "commerce":                    ["commercial","trade","sci commerciales"],
        "marketing":                   ["mkt","market"],
        "audit":                       ["contrôle","révision","audit financier"],
        "banque":                      ["banking","banques","monnaie"],
        "assurance":                   ["insurance"],
        "sciences économiques":        ["eco","sciences eco","economic sciences"],
        "sciences commerciales":       ["commerce","sci com"],
        "sciences de gestion":         ["gestion","sci gestion","management sciences"],
        "licence":                     ["l1","l2","l3","bachelor"],
        "master":                      ["m1","m2"],
        "doctorat":                    ["phd","doc"],
        "general":                     ["général","générale","commun"],
    }

    def __init__(self):
        self.nodes:      Dict[str, AcademicNode] = {}   # slug → AcademicNode
        self.name_index: Dict[str, List[str]]    = {}   # lower name → [slugs]
        self.path_index: Dict[str, str]          = {}   # path string → slug
        self._loaded     = False
        self._load()

    def _load(self):
        try:
            coll = _get_collection(CHROMA_COLLECTION)
            # Fetch all metadata (no embeddings) — may be large; done once at startup
            result = coll.get(include=["metadatas"])
            metas  = result.get("metadatas") or []
            log.info(f"   Building node index from {len(metas)} chunk metadata records…")

            seen_slugs: Set[str] = set()
            for m in metas:
                if not m:
                    continue
                slug  = m.get("classification_id", "")
                name  = m.get("classification_name", "")
                label = m.get("classification_label", "General")
                path  = m.get("hierarchy_path", "")
                if not slug or slug in seen_slugs:
                    continue
                seen_slugs.add(slug)

                node = AcademicNode(
                    node_id=slug,
                    name=name,
                    label=label,
                    path=path,
                    aliases=self._generate_aliases(name),
                )
                self.nodes[slug] = node
                self.path_index[path] = slug

                # Index by name + aliases
                for term in [name.lower().strip()] + node.aliases:
                    if term:
                        self.name_index.setdefault(term, [])
                        if slug not in self.name_index[term]:
                            self.name_index[term].append(slug)

            self._loaded = True
            log.info(f"✓ Node index: {len(self.nodes)} unique classification nodes")
        except Exception as e:
            log.error(f"Failed to build node index: {e}")
            self._loaded = False

    def _generate_aliases(self, name: str) -> List[str]:
        name_lower = name.lower().strip()
        aliases: Set[str] = set()
        if name_lower in self._ABBR_MAP:
            aliases.update(self._ABBR_MAP[name_lower])
        for key, vals in self._ABBR_MAP.items():
            if key != name_lower and (key in name_lower or name_lower in key):
                aliases.update(vals)
        words = name_lower.split()
        if len(words) > 1:
            acronym = "".join(w[0] for w in words if w)
            if len(acronym) >= 2:
                aliases.add(acronym)
        aliases.discard(name_lower)
        return list(aliases)

    def detect_nodes(self, query: str) -> List[Tuple[AcademicNode, float]]:
        """
        Returns (node, confidence_score) sorted by confidence desc.
        Multi-strategy: exact substring → word overlap → fuzzy.
        """
        if not self._loaded:
            return []

        q_lower = query.lower().strip()
        scores:  Dict[str, float] = {}

        # Strategy 1: exact substring
        for term, slugs in self.name_index.items():
            if len(term) < 3:
                continue
            if term in q_lower:
                coverage = len(term) / max(len(q_lower), 1)
                s = min(2.0, coverage * 3.0)
                for slug in slugs:
                    scores[slug] = max(scores.get(slug, 0.0), s)

        # Strategy 2: word overlap
        q_words = set(q_lower.split()) - _STOPWORDS
        for term, slugs in self.name_index.items():
            t_words = set(term.split()) - _STOPWORDS
            if not t_words:
                continue
            overlap = q_words & t_words
            if overlap:
                s = len(overlap) / len(t_words) * 0.8
                for slug in slugs:
                    scores[slug] = max(scores.get(slug, 0.0), s)

        # Strategy 3: fuzzy (short queries)
        if _FUZZ_OK and len(q_lower.split()) <= 5:
            for term, slugs in self.name_index.items():
                if len(term) < 4:
                    continue
                ratio = _fuzz.partial_ratio(q_lower, term)
                if ratio >= 75:
                    s = (ratio / 100.0) * 0.65
                    for slug in slugs:
                        scores[slug] = max(scores.get(slug, 0.0), s)

        THRESHOLD = 0.35
        results: List[Tuple[AcademicNode, float]] = []
        for slug, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if sc < THRESHOLD:
                continue
            node = self.nodes.get(slug)
            if node and node.label != "General":
                results.append((node, sc))
                log.info(f"   🎯 Detected node: '{node.name}' [{node.label}] score={sc:.2f}")
            if len(results) >= 6:
                break
        return results

    def build_boost_map(
        self,
        matched: List[Tuple[AcademicNode, float]],
    ) -> Dict[str, Tuple[float, int]]:
        """
        Returns {classification_id: (boost_value, level)}.
        Level 0 = exact match, 1 = parent path contains slug, 99 = general.

        Since ChromaDB has no graph edges, we use path containment:
          - A chunk's hierarchy_path containing a matched node's name → it's
            at least a descendant.
          - We also match by classification_id directly.
        """
        boost_map: Dict[str, Tuple[float, int]] = {}

        if not matched:
            # No node detected → give all General chunks a soft lift
            self._add_general_boosts(boost_map)
            return boost_map

        matched_slugs = {n.node_id for n, _ in matched}
        matched_names = {n.name.lower() for n, _ in matched}

        for slug, node in self.nodes.items():
            level  = self._classify_depth(node, matched_slugs, matched_names)
            if level < 0:
                continue
            boost_vals = {0: BOOST_EXACT, 1: BOOST_PARENT, 2: 0.08, 3: 0.05}
            boost = boost_vals.get(level, 0.04)
            boost_map[slug] = (boost, level)

        self._add_general_boosts(boost_map)

        log.info(
            f"   📈 Boost map: {len(boost_map)} nodes  "
            f"L0={sum(1 for _,l in boost_map.values() if l==0)}  "
            f"L1={sum(1 for _,l in boost_map.values() if l==1)}  "
            f"L2+={sum(1 for _,l in boost_map.values() if l>=2)}"
        )
        return boost_map

    def _classify_depth(
        self,
        node: AcademicNode,
        matched_slugs: Set[str],
        matched_names: Set[str],
    ) -> int:
        """Returns 0-3 for depth, -1 if not related."""
        # Level 0: exact slug match
        if node.node_id in matched_slugs:
            return 0
        # Level 1: this node's path contains a matched node name (it's a descendant)
        path_lower = node.path.lower()
        for name in matched_names:
            if name and name in path_lower:
                return 1
        # Level 2: partial path match (shared ancestor keyword)
        for name in matched_names:
            name_words = set(name.split()) - _STOPWORDS
            path_words = set(path_lower.split())
            if name_words and len(name_words & path_words) >= 2:
                return 2
        return -1

    def _add_general_boosts(self, boost_map: Dict[str, Tuple[float, int]]):
        for slug, node in self.nodes.items():
            if node.label == "General":
                existing, _ = boost_map.get(slug, (0.0, 99))
                if BOOST_GENERAL > existing:
                    boost_map[slug] = (BOOST_GENERAL, 99)


# ══════════════════════════════════════════════════════════════
# SEMANTIC RETRIEVER — ChromaDB vector search
# ══════════════════════════════════════════════════════════════

class SemanticRetriever:
    def __init__(self):
        log.info(f"Loading {EMBED_MODEL_NAME}…")
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
        self._model  = SentenceTransformer(EMBED_MODEL_NAME, device=dev)
        self._cache: Dict[str, np.ndarray] = {}
        log.info("Embedder ready")

    def encode(self, texts: List[str]) -> np.ndarray:
        results: List[Optional[np.ndarray]] = []
        miss_idx:   List[int] = []
        miss_texts: List[str] = []
        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
            else:
                results.append(None)
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            embs = self._model.encode(
                miss_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for li, gi in enumerate(miss_idx):
                vec = embs[li]
                if len(self._cache) >= EMBED_CACHE_SIZE:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[miss_texts[li]] = vec
                results[gi] = vec
        valid = [r for r in results if r is not None]
        return np.vstack(valid) if valid else np.empty((0, VECTOR_DIMENSIONS))

    def search(
        self,
        variants: List[str],
        top_k: int,
        where: Optional[Dict] = None,
    ) -> Dict[str, RetrievedChunk]:
        """
        Search ChromaDB.  `where` is an optional ChromaDB metadata filter.
        Returns {chunk_id: RetrievedChunk}.
        """
        if not variants:
            return {}
        embeddings = self.encode(variants)
        if embeddings.shape[0] == 0:
            return {}

        coll = _get_collection(CHROMA_COLLECTION)
        candidates: Dict[str, RetrievedChunk] = {}

        for vec in embeddings:
            kwargs: Dict[str, Any] = {
                "query_embeddings": [vec.tolist()],
                "n_results":        min(top_k, coll.count() or 1),
                "include":          ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            try:
                res = coll.query(**kwargs)
            except Exception as e:
                log.warning(f"ChromaDB query error: {e}")
                continue

            for cid, text, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                # ChromaDB cosine distance → similarity
                sim = max(0.0, 1.0 - float(dist))
                if cid not in candidates or sim > candidates[cid].sem_score:
                    candidates[cid] = RetrievedChunk(
                        chunk_id=cid,
                        text=text or "",
                        score=sim,
                        metadata={
                            "title":               meta.get("title", ""),
                            "url":                 meta.get("url", ""),
                            "source_type":         meta.get("source_type", ""),
                            "language":            meta.get("language", ""),
                            "chunk_index":         meta.get("chunk_index", 0),
                            "url_id":              meta.get("url_id", ""),
                            "classification_label": meta.get("classification_label", ""),
                            "classification_name": meta.get("classification_name", ""),
                            "classification_id":   meta.get("classification_id", ""),
                            "hierarchy_path":      meta.get("hierarchy_path", ""),
                            "match_method":        meta.get("match_method", ""),
                            "confidence":          meta.get("confidence", 0.0),
                        },
                        sem_score=sim,
                    )
        return candidates


# ══════════════════════════════════════════════════════════════
# BM25 RETRIEVER — built from ChromaDB documents
# ══════════════════════════════════════════════════════════════

class BM25Retriever:
    def __init__(self):
        self._chunk_ids: List[str]       = []
        self._texts:     List[str]       = []
        self._meta:      Dict[str, Dict] = {}
        self._bm25 = None

        if not _BM25_OK:
            return

        log.info("Building BM25 index from ChromaDB…")
        qa = QueryAnalyzer()
        tokenized: List[List[str]] = []

        try:
            coll   = _get_collection(CHROMA_COLLECTION)
            result = coll.get(include=["documents", "metadatas"])
            ids    = result.get("ids", [])
            docs   = result.get("documents", [])
            metas  = result.get("metadatas", [])

            for cid, text, meta in zip(ids, docs, metas):
                if not cid or not (text or "").strip():
                    continue
                self._chunk_ids.append(cid)
                self._texts.append(text)
                self._meta[cid] = {
                    "title":               (meta or {}).get("title", ""),
                    "url":                 (meta or {}).get("url", ""),
                    "source_type":         (meta or {}).get("source_type", ""),
                    "language":            (meta or {}).get("language", "fr"),
                    "chunk_index":         (meta or {}).get("chunk_index", 0),
                    "url_id":              (meta or {}).get("url_id", ""),
                    "classification_label": (meta or {}).get("classification_label", ""),
                    "classification_name": (meta or {}).get("classification_name", ""),
                    "classification_id":   (meta or {}).get("classification_id", ""),
                    "hierarchy_path":      (meta or {}).get("hierarchy_path", ""),
                }
                tokenized.append(qa._normalize(text).split())
        except Exception as e:
            log.error(f"BM25 load failed: {e}")
            return

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            log.info(f"BM25 ready: {len(tokenized)} chunks")

    def search(self, keywords: List[str], top_k: int) -> List[Tuple[str, str, float]]:
        if self._bm25 is None or not keywords:
            return []
        raw = self._bm25.get_scores(keywords)
        mx  = raw.max()
        if mx <= 0:
            return []
        normed  = raw / mx
        results: List[Tuple[str, str, float]] = []
        for i in np.argsort(normed)[::-1][: top_k * 2]:
            if normed[i] <= 0:
                continue
            results.append((self._chunk_ids[i], self._texts[i], float(normed[i])))
            if len(results) >= top_k:
                break
        return results

    def get_meta(self, cid: str) -> Optional[Dict]:
        return self._meta.get(cid)


# ══════════════════════════════════════════════════════════════
# FUSION — hybrid scoring
# ══════════════════════════════════════════════════════════════

def _fingerprint(text: str) -> str:
    return hashlib.md5(
        re.sub(r"\s+", " ", text[:DEDUP_CHARS]).strip().lower().encode()
    ).hexdigest()


def fuse_results(
    semantic:    Dict[str, RetrievedChunk],
    bm25:        List[Tuple[str, str, float]],
    boost_map:   Dict[str, Tuple[float, int]],   # classification_id → (boost, level)
    query_lang:  str,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> List[RetrievedChunk]:
    pool: Dict[str, Dict[str, Any]] = {}

    for cid, chunk in semantic.items():
        pool[cid] = {"sem": chunk.sem_score, "bm25": 0.0, "text": chunk.text, "meta": chunk.metadata}

    for cid, text, score in bm25:
        if cid not in pool:
            meta = (bm25_retriever.get_meta(cid) or {}) if bm25_retriever else {}
            pool[cid] = {"sem": 0.0, "bm25": score, "text": text, "meta": meta}
        else:
            pool[cid]["bm25"] = max(pool[cid]["bm25"], score)

    fused:    List[RetrievedChunk] = []
    seen_fp:  Dict[str, float]    = {}

    for cid, d in pool.items():
        sem = float(d["sem"]); bm = float(d["bm25"])
        if sem == 0.0 and bm == 0.0:
            continue

        score = W_SEM * sem + W_BM25 * bm

        # Language boost
        meta       = d["meta"] if isinstance(d["meta"], dict) else {}
        chunk_lang = meta.get("language", "")
        if chunk_lang and chunk_lang == query_lang:
            score += LANG_EXACT_BOOST
        elif chunk_lang and chunk_lang in ("ar","fr","en") and chunk_lang != query_lang:
            score -= LANG_MISMATCH_PEN

        # Soft graph boost from classification_id
        cls_id = meta.get("classification_id", "")
        gb, level = boost_map.get(cls_id, (0.0, -1))
        score += gb

        fp = _fingerprint(d["text"])
        if fp in seen_fp and score <= seen_fp[fp]:
            continue
        seen_fp[fp] = score

        fused.append(RetrievedChunk(
            chunk_id=cid, text=d["text"], score=score,
            metadata=meta, sem_score=sem, bm25_score=bm,
            fused_score=score, graph_boost=gb, node_level=level,
        ))

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


# ══════════════════════════════════════════════════════════════
# RERANKER
# ══════════════════════════════════════════════════════════════

class Reranker:
    def __init__(self):
        log.info(f"Loading reranker: {RERANK_MODEL}")
        self._model = CrossEncoder(RERANK_MODEL)

    def rerank(
        self,
        query:      str,
        chunks:     List[RetrievedChunk],
        top_k:      int,
        min_cal:    float = RERANK_MIN_CAL,
        query_lang: str   = "",
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs  = [(query, c.text) for c in chunks]
        logits = self._model.predict(pairs)

        for chunk, logit in zip(chunks, logits):
            raw = 1.0 / (1.0 + float(np.exp(-float(logit))))
            cal = raw ** RERANK_POWER
            chunk_lang = chunk.metadata.get("language", "")
            if query_lang and chunk_lang == query_lang:
                cal = min(1.0, cal + 0.05)
            elif query_lang and chunk_lang and chunk_lang != query_lang:
                cal = max(0.0, cal - 0.04)
            chunk.rerank_raw = raw
            chunk.rerank_cal = cal
            chunk.score      = W_FUSED * chunk.fused_score + W_RERANK * cal

        return sorted(
            [c for c in chunks if c.rerank_cal >= min_cal],
            key=lambda c: c.score,
            reverse=True,
        )[:top_k]


# ══════════════════════════════════════════════════════════════
# QUERY ROUTER
# ══════════════════════════════════════════════════════════════

@dataclass
class RetrievalStrategy:
    mode:               str
    top_k_multiplier:   float = 1.0
    min_score_override: Optional[float] = None
    expand_label:       Optional[str]   = None  # restrict to a Chroma metadata label

class QueryRouter:
    _MAP: Dict[str, RetrievalStrategy] = {
        "node_query":    RetrievalStrategy("metadata_first",   top_k_multiplier=1.2),
        "course_query":  RetrievalStrategy("hybrid",           top_k_multiplier=1.5),
        "admin_query":   RetrievalStrategy("hybrid"),
        "person_lookup": RetrievalStrategy("wide",             min_score_override=ANSWER_THRESHOLD_LOOSE),
        "general_info":  RetrievalStrategy("semantic_first"),
    }

    def route(self, analysis: Dict) -> RetrievalStrategy:
        intent   = analysis.get("intent", "general_info")
        strategy = self._MAP.get(intent, self._MAP["general_info"])
        log.info(f"   🗺  Router: intent={intent}  mode={strategy.mode}")
        return strategy


# ══════════════════════════════════════════════════════════════
# MAIN RAG RETRIEVER
# ══════════════════════════════════════════════════════════════

class RAGRetriever:

    def __init__(self):
        log.info("═" * 55)
        log.info("Initializing ChromaDB RAGRetriever — Economics Faculty")
        log.info("═" * 55)

        self.analyzer  = QueryAnalyzer()
        self.router    = QueryRouter()
        self.nodes     = AcademicNodeIndex()
        self.semantic  = SemanticRetriever()
        self.bm25      = BM25Retriever()
        self.reranker  = Reranker()

        log.info("✅ RAGRetriever ready")

    # ── Public entry point ────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> List[RetrievedChunk]:
        log.info(f"\n🔍 QUERY: {query[:120]}")

        # Step 1: Analyse
        analysis = self.analyzer.analyze(query)
        lang     = analysis["language"]
        intent   = analysis["intent"]
        keywords = analysis["keywords"]
        log.info(f"   Language={lang}  Intent={intent}  Keywords={keywords[:6]}")

        # Step 2: Route
        strategy = self.router.route(analysis)

        # Step 3: Detect academic nodes
        matched = self.nodes.detect_nodes(query)

        # Step 4: Build boost map (classification_id → boost)
        boost_map = self.nodes.build_boost_map(matched)

        # Step 5: Build optional Chroma where-filter for metadata_first mode
        where_filter = None
        if strategy.mode == "metadata_first" and matched:
            # Soft approach: filter to chunks whose classification_id is in matched set
            # or whose label matches a relevant level
            matched_slugs = [n.node_id for n, _ in matched[:3]]
            if len(matched_slugs) == 1:
                where_filter = {"classification_id": matched_slugs[0]}
            elif len(matched_slugs) > 1:
                where_filter = {"classification_id": {"$in": matched_slugs}}

        # Step 6: Retrieval
        variants   = self._build_variants(query, analysis)
        top_k_vec  = max(TOP_K_VECTOR, int(TOP_K_VECTOR * strategy.top_k_multiplier))
        top_k_bm25 = max(TOP_K_BM25,  int(TOP_K_BM25  * strategy.top_k_multiplier))

        sem_hits  = self.semantic.search(variants, top_k_vec, where=where_filter)
        bm25_hits = self.bm25.search(keywords, top_k_bm25)

        # If metadata filter yielded too few results, fall back to global search
        if len(sem_hits) < 5 and where_filter is not None:
            log.info("   ⚠️ Filtered search sparse → expanding to global")
            sem_hits = self.semantic.search(variants, top_k_vec, where=None)

        log.info(f"   Semantic={len(sem_hits)}  BM25={len(bm25_hits)}")

        # Step 7: Fusion
        fused = fuse_results(sem_hits, bm25_hits, boost_map, lang, self.bm25)

        # Stage 2 fallback: general nodes only
        if not fused:
            log.warning("   ⚠️ Fusion empty → Fallback: General label only")
            gen_filter = {"classification_label": "General"}
            sem_hits2  = self.semantic.search(variants, TOP_K_VECTOR, where=gen_filter)
            bm25_hits2 = self.bm25.search(keywords, TOP_K_BM25)
            fused      = fuse_results(sem_hits2, bm25_hits2, {}, lang, self.bm25)

        # Stage 3 fallback: full corpus, no filter, no boost
        if not fused:
            log.warning("   ⚠️ General fallback empty → Fallback: full corpus")
            sem_hits3  = self.semantic.search(variants, TOP_K_VECTOR, where=None)
            bm25_hits3 = self.bm25.search(keywords, TOP_K_BM25)
            fused      = fuse_results(sem_hits3, bm25_hits3, {}, lang, self.bm25)

        if not fused:
            log.warning("   ❌ All fallbacks exhausted — NO_ANSWER")
            return []

        # Step 8: Rerank
        rerank_min = (
            0.12
            if intent in ("person_lookup", "node_query") or len(query.split()) <= 2
            else RERANK_MIN_CAL
        )
        reranked = self.reranker.rerank(
            query, fused[:TOP_K_RERANK], top_k=TOP_K_RERANK,
            min_cal=rerank_min, query_lang=lang,
        )

        if not reranked:
            log.warning("   ⚠️ Reranker dropped all → using top fused")
            reranked = fused[:DYN_TOP_K_MAX]

        # Step 9: Dynamic top-k
        if reranked:
            best  = reranked[0].score
            floor = best - DYNAMIC_SCORE_MARGIN
            final = [c for c in reranked if c.score >= floor][:DYN_TOP_K_MAX]
            if len(final) < DYN_TOP_K_MIN:
                final = reranked[:DYN_TOP_K_MIN]
        else:
            final = []

        # Step 10: Answerability gate
        threshold = (
            strategy.min_score_override
            if strategy.min_score_override is not None
            else (
                ANSWER_THRESHOLD_LOOSE
                if intent in ("person_lookup", "node_query") or len(query.split()) <= 2
                else ANSWER_THRESHOLD_STRICT
            )
        )

        if not final or final[0].score < threshold:
            best_score = final[0].score if final else 0.0
            log.warning(
                f"   ❌ Answerability gate FAILED: best={best_score:.3f} "
                f"< threshold={threshold:.3f} → NO_ANSWER"
            )
            return []

        log.info(f"   ✅ Final: {len(final)} chunks  scores={[round(c.score,3) for c in final]}")
        return final

    # ── Helpers ───────────────────────────────────────────────

    def _build_variants(self, query: str, analysis: Dict) -> List[str]:
        variants = [query]
        intent   = analysis.get("intent", "")
        lang     = analysis.get("language", "en")

        if intent == "course_query":
            variants.append(f"cours module programme {query}")
        elif intent == "node_query":
            variants.append(f"département spécialité filière {query}")
        elif intent == "admin_query":
            variants.append(f"inscription administration {query}")

        if lang == "ar" and len(query.split()) <= 5:
            ar_to_fr = {
                "تخصص": "spécialité", "قسم": "département", "كلية": "faculté",
                "مقياس": "cours", "ماستر": "master", "ليسانس": "licence",
                "اقتصاد": "économie", "مالية": "finance", "محاسبة": "comptabilité",
            }
            translated = query
            for ar, fr in ar_to_fr.items():
                translated = translated.replace(ar, fr)
            if translated != query:
                variants.append(translated)

        return variants[:3]

    def search_nodes(self, query: str, label: Optional[str] = None) -> List[Dict]:
        matched = self.nodes.detect_nodes(query)
        if label:
            matched = [(n, s) for n, s in matched if n.label == label]
        return [
            {"node_id": n.node_id, "name": n.name, "label": n.label,
             "path": n.path, "score": round(s, 3)}
            for n, s in matched
        ]

    def list_nodes(self, label: Optional[str] = None) -> List[Dict]:
        nodes = self.nodes.nodes.values()
        if label:
            nodes = [n for n in nodes if n.label == label]
        return [
            {"node_id": n.node_id, "name": n.name, "label": n.label, "path": n.path}
            for n in sorted(nodes, key=lambda x: x.path)
        ]

    def close(self):
        pass  # ChromaDB PersistentClient flushes automatically


# ══════════════════════════════════════════════════════════════
# SINGLETON + PUBLIC API
# ══════════════════════════════════════════════════════════════

_retriever: Optional[RAGRetriever] = None
_lock = threading.Lock()


def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = RAGRetriever()
    return _retriever


def retrieve_for_llm(query: str, top_k: int = TOP_K_FINAL) -> List[Dict]:
    """
    Main public API.
    Returns a list of dicts.  First element contains 'llm_context' with
    a formatted text block ready to pass to your LLM.
    """
    chunks = _get_retriever().retrieve(query, top_k)
    if not chunks:
        return []

    result = []
    for c in chunks:
        meta = c.metadata or {}
        result.append({
            "chunk_id":            c.chunk_id,
            "text":                c.text,
            "score":               round(c.score,       4),
            "sem_score":           round(c.sem_score,   4),
            "bm25_score":          round(c.bm25_score,  4),
            "fused_score":         round(c.fused_score, 4),
            "rerank_cal":          round(c.rerank_cal,  4),
            "graph_boost":         round(c.graph_boost, 4),
            "node_level":          c.node_level,
            # source info
            "url":                 meta.get("url", ""),
            "title":               meta.get("title", ""),
            "source_type":         meta.get("source_type", ""),
            "language":            meta.get("language", ""),
            "chunk_index":         meta.get("chunk_index"),
            # classification info
            "classification_label": meta.get("classification_label", ""),
            "classification_name": meta.get("classification_name", ""),
            "hierarchy_path":      meta.get("hierarchy_path", ""),
        })

    if result:
        result[0]["llm_context"] = _format_context(chunks)

    return result


def search_nodes(query: str, label: Optional[str] = None) -> List[Dict]:
    return _get_retriever().search_nodes(query, label)


def list_nodes(label: Optional[str] = None) -> List[Dict]:
    return _get_retriever().list_nodes(label)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    sep    = "\n" + "─" * 60 + "\n"
    blocks = []
    for i, c in enumerate(chunks, 1):
        meta       = c.metadata or {}
        title      = meta.get("title", "Unknown Source")
        url        = meta.get("url", "")
        stype      = meta.get("source_type", "")
        cls_name   = meta.get("classification_name", "")
        hp         = meta.get("hierarchy_path", "")
        level_tag  = f" [depth={c.node_level}]" if c.node_level >= 0 else ""

        lines = [f"[{i}] {title}{level_tag}"]
        if cls_name:
            lines.append(f"📚 Classification : {cls_name}  ({hp})")
        if stype == "pdf" and url:
            lines.append(f"📄 Source PDF     : {url}")
        elif url:
            lines.append(f"🔗 Source         : {url}")
        lines.append(c.text.strip())
        blocks.append("\n".join(lines))
    return sep.join(blocks)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python rag_pipeline_chroma.py \"your query\" [--top-k N]\n"
            "  python rag_pipeline_chroma.py --search-nodes \"term\" [--label Department]\n"
            "  python rag_pipeline_chroma.py --list-nodes [--label Specialization]\n"
        )
        sys.exit(1)

    query      = None
    top_k      = TOP_K_FINAL
    nodes_q    = None
    node_label = None
    list_lbl   = None
    do_list    = False

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1]); i += 2
        elif arg == "--search-nodes" and i + 1 < len(sys.argv):
            nodes_q = sys.argv[i + 1]; i += 2
        elif arg == "--label" and i + 1 < len(sys.argv):
            node_label = sys.argv[i + 1]; i += 2
        elif arg == "--list-nodes":
            do_list = True; i += 1
        else:
            if query is None and not arg.startswith("--"):
                query = arg
            i += 1

    if do_list:
        for n in list_nodes(node_label):
            print(f"  {n['name']:<45} [{n['label']:<18}] {n['node_id']}")
        sys.exit(0)

    if nodes_q:
        for n in search_nodes(nodes_q, node_label):
            print(f"  {n['name']:<45} [{n['label']:<18}] score={n['score']:.3f}")
        sys.exit(0)

    if query:
        results = retrieve_for_llm(query, top_k)

        print(f"\n{'=' * 65}")
        print(f"QUERY  : {query}")
        print(f"RESULTS: {len(results)}")
        print(f"{'=' * 65}")

        if not results:
            print("\n⚠️  NO_ANSWER — no relevant chunks found above threshold.\n")
            sys.exit(0)

        for rank, r in enumerate(results, 1):
            lvl_tag  = f" [L{r['node_level']}]" if r.get("node_level", -1) >= 0 else ""
            src_icon = "📄" if r.get("source_type") == "pdf" else "🔗"
            print(f"\n{rank}.  {r['chunk_id']}{lvl_tag}")
            print(f"   Score    : {r['score']:.4f}  "
                  f"(sem={r['sem_score']:.3f}  bm25={r['bm25_score']:.3f}  "
                  f"fused={r['fused_score']:.3f}  rerank={r['rerank_cal']:.3f}  "
                  f"boost={r['graph_boost']:.3f})")
            print(f"   Title    : {(r['title'] or 'N/A')[:80]}")
            print(f"   {src_icon} Source : {r.get('url') or 'N/A'}")
            print(f"   Class    : {r.get('classification_name','?')}  [{r.get('classification_label','?')}]")
            print(f"   Language : {r.get('language','?')}  |  Type: {r.get('source_type') or 'page'}")
            print(f"   Preview  : {r['text'][:220]}…")

        if results[0].get("llm_context"):
            print(f"\n{'═' * 65}")
            print("LLM CONTEXT PREVIEW (first 1400 chars)")
            print(f"{'═' * 65}")
            print(results[0]["llm_context"][:1400])
    else:
        print("No query provided.")
        sys.exit(1)