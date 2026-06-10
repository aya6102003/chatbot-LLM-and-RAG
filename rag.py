from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import CrossEncoder

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
# CONFIG
# ─────────────────────────────────────────────────────────────
NEO4J_URI           = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER          = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD      = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_TIMEOUT       = int(os.getenv("NEO4J_TIMEOUT", "30"))
NEO4J_VECTOR_INDEX  = "chunk_embedding"

EMBED_MODEL_NAME  = "gemini-embedding-001"
VECTOR_DIMENSIONS = 3072
RERANK_MODEL      = "BAAI/bge-reranker-v2-m3"

TOP_K_VECTOR  = 60     # wider net for soft-boost approach
TOP_K_BM25    = 30
TOP_K_RERANK  = 20
TOP_K_FINAL   = 8

W_SEM    = 0.65
W_BM25   = 0.15
W_FUSED  = 0.45
W_RERANK = 0.55

RERANK_POWER   = 0.75
RERANK_MIN_CAL = 0.20   # lower floor — reranker decides, not hard filter

DYN_TOP_K_MIN        = 3
DYN_TOP_K_MAX        = 8
DYNAMIC_SCORE_MARGIN = 0.28

# ── Language boost (stronger than v15) ───────────────────────
LANG_EXACT_BOOST  = 0.10   # same language as query
LANG_MISMATCH_PEN = 0.05   # penalise opposite language

# ── Graph boost — soft scoring by depth ──────────────────────
GRAPH_BOOST_L0 = 0.25   # exact matched node
GRAPH_BOOST_L1 = 0.18   # direct children
GRAPH_BOOST_L2 = 0.12   # grandchildren
GRAPH_BOOST_L3 = 0.07   # deeper descendants
GRAPH_BOOST_GENERAL = 0.04  # GENERAL fallback layer
GRAPH_BOOST_SIBLING = 0.06  # sibling nodes (same parent)

# ── Answerability gates ───────────────────────────────────────
ANSWER_THRESHOLD_STRICT = 0.38   # normal queries
ANSWER_THRESHOLD_LOOSE  = 0.20   # person_lookup / node_query / short queries

DEDUP_CHARS = 200

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
    is_neighbor: bool = False
    sem_score:   float = 0.0
    bm25_score:  float = 0.0
    fused_score: float = 0.0
    rerank_raw:  float = 0.0
    rerank_cal:  float = 0.0
    graph_boost: float = 0.0
    node_level:  int   = -1   # -1 = not in graph, 0-3 = depth level


@dataclass
class AcademicNode:
    node_id:      str
    name:         str
    label:        str
    parent_id:    Optional[str]        = None
    children_ids: List[str]            = field(default_factory=list)
    aliases:      List[str]            = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# NEO4J CONNECTION — singleton with safe teardown
# ══════════════════════════════════════════════════════════════

_neo4j_driver = None
_neo4j_lock = threading.Lock()


def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        with _neo4j_lock:
            if _neo4j_driver is None:
                _neo4j_driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                    connection_timeout=NEO4J_TIMEOUT,
                )
                with _neo4j_driver.session() as sess:
                    sess.run("RETURN 1")
                log.info("✓ Neo4j connected")
    return _neo4j_driver


def _close_driver():
    global _neo4j_driver
    if _neo4j_driver:
        _neo4j_driver.close()
        _neo4j_driver = None


atexit.register(_close_driver)


# ══════════════════════════════════════════════════════════════
# QUERY ANALYZER  — language + intent + keywords
# ══════════════════════════════════════════════════════════════

class QueryAnalyzer:
    _RE_AR = re.compile(r"[\u0600-\u06FF]")
    _RE_FR = re.compile(
        r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
        r"comment|quoi|qui|quel|quelle|où|quand|pourquoi)\b",
        re.IGNORECASE,
    )

    # Intent detection — ordered by specificity
    _INTENT_RULES: List[Tuple[str, List[str]]] = [
        ("graph_query",    ["hiérarchie","hierarchy","arbre","tree","structure","organigramme","graph","schéma","هيكل","شجرة"]),
        ("person_lookup",  ["prof","dr ","docteur","enseignant","teacher","professeur","مدرس","أستاذ","دكتور","email","contact","responsable"]),
        ("course_query",   ["cours","course","module","matière","td","tp","semestre","semester","programme","مقياس","مادة","برنامج"]),
        ("admin_query",    ["inscription","registration","examen","exam","résultat","result","calendrier","emploi du temps","تسجيل","امتحان","نتيجة"]),
        ("node_query",     ["spécialité","specialization","filière","département","department","faculté","faculty","تخصص","قسم","كلية","شعبة"]),
        ("general_info",   []),  # fallback
    ]

    def analyze(self, query: str) -> Dict:
        lang     = self._detect_language(query)
        intent   = self._detect_intent(query)
        keywords = self._extract_keywords(query)
        return {
            "query":      query,
            "language":   lang,
            "intent":     intent,
            "keywords":   keywords,
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
        for intent, keywords in self._INTENT_RULES:
            if any(kw in q for kw in keywords):
                return intent
        return "general_info"

    def _extract_keywords(self, query: str) -> List[str]:
        normalized = self._normalize(query)
        tokens     = normalized.split()
        keywords   = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
        return keywords if keywords else tokens

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text



# ══════════════════════════════════════════════════════════════
# QUERY ROUTER — decides retrieval strategy before search
# ══════════════════════════════════════════════════════════════

@dataclass
class RetrievalStrategy:
    mode:            str            # "graph_first" | "graph_only" | "hybrid" | "semantic_first" | "wide_search"
    use_soft_boost:  bool   = True  # always true except graph_only
    expand_siblings: bool   = False
    expand_general:  bool   = True
    graph_depth_max: int    = 3
    top_k_multiplier: float = 1.0
    min_score_override: Optional[float] = None


class QueryRouter:
    """
    Maps intent → retrieval strategy.
    This runs BEFORE any retrieval so the pipeline knows how to proceed.
    """

    _STRATEGY_MAP: Dict[str, RetrievalStrategy] = {
        "graph_query":   RetrievalStrategy(mode="graph_only",    use_soft_boost=False, graph_depth_max=6),
        "node_query":    RetrievalStrategy(mode="graph_first",   use_soft_boost=True,  graph_depth_max=4, expand_siblings=True),
        "course_query":  RetrievalStrategy(mode="hybrid",        use_soft_boost=True,  graph_depth_max=3, top_k_multiplier=1.5),
        "admin_query":   RetrievalStrategy(mode="hybrid",        use_soft_boost=True,  graph_depth_max=2),
        "person_lookup": RetrievalStrategy(mode="wide_search",   use_soft_boost=True,  graph_depth_max=1,
                                           expand_siblings=True, min_score_override=ANSWER_THRESHOLD_LOOSE),
        "general_info":  RetrievalStrategy(mode="semantic_first",use_soft_boost=True,  graph_depth_max=2),
    }

    def route(self, analysis: Dict) -> RetrievalStrategy:
        intent   = analysis.get("intent", "general_info")
        strategy = self._STRATEGY_MAP.get(intent, self._STRATEGY_MAP["general_info"])
        log.info(f"   🗺  Router: intent={intent}  mode={strategy.mode}  depth={strategy.graph_depth_max}")
        return strategy


# ══════════════════════════════════════════════════════════════
# ACADEMIC GRAPH — loads tree, detects entities, builds boosts
# ══════════════════════════════════════════════════════════════

class AcademicGraph:
    """
    Full academic tree from Neo4j.
    Provides:
      - Entity detection in queries (multi-strategy)
      - Soft boost map by BFS depth
      - GENERAL node as explicit fallback layer
      - Sibling expansion
      - graph_query → direct hierarchy traversal
    """

    def __init__(self):
        self.nodes:           Dict[str, AcademicNode] = {}
        self.children_map:    Dict[str, List[str]]    = {}
        self.parent_map:      Dict[str, str]          = {}
        self.node_to_urls:    Dict[str, List[str]]    = {}
        self.url_to_chunks:   Dict[str, List[str]]    = {}
        self.name_index:      Dict[str, List[str]]    = {}  # lower name → [node_ids]
        self.general_ids:     List[str]               = []
        self._loaded          = False
        self._load_graph()

    # ── Loading ───────────────────────────────────────────────

    def _load_graph(self):
        try:
            driver = _get_driver()
            with driver.session() as sess:
                self._load_nodes(sess)
                self._load_url_mappings(sess)
                self._load_chunk_mappings(sess)
            self._loaded = True
            log.info(
                f"✓ Academic graph loaded: {len(self.nodes)} nodes  "
                f"{len(self.url_to_chunks)} URLs  "
                f"GENERAL={self.general_ids}"
            )
        except Exception as e:
            log.error(f"Failed to load academic graph: {e}")
            self._loaded = False

    def _load_nodes(self, sess):
        """Load all hierarchy nodes and build indexes."""
        result = sess.run("""
            MATCH (n)
            WHERE n:Faculty OR n:Department OR n:Program
               OR n:Specialization OR n:General OR n:Level
               OR n:Category OR n:Year
            RETURN
                n.id      AS id,
                n.name    AS name,
                labels(n) AS lbls,
                n.aliases AS aliases
        """)
        for rec in result:
            nid  = rec["id"]
            name = rec["name"] or nid
            if not nid:
                continue

            label = "General"
            priority = {"Faculty":0,"Department":1,"Level":2,"Category":3,
                        "Program":4,"Specialization":5,"Year":6,"General":7}
            best_p = 99
            for lbl in (rec["lbls"] or []):
                p = priority.get(lbl, 99)
                if p < best_p:
                    best_p = p
                    label = lbl

            # توليد aliases الأساسية
            generated_aliases = self._generate_aliases(name, label)
            
            # ✨ إضافة aliases من Neo4j
            neo4j_aliases = rec.get("aliases")
            if neo4j_aliases:
                for a in neo4j_aliases:
                    a_clean = a.lower().strip()
                    if a_clean and a_clean not in generated_aliases:
                        generated_aliases.append(a_clean)
            
            node = AcademicNode(
                node_id=nid,
                name=name,
                label=label,
                aliases=generated_aliases,
            )
            self.nodes[nid]       = node
            self.children_map[nid] = []

            if label == "General":
                self.general_ids.append(nid)

            # Index by canonical name
            self._index_name(name.lower().strip(), nid)
            # Index ALL aliases
            for alias in node.aliases:
                self._index_name(alias.lower().strip(), nid)

        # Load parent→child relationships
        rel_result = sess.run("""
            MATCH (parent)-[r]->(child)
            WHERE type(r) IN ['HAS_DEPARTMENT','HAS_LEVEL','HAS_PROGRAM',
                              'HAS_CATEGORY','HAS_SPECIALIZATION','HAS_YEAR',
                              'HAS_CHILD']
              AND parent.id IS NOT NULL AND child.id IS NOT NULL
            RETURN parent.id AS pid, child.id AS cid
        """)
        for rec in rel_result:
            pid = rec["pid"]
            cid = rec["cid"]
            if pid and cid:
                if pid not in self.children_map:
                    self.children_map[pid] = []
                if cid not in self.children_map[pid]:
                    self.children_map[pid].append(cid)
                self.parent_map[cid] = pid
                if cid in self.nodes:
                    self.nodes[cid].parent_id = pid


    def _load_url_mappings(self, sess):
        """Load node → URL mappings using CLASSIFIED_AS relationship."""
        # Try CLASSIFIED_AS first (direction: URL → Node)
        result = sess.run("""
            MATCH (u:URL)-[:CLASSIFIED_AS]->(n)
            WHERE n.id IS NOT NULL AND u.id IS NOT NULL
            RETURN n.id AS nid, u.id AS uid
        """)
        
        records = list(result)
        
        # Fallback: if CLASSIFIED_AS returns nothing, try HAS_CONTENT (legacy)
        if not records:
            log.warning("   ⚠️ No CLASSIFIED_AS relationships found, trying HAS_CONTENT...")
            result = sess.run("""
                MATCH (n)-[:HAS_CONTENT]->(u:URL)
                WHERE n.id IS NOT NULL AND u.id IS NOT NULL
                RETURN n.id AS nid, u.id AS uid
            """)
            records = list(result)
        
        for rec in records:
            nid = rec["nid"]
            uid = rec["uid"]
            if nid and uid:
                if nid not in self.node_to_urls:
                    self.node_to_urls[nid] = []
                self.node_to_urls[nid].append(uid)
        
        log.info(f"   Loaded {len(records)} node→URL mappings")
        def _load_chunk_mappings(self, sess):
            """Load URL → chunk IDs."""
            result = sess.run("""
                MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk)
                WHERE u.id IS NOT NULL AND c.id IS NOT NULL
                RETURN u.id AS uid, collect(c.id) AS cids
            """)
            for rec in result:
                self.url_to_chunks[rec["uid"]] = rec["cids"] or []


    def _load_chunk_mappings(self, sess):
        """Load URL → chunk IDs."""
        result = sess.run("""
            MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk)
            WHERE u.id IS NOT NULL AND c.id IS NOT NULL
            RETURN u.id AS uid, collect(c.id) AS cids
        """)
        for rec in result:
            self.url_to_chunks[rec["uid"]] = rec["cids"] or []

    def _index_name(self, name: str, nid: str):
        if not name:
            return
        if name not in self.name_index:
            self.name_index[name] = []
        if nid not in self.name_index[name]:
            self.name_index[name].append(nid)

    # ── Alias generation ──────────────────────────────────────

    _ABBR_MAP: Dict[str, List[str]] = {
        "informatique":                            ["info","isi","cs","gl","génie logiciel","rsd"],
        "informatique industrielle":               ["isi","ii"],
        "mathématiques":                           ["maths","math","mi"],
        "mathématiques et informatique":           ["mi","maths info","math info"],
        "physique":                                ["phys","sm"],
        "chimie":                                  ["chim","sm"],
        "biologie":                                ["bio","sv","svt","snv"],
        "sciences de la matière":                  ["sm","physique chimie"],
        "sciences et technologie":                 ["st"],
        "sciences de la vie et de la nature":      ["svt","snv","sv"],
        "sciences de la nature et de la vie":      ["snv","svt","sv"],
        "génie logiciel":                          ["gl","software engineering"],
        "réseaux et systèmes distribués":          ["rsd","networks","réseaux"],
        "faculté des sciences":                    ["fs","fac science"],
        "faculté de technologie":                  ["ft","fac technologie"],
        "faculté de médecine":                     ["fm","fac médecine"],
        "licence":                                 ["l1","l2","l3","bachelor"],
        "master":                                  ["m1","m2"],
        "doctorat":                                ["phd","doctorate","doc"],
        "ingénieur":                               ["ing","ingenieur","ing1","ing2","ing3"],
        "general":                                 ["général","générale","commun"],
    }

    def _generate_aliases(self, name: str, label: str) -> List[str]:
        name_lower = name.lower().strip()
        aliases: Set[str] = set()

        # Direct lookup
        if name_lower in self._ABBR_MAP:
            aliases.update(self._ABBR_MAP[name_lower])

        # Partial matches
        for key, vals in self._ABBR_MAP.items():
            if key != name_lower and (key in name_lower or name_lower in key):
                aliases.update(vals)

        # Acronym
        words = name_lower.split()
        if len(words) > 1:
            acronym = "".join(w[0] for w in words if w)
            if len(acronym) >= 2:
                aliases.add(acronym)

        # Remove self
        aliases.discard(name_lower)
        return list(aliases)

    # ── Entity detection ──────────────────────────────────────

    def detect_nodes_in_query(self, query: str, strategy: Optional[RetrievalStrategy] = None) -> List[AcademicNode]:
        """
        Multi-strategy entity detection.
        Returns matched nodes ordered by confidence.
        Does NOT apply hard filtering — soft boost map is built later.
        """
        if not self._loaded:
            return []

        query_lower = query.lower().strip()
        scores: Dict[str, float] = {}  # node_id → score

        # Strategy 1: Exact substring match (highest confidence)
        for name_lower, node_ids in self.name_index.items():
            if len(name_lower) < 3:
                continue
            if name_lower in query_lower:
                coverage = len(name_lower) / max(len(query_lower), 1)
                s = min(2.0, coverage * 3.0)
                for nid in node_ids:
                    scores[nid] = max(scores.get(nid, 0.0), s)

        # Strategy 2: Word overlap
        query_words = set(query_lower.split()) - _STOPWORDS
        for name_lower, node_ids in self.name_index.items():
            name_words = set(name_lower.split()) - _STOPWORDS
            if not name_words:
                continue
            overlap = query_words & name_words
            if overlap:
                s = len(overlap) / len(name_words) * 0.8
                for nid in node_ids:
                    scores[nid] = max(scores.get(nid, 0.0), s)

        # Strategy 3: Fuzzy (for short queries / typos)
        if _FUZZ_OK and len(query_lower.split()) <= 4:
            for name_lower, node_ids in self.name_index.items():
                if len(name_lower) < 4:
                    continue
                ratio = _fuzz.partial_ratio(query_lower, name_lower)
                if ratio >= 78:
                    s = (ratio / 100.0) * 0.65
                    for nid in node_ids:
                        scores[nid] = max(scores.get(nid, 0.0), s)

        # Filter by threshold, sort by score
        DETECTION_THRESHOLD = 0.35
        sorted_matches = sorted(
            [(nid, sc) for nid, sc in scores.items() if sc >= DETECTION_THRESHOLD],
            key=lambda x: x[1],
            reverse=True,
        )

        # Don't return GENERAL nodes from detection — they are fallback layer
        result: List[AcademicNode] = []
        for nid, sc in sorted_matches[:6]:
            node = self.nodes.get(nid)
            if node and node.label != "General":
                result.append(node)
                log.info(f"   🎯 Detected: '{node.name}' [{node.label}] score={sc:.2f}")

        return result

    # ── Soft boost map construction ───────────────────────────

    def build_boost_map(
        self,
        matched_nodes: List[AcademicNode],
        strategy: RetrievalStrategy,
    ) -> Dict[str, Tuple[float, int]]:
        """
        Returns {chunk_id: (boost_value, depth_level)}.
        Soft scoring: ALL chunks remain candidates; matched ones get boosted.
        """
        boost_map: Dict[str, Tuple[float, int]] = {}

        if not matched_nodes:
            # No match → add GENERAL boost as soft signal
            return self._boost_general_nodes(boost_map)

        depth_boosts = {
            0: GRAPH_BOOST_L0,
            1: GRAPH_BOOST_L1,
            2: GRAPH_BOOST_L2,
            3: GRAPH_BOOST_L3,
        }

        for node in matched_nodes:
            nid = node.node_id
            # BFS with depth tracking
            queue:   deque = deque([(nid, 0)])
            visited: Set[str] = set()

            while queue:
                current_id, depth = queue.popleft()
                if current_id in visited or depth > strategy.graph_depth_max:
                    continue
                visited.add(current_id)

                boost = depth_boosts.get(depth, GRAPH_BOOST_L3)

                # Apply boost to all chunks under this node
                for url_id in self.node_to_urls.get(current_id, []):
                    for cid in self.url_to_chunks.get(url_id, []):
                        existing_boost, existing_level = boost_map.get(cid, (0.0, 99))
                        if boost > existing_boost:
                            boost_map[cid] = (boost, depth)

                # Enqueue children
                for child_id in self.children_map.get(current_id, []):
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))

            # Sibling expansion (optional per strategy)
            if strategy.expand_siblings:
                parent_id = self.parent_map.get(nid)
                if parent_id:
                    for sibling_id in self.children_map.get(parent_id, []):
                        if sibling_id == nid or sibling_id in visited:
                            continue
                        for url_id in self.node_to_urls.get(sibling_id, []):
                            for cid in self.url_to_chunks.get(url_id, []):
                                existing_boost, _ = boost_map.get(cid, (0.0, 99))
                                if GRAPH_BOOST_SIBLING > existing_boost:
                                    boost_map[cid] = (GRAPH_BOOST_SIBLING, 99)

        # Always add GENERAL as background signal
        if strategy.expand_general:
            boost_map = self._boost_general_nodes(boost_map)

        log.info(
            f"   📈 Boost map: {len(boost_map)} chunks  "
            f"L0={sum(1 for _,l in boost_map.values() if l==0)}  "
            f"L1={sum(1 for _,l in boost_map.values() if l==1)}  "
            f"L2={sum(1 for _,l in boost_map.values() if l==2)}  "
            f"L3+={sum(1 for _,l in boost_map.values() if l>=3)}"
        )
        return boost_map

    def _boost_general_nodes(self, boost_map: Dict[str, Tuple[float, int]]) -> Dict[str, Tuple[float, int]]:
        for gid in self.general_ids:
            for url_id in self.node_to_urls.get(gid, []):
                for cid in self.url_to_chunks.get(url_id, []):
                    existing_boost, _ = boost_map.get(cid, (0.0, 99))
                    if GRAPH_BOOST_GENERAL > existing_boost:
                        boost_map[cid] = (GRAPH_BOOST_GENERAL, 99)
        return boost_map

    def get_all_chunks_for_nodes(
        self,
        node_ids: List[str],
        max_depth: int = 4,
    ) -> Set[str]:
        """Collect ALL chunk IDs under given nodes up to max_depth."""
        all_chunks: Set[str] = set()
        visited:    Set[str] = set()
        queue: deque = deque([(nid, 0) for nid in node_ids])

        while queue:
            current, depth = queue.popleft()
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            for url_id in self.node_to_urls.get(current, []):
                all_chunks.update(self.url_to_chunks.get(url_id, []))

            for child_id in self.children_map.get(current, []):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        return all_chunks

    # ── graph_query mode: direct hierarchy traversal ──────────

    def traverse_for_graph_query(self, query: str) -> Dict:
        """
        For queries about structure/hierarchy: directly traverse Neo4j.
        Returns hierarchy tree without semantic search.
        """
        matched = self.detect_nodes_in_query(query)
        if not matched:
            # Return full faculty tree
            return self._get_faculty_tree()

        results = []
        for node in matched[:3]:
            results.append({
                "node":     {"id": node.node_id, "name": node.name, "label": node.label},
                "path":     self._get_node_path(node.node_id),
                "children": self._get_children(node.node_id),
            })
        return {"graph_results": results, "query": query}

    def _get_node_path(self, node_id: str) -> List[Dict]:
        path, current = [], node_id
        depth = 0
        while current and depth < 10:
            node = self.nodes.get(current)
            if not node:
                break
            path.append({"id": node.node_id, "name": node.name, "label": node.label})
            current = self.parent_map.get(current)
            depth += 1
        return list(reversed(path))

    def _get_children(self, node_id: str) -> List[Dict]:
        result = []
        for cid in self.children_map.get(node_id, []):
            node = self.nodes.get(cid)
            if node:
                grandchildren = [
                    {"id": gc, "name": self.nodes[gc].name, "label": self.nodes[gc].label}
                    for gc in self.children_map.get(cid, [])
                    if gc in self.nodes
                ]
                result.append({
                    "id":       node.node_id,
                    "name":     node.name,
                    "label":    node.label,
                    "children": grandchildren,
                })
        return result

    def _get_faculty_tree(self) -> Dict:
        roots = [n for n in self.nodes.values() if n.label == "Faculty"]
        tree = []
        for root in roots:
            tree.append({
                "node":     {"id": root.node_id, "name": root.name, "label": root.label},
                "children": self._get_children(root.node_id),
            })
        return {"graph_results": tree, "query": "full_faculty_tree"}

    def get_node_info(self, node_id: str) -> Optional[Dict]:
        node = self.nodes.get(node_id)
        if not node:
            return None
        direct_chunks = sum(
            len(self.url_to_chunks.get(uid, []))
            for uid in self.node_to_urls.get(node_id, [])
        )
        total_chunks = len(self.get_all_chunks_for_nodes([node_id]))
        return {
            "node_id":       node.node_id,
            "name":          node.name,
            "label":         node.label,
            "parent_id":     node.parent_id,
            "path":          self._get_node_path(node_id),
            "children":      self._get_children(node_id),
            "direct_chunks": direct_chunks,
            "total_chunks":  total_chunks,
        }


# ══════════════════════════════════════════════════════════════
# SEMANTIC RETRIEVER — global (no hard filter)
# ══════════════════════════════════════════════════════════════

class SemanticRetriever:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set! Run: export GEMINI_API_KEY='...'")
        
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model_name = "models/gemini-embedding-001"
        self._cache: Dict[str, np.ndarray] = {}
        log.info("✓ Gemini embeddings ready (3072-dim)")

    def _encode_gemini(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Gemini API with caching."""
        results = []
        miss_idx, miss_texts = [], []
        
        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
            else:
                results.append(None)
                miss_idx.append(i)
                miss_texts.append(t)
        
        if miss_texts:
            for local_i, global_i in enumerate(miss_idx):
                text = miss_texts[local_i]
                try:
                    response = self._client.models.embed_content(
                        model=self._model_name,
                        contents=text
                    )
                    vec = np.array(response.embeddings[0].values, dtype=np.float32)
                    vec = vec / np.linalg.norm(vec)  # normalize
                    
                    if len(self._cache) >= EMBED_CACHE_SIZE:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[text] = vec
                    results[global_i] = vec
                except Exception as e:
                    log.error(f"Gemini encoding failed: {e}")
                    results[global_i] = np.zeros(VECTOR_DIMENSIONS, dtype=np.float32)
        
        valid = [r for r in results if r is not None]
        return np.vstack(valid) if valid else np.empty((0, VECTOR_DIMENSIONS))

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode_gemini(texts)

    def encode_passage(self, texts: List[str]) -> np.ndarray:
        return self._encode_gemini(texts)

    def search(
        self,
        variants: List[str],
        top_k: int,
    ) -> Dict[str, RetrievedChunk]:
        if not variants:
            return {}

        embeddings = self.encode(variants)
        if embeddings.shape[0] == 0:
            return {}

        candidates: Dict[str, RetrievedChunk] = {}
        driver = _get_driver()

        for vec in embeddings:
            cypher = """
                CALL db.index.vector.queryNodes($index, $k, $vec)
                YIELD node AS c, score
                MATCH (u:URL)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                RETURN
                    c.id           AS cid,
                    c.text         AS text,
                    c.chunk_index  AS ci,
                    c.language     AS lang,
                    score          AS sim,
                    u.url          AS url,
                    u.title        AS title,
                    u.source_type  AS st,
                    page.url       AS page_url,
                    u.id           AS url_id
            """
            try:
                with driver.session() as sess:
                    for rec in sess.run(cypher, {
                        "index": NEO4J_VECTOR_INDEX,
                        "k": top_k,
                        "vec": vec.tolist()
                    }):
                        cid = rec["cid"]
                        sim = float(rec["sim"])
                        if cid and (cid not in candidates or sim > candidates[cid].sem_score):
                            candidates[cid] = RetrievedChunk(
                                chunk_id=cid,
                                text=rec["text"] or "",
                                score=sim,
                                metadata={
                                    "title":       rec["title"]    or "",
                                    "url":         rec["url"]      or "",
                                    "page_url":    rec["page_url"] or rec["url"] or "",
                                    "source_type": rec["st"]       or "",
                                    "language":    rec["lang"]     or "",
                                    "chunk_index": rec["ci"],
                                    "url_id":      rec["url_id"]   or "",
                                    "pdf_url":     rec["url"] if rec["st"] == "pdf" else "",
                                },
                                sem_score=sim,
                            )
            except Exception as e:
                log.warning(f"Vector search error: {e}")

        return candidates
    

# ══════════════════════════════════════════════════════════════
# BM25 RETRIEVER — global index, no hard filter
# ══════════════════════════════════════════════════════════════

class BM25Retriever:
    def __init__(self):
        self._chunk_ids: List[str]       = []
        self._texts:     List[str]       = []
        self._meta:      Dict[str, Dict] = {}
        self._bm25 = None

        if not _BM25_OK:
            return

        log.info("Building BM25 index...")
        qa        = QueryAnalyzer()
        tokenized: List[List[str]] = []

        try:
            driver = _get_driver()
            with driver.session() as sess:
                for rec in sess.run("""
                    MATCH (u:URL)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (page:URL)-[:HAS_FILE]->(u)
                    RETURN
                        c.id           AS cid,
                        c.text         AS text,
                        c.language     AS lang,
                        c.chunk_index  AS ci,
                        u.url          AS url,
                        u.title        AS title,
                        u.source_type  AS st,
                        page.url       AS page_url,
                        u.id           AS url_id
                """):
                    cid  = rec["cid"]
                    text = rec["text"] or ""
                    if not cid or not text.strip():
                        continue
                    self._chunk_ids.append(cid)
                    self._texts.append(text)
                    self._meta[cid] = {
                        "title":       rec["title"]    or "",
                        "url":         rec["url"]      or "",
                        "page_url":    rec["page_url"] or rec["url"] or "",
                        "source_type": rec["st"]       or "",
                        "language":    rec["lang"]     or "fr",
                        "chunk_index": rec["ci"],
                        "url_id":      rec["url_id"]   or "",
                        "pdf_url":     rec["url"] if rec["st"] == "pdf" else "",
                    }
                    tokenized.append(qa._normalize(text).split())
        except Exception as e:
            log.error(f"BM25 load failed: {e}")
            return

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
            log.info(f"BM25 ready: {len(tokenized)} chunks")

    def search(
        self,
        keywords: List[str],
        top_k: int,
    ) -> List[Tuple[str, str, float]]:
        """Always searches full corpus (no hard filter)."""
        if self._bm25 is None or not keywords:
            return []

        raw = self._bm25.get_scores(keywords)
        mx  = raw.max()
        if mx <= 0:
            return []

        normed = raw / mx
        results: List[Tuple[str, str, float]] = []

        for i in np.argsort(normed)[::-1][: top_k * 2]:
            if normed[i] <= 0:
                continue
            cid = self._chunk_ids[i]
            results.append((cid, self._texts[i], float(normed[i])))
            if len(results) >= top_k:
                break

        return results

    def get_meta(self, cid: str) -> Optional[Dict]:
        return self._meta.get(cid)


# ══════════════════════════════════════════════════════════════
# FUSION — hybrid scoring with language-aware boost
# ══════════════════════════════════════════════════════════════

def _fingerprint(text: str) -> str:
    return hashlib.md5(
        re.sub(r"\s+", " ", text[:DEDUP_CHARS]).strip().lower().encode()
    ).hexdigest()


def fuse_results(
    semantic:    Dict[str, RetrievedChunk],
    bm25:        List[Tuple[str, str, float]],
    boost_map:   Dict[str, Tuple[float, int]],
    query_lang:  str,
    bm25_retriever: Optional[BM25Retriever] = None,
) -> List[RetrievedChunk]:
    """
    Merge semantic + BM25, apply:
      - Graph soft boost
      - Language match boost / mismatch penalty
    No hard filtering.
    """
    pool: Dict[str, Dict[str, Any]] = {}

    for cid, chunk in semantic.items():
        pool[cid] = {
            "sem":  chunk.sem_score,
            "bm25": 0.0,
            "text": chunk.text,
            "meta": chunk.metadata,
        }

    for cid, text, score in bm25:
        if cid not in pool:
            meta = (bm25_retriever.get_meta(cid) or {}) if bm25_retriever else {}
            pool[cid] = {"sem": 0.0, "bm25": score, "text": text, "meta": meta}
        else:
            pool[cid]["bm25"] = max(pool[cid]["bm25"], score)

    fused:  List[RetrievedChunk] = []
    seen_fp: Dict[str, float]   = {}

    for cid, d in pool.items():
        sem = float(d["sem"])
        bm  = float(d["bm25"])
        if sem == 0.0 and bm == 0.0:
            continue

        score = W_SEM * sem + W_BM25 * bm

        # Language-aware scoring: REWARD only, NO PENALTY
        meta = d["meta"] if isinstance(d["meta"], dict) else {}
        chunk_lang = meta.get("language", "")
        if chunk_lang and chunk_lang == query_lang:
            score += LANG_EXACT_BOOST

        # Soft graph boost
        gb, level = boost_map.get(cid, (0.0, -1))
        score += gb

        fp = _fingerprint(d["text"])
        if fp in seen_fp and score <= seen_fp[fp]:
            continue
        seen_fp[fp] = score

        fused.append(RetrievedChunk(
            chunk_id=cid,
            text=d["text"],
            score=score,
            metadata=meta,
            sem_score=sem,
            bm25_score=bm,
            fused_score=score,
            graph_boost=gb,
            node_level=level,
        ))

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


# ══════════════════════════════════════════════════════════════
# RERANKER — language-consistent cross-encoder
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

            # ⚠️ لا تعاقب عدم تطابق اللغة - LaBSE/Gemini يتعامل مع هذا
            # فقط كافئ تطابق اللغة بشكل طفيف
            chunk_lang = chunk.metadata.get("language", "")
            if query_lang and chunk_lang == query_lang:
                cal = min(1.0, cal + 0.03)   # مكافأة صغيرة فقط

            chunk.rerank_raw = raw
            chunk.rerank_cal = cal
            chunk.score      = W_FUSED * chunk.fused_score + W_RERANK * cal

        kept = sorted(
            [c for c in chunks if c.rerank_cal >= min_cal],
            key=lambda c: c.score,
            reverse=True,
        )
        return kept[:top_k]

# ══════════════════════════════════════════════════════════════
# FALLBACK MANAGER — staged fallback strategy
# ══════════════════════════════════════════════════════════════

class FallbackManager:
    """
    Implements multi-stage fallback:
      Stage 1: detected nodes (already done in main pipeline)
      Stage 2: GENERAL nodes only
      Stage 3: global corpus (no boost, pure semantic+BM25)
    """

    def __init__(
        self,
        semantic:   SemanticRetriever,
        bm25:       BM25Retriever,
        reranker:   Reranker,
        graph:      AcademicGraph,
    ):
        self.semantic  = semantic
        self.bm25      = bm25
        self.reranker  = reranker
        self.graph     = graph

    def fallback_general(
        self,
        query:    str,
        keywords: List[str],
        lang:     str,
        top_k:    int,
    ) -> List[RetrievedChunk]:
        """Stage 2: search only within GENERAL nodes."""
        log.info("   🔄 Fallback Stage 2: GENERAL nodes")
        boost_map = self.graph._boost_general_nodes({})
        if not boost_map:
            return []

        variants     = [query]
        semantic_hits = self.semantic.search(variants, TOP_K_VECTOR)
        bm25_hits     = self.bm25.search(keywords, TOP_K_BM25)
        fused         = fuse_results(semantic_hits, bm25_hits, boost_map, lang, self.bm25)

        # Keep only chunks that have any GENERAL boost
        general_cids = set(boost_map.keys())
        fused = [c for c in fused if c.chunk_id in general_cids]
        return fused

    def fallback_global(
        self,
        query:    str,
        keywords: List[str],
        lang:     str,
        top_k:    int,
    ) -> List[RetrievedChunk]:
        """Stage 3: full corpus retrieval, no graph boost."""
        log.info("   🔄 Fallback Stage 3: Global corpus")
        variants     = [query]
        semantic_hits = self.semantic.search(variants, TOP_K_VECTOR)
        bm25_hits     = self.bm25.search(keywords, TOP_K_BM25)
        fused         = fuse_results(semantic_hits, bm25_hits, {}, lang, self.bm25)
        return fused


# ══════════════════════════════════════════════════════════════
# MAIN RAG RETRIEVER — orchestrates everything
# ══════════════════════════════════════════════════════════════

class RAGRetriever:

    def __init__(self):
        log.info("═" * 55)
        log.info("Initializing RAGRetriever v16.0 — Graph-Aware Production")
        log.info("═" * 55)

        self.analyzer  = QueryAnalyzer()
        self.router    = QueryRouter()
        self.graph     = AcademicGraph()
        self.semantic  = SemanticRetriever()
        self.bm25      = BM25Retriever()
        
        self.reranker  = Reranker()
        self.fallback  = FallbackManager(self.semantic, self.bm25, self.reranker, self.graph)

        log.info("✅ RAGRetriever v16.0 ready")

    # ── Public entry point ────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> List[RetrievedChunk]:
        log.info(f"\n🔍 QUERY: {query[:120]}")

        # ── Step 1: Analyse query ─────────────────────────────
        analysis = self.analyzer.analyze(query)
        lang     = analysis["language"]
        intent   = analysis["intent"]
        keywords = analysis["keywords"]
        log.info(f"   Language={lang}  Intent={intent}  Keywords={keywords[:6]}")

        # ── Step 2: Route to strategy ─────────────────────────
        strategy = self.router.route(analysis)

        # ── Step 3: graph_query → direct hierarchy, no vectors ─
        if strategy.mode == "graph_only":
            log.info("   📊 graph_only mode → direct Neo4j traversal")
            result = self.graph.traverse_for_graph_query(query)
            # Wrap into RetrievedChunk for uniform output
            if result.get("graph_results"):
                text = json.dumps(result, ensure_ascii=False, indent=2)
                return [RetrievedChunk(
                    chunk_id="graph_result",
                    text=text,
                    score=1.0,
                    metadata={"source_type": "graph_traversal", "url": "", "title": "Graph Structure"},
                    graph_boost=1.0,
                    node_level=0,
                )]
            return []

        # ── Step 4: Detect academic entities ─────────────────
        matched_nodes = self.graph.detect_nodes_in_query(query, strategy)

        # ── Step 5: Build SOFT boost map (no hard filter) ─────
        boost_map = self.graph.build_boost_map(matched_nodes, strategy)

        # ── Step 6: Retrieval ─────────────────────────────────
        variants     = self._build_variants(query, analysis)
        top_k_scaled = max(TOP_K_VECTOR, int(TOP_K_VECTOR * strategy.top_k_multiplier))

        semantic_hits = self.semantic.search(variants, top_k_scaled)
        bm25_hits     = self.bm25.search(keywords, max(TOP_K_BM25, int(TOP_K_BM25 * strategy.top_k_multiplier)))

        log.info(f"   Semantic={len(semantic_hits)}  BM25={len(bm25_hits)}")

        # ── Step 7: Fusion ────────────────────────────────────
        fused = fuse_results(semantic_hits, bm25_hits, boost_map, lang, self.bm25)

        if not fused:
            log.warning("   ⚠️ Fusion empty → Fallback Stage 2 (GENERAL)")
            fused = self.fallback.fallback_general(query, keywords, lang, top_k)

        if not fused:
            log.warning("   ⚠️ GENERAL fallback empty → Fallback Stage 3 (global)")
            fused = self.fallback.fallback_global(query, keywords, lang, top_k)

        if not fused:
            log.warning("   ❌ All fallbacks exhausted — NO_ANSWER")
            return []

        # ── Step 8: Rerank ────────────────────────────────────
        rerank_min = (
            0.12
            if intent in ("person_lookup", "node_query") or len(query.split()) <= 2
            else RERANK_MIN_CAL
        )
        
        if lang == "ar":
            rerank_weight = 0.35   # Arabic: trust embedding more than reranker
        else:
            rerank_weight = W_RERANK   # 0.55 for other languages
        
        reranked = self.reranker.rerank(
            query, fused[:TOP_K_RERANK], top_k=TOP_K_RERANK,
            min_cal=rerank_min, query_lang=lang,
        )
        
        if lang == "ar":
            for chunk in reranked:
                chunk.score = W_FUSED * chunk.fused_score + rerank_weight * chunk.rerank_cal
            reranked.sort(key=lambda c: c.score, reverse=True)

        if not reranked:
            log.warning("   ⚠️ Reranker dropped everything → using top fused")
            # Safe fallback: use fused pre-rerank with lower confidence
            reranked = fused[:DYN_TOP_K_MAX]
            for c in reranked:
                c.rerank_cal = 0.0

        # ── Step 9: Dynamic top-k selection ───────────────────
        if reranked:
            best  = reranked[0].score
            floor = best - DYNAMIC_SCORE_MARGIN
            final = [c for c in reranked if c.score >= floor][:DYN_TOP_K_MAX]
            if len(final) < DYN_TOP_K_MIN:
                final = reranked[:DYN_TOP_K_MIN]
        else:
            final = []

        # ── Step 10: Answerability gate ───────────────────────
        ans_threshold = (
            strategy.min_score_override
            if strategy.min_score_override is not None
            else (
                ANSWER_THRESHOLD_LOOSE
                if intent in ("person_lookup", "node_query") or len(query.split()) <= 2
                else ANSWER_THRESHOLD_STRICT
            )
        )

        if not final or final[0].score < ans_threshold:
            best_score = final[0].score if final else 0.0
            log.warning(f"   ❌ Answerability gate FAILED: best={best_score:.3f} < threshold={ans_threshold:.3f} → NO_ANSWER")
            return []

        log.info(
            f"   ✅ Final: {len(final)} chunks  "
            f"scores={[round(c.score, 3) for c in final]}"
        )
        return final

    # ── Helpers ───────────────────────────────────────────────

    def _build_variants(self, query: str, analysis: Dict) -> List[str]:
        """Build query variants for multi-vector search."""
        variants = [query]
        intent = analysis.get("intent", "")
        lang   = analysis.get("language", "en")

        # Add intent-aware variant
        if intent == "course_query":
            variants.append(f"cours module programme {query}")
        elif intent == "node_query":
            variants.append(f"département spécialité filière {query}")
        elif intent == "admin_query":
            variants.append(f"inscription administration {query}")

        # Add language variant for Arabic queries (translate key terms)
        if lang == "ar" and len(query.split()) <= 5:
            ar_to_fr = {
                "تخصص": "spécialité",
                "قسم":  "département",
                "كلية": "faculté",
                "مقياس": "cours",
                "ماستر": "master",
                "ليسانس": "licence",
            }
            translated = query
            for ar, fr in ar_to_fr.items():
                translated = translated.replace(ar, fr)
            if translated != query:
                variants.append(translated)

        return variants[:3]   # max 3 variants to keep latency low

    def search_nodes(self, query: str, node_type: Optional[str] = None) -> List[Dict]:
        nodes = self.graph.detect_nodes_in_query(query)
        if node_type:
            nodes = [n for n in nodes if n.label == node_type]
        return [
            {
                "node_id":   n.node_id,
                "name":      n.name,
                "label":     n.label,
                "parent_id": n.parent_id,
            }
            for n in nodes
        ]

    def get_node_info(self, node_id: str) -> Optional[Dict]:
        return self.graph.get_node_info(node_id)

    def close(self):
        _close_driver()


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
    Returns list of dicts with full source traceability on every chunk.
    First element contains 'llm_context' key with formatted text block.
    """
    chunks = _get_retriever().retrieve(query, top_k)
    if not chunks:
        return []

    # ✨ احسب أفضل score للـ chunks
    best_score = chunks[0].score if chunks else 0.0
    
    # ✨ حد أدنى للـ score النسبي: 70% من أفضل score
    MIN_RELATIVE_SCORE = 0.70
    min_score_threshold = best_score * MIN_RELATIVE_SCORE

    result = []
    seen_urls = set()
    
    for c in chunks:
        # ✨ تخطي الـ chunks الضعيفة جداً
        if c.score < min_score_threshold:
            continue
            
        meta = c.metadata or {}
        source_url = meta.get("page_url") or meta.get("url") or ""
        pdf_url    = meta.get("pdf_url") or ""

        # ✨ تجنب تكرار نفس المصدر
        canonical_url = source_url or pdf_url
        if canonical_url and canonical_url in seen_urls:
            continue
        if canonical_url:
            seen_urls.add(canonical_url)

        result.append({
            "chunk_id":    c.chunk_id,
            "text":        c.text,
            "score":       round(c.score,       4),
            "sem_score":   round(c.sem_score,   4),
            "bm25_score":  round(c.bm25_score,  4),
            "fused_score": round(c.fused_score, 4),
            "rerank_cal":  round(c.rerank_cal,  4),
            "graph_boost": round(c.graph_boost, 4),
            "node_level":  c.node_level,
            # ── Source traceability ───────────────────────────
            "url":         source_url,
            "pdf_url":     pdf_url,
            "title":       meta.get("title",       ""),
            "source_type": meta.get("source_type", ""),
            "language":    meta.get("language",    ""),
            "chunk_index": meta.get("chunk_index"),
        })

    if result:
        result[0]["llm_context"] = _format_context(
            [c for c in chunks if c.score >= min_score_threshold][:len(result)]
        )

    return result

def search_nodes(query: str, node_type: Optional[str] = None) -> List[Dict]:
    return _get_retriever().search_nodes(query, node_type)


def get_node_info(node_id: str) -> Optional[Dict]:
    return _get_retriever().get_node_info(node_id)


def traverse_graph(query: str) -> Dict:
    """Direct graph traversal — returns hierarchy tree (for graph_query intent)."""
    return _get_retriever().graph.traverse_for_graph_query(query)


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into an LLM-ready context block.
    Every chunk includes its source URL and file type clearly.
    """
    sep    = "\n" + "─" * 60 + "\n"
    blocks = []

    for i, c in enumerate(chunks, 1):
        meta       = c.metadata or {}
        title      = meta.get("title", "Unknown Source")
        source_url = meta.get("page_url") or meta.get("url") or ""
        pdf_url    = meta.get("pdf_url") or ""
        stype      = meta.get("source_type", "")
        level_tag  = f" [graph_depth={c.node_level}]" if c.node_level >= 0 else ""
        
        # ✨ أضف درجة الثقة
        score_tag  = f" [score={c.score:.2f}]"

        lines = [f"[{i}] {title}{level_tag}{score_tag}"]

        # Clear source identification
        if stype == "pdf" and pdf_url:
            lines.append(f"📄 Source PDF : {pdf_url}")
            if source_url and source_url != pdf_url:
                lines.append(f"🔗 Page web   : {source_url}")
        elif source_url:
            lines.append(f"🔗 Source     : {source_url}")

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
            "  python rag_pipeline.py \"query\" [--top-k N]\n"
            "  python rag_pipeline.py --search-nodes \"term\" [--type Department]\n"
            "  python rag_pipeline.py --node-info <node_id>\n"
            "  python rag_pipeline.py --graph \"query\"\n"
        )
        sys.exit(1)

    query            = None
    top_k            = TOP_K_FINAL
    search_nodes_q   = None
    search_node_type = None
    node_info_id     = None
    graph_query      = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1]); i += 2
        elif arg == "--search-nodes" and i + 1 < len(sys.argv):
            search_nodes_q = sys.argv[i + 1]; i += 2
        elif arg == "--type" and i + 1 < len(sys.argv):
            search_node_type = sys.argv[i + 1]; i += 2
        elif arg == "--node-info" and i + 1 < len(sys.argv):
            node_info_id = sys.argv[i + 1]; i += 2
        elif arg == "--graph" and i + 1 < len(sys.argv):
            graph_query = sys.argv[i + 1]; i += 2
        else:
            if query is None and not arg.startswith("--"):
                query = arg
            i += 1

    # ── Graph traversal ───────────────────────────────────────
    if graph_query:
        result = traverse_graph(graph_query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(0)

    # ── Node info ─────────────────────────────────────────────
    if node_info_id:
        info = get_node_info(node_info_id)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        sys.exit(0)

    # ── Node search ───────────────────────────────────────────
    if search_nodes_q:
        nodes = search_nodes(search_nodes_q, search_node_type)
        for n in nodes:
            print(f"  {n['name']:<40} [{n['label']:<18}] {n['node_id']}")
        sys.exit(0)

    # ── Full retrieval ────────────────────────────────────────
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
            level_tag = f" [L{r['node_level']}]" if r.get("node_level", -1) >= 0 else ""
            stype     = r.get("source_type", "")
            src_icon  = "📄" if stype == "pdf" else "🔗"
            print(f"\n{rank}.  {r['chunk_id']}{level_tag}")
            print(f"   Score    : {r['score']:.4f}  "
                  f"(sem={r['sem_score']:.3f}  bm25={r['bm25_score']:.3f}  "
                  f"fused={r['fused_score']:.3f}  rerank={r['rerank_cal']:.3f}  "
                  f"boost={r['graph_boost']:.3f})")
            print(f"   Title    : {(r['title'] or 'N/A')[:80]}")
            print(f"   {src_icon} Source : {r.get('url') or r.get('pdf_url') or 'N/A'}")
            print(f"   Language : {r.get('language','?')}  |  Type: {stype or 'page'}")
            print(f"   Preview  : {r['text'][:220]}…")

        if results[0].get("llm_context"):
            print(f"\n{'═' * 65}")
            print("LLM CONTEXT PREVIEW (first 1200 chars)")
            print(f"{'═' * 65}")
            print(results[0]["llm_context"][:1200])
    else:
        print("No query provided.  Use --help for usage.")
        sys.exit(1)
