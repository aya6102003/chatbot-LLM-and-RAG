#!/usr/bin/env python3
"""
RAG Ingestion Pipeline v6 — Farhat Abbas University Sétif 1
============================================================
Fully fixed version:
  • Graph classification nodes ALREADY EXIST — only LINKS chunks to them
  • Uses aliases.json (keywords section) to resolve abbreviations to canonical names
  • Validates ALL found canonical names against actual structure.json
  • Multi-entity classification: links EACH chunk to ALL matched graph nodes
  • NO false classifications — each chunk goes to ALL its true places

Dependencies
────────────
  pip install sentence-transformers chromadb neo4j numpy tiktoken
  pip install rank-bm25 orjson rapidfuzz
"""

# ─────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────
import hashlib
import json
import logging
import os
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from neo4j import GraphDatabase

try:
    from rapidfuzz import fuzz as _rfuzz, process as _rfprocess
    _RAPIDFUZZ_OK = True
except ImportError:
    _RAPIDFUZZ_OK = False
    logging.warning("rapidfuzz not installed — fuzzy matching disabled")

try:
    import orjson as _json_lib
    def _load_json(fh):
        return _json_lib.loads(fh.read())
except ImportError:
    _json_lib = None
    def _load_json(fh):
        return json.load(fh)

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False
    logging.warning("tiktoken not installed — char-based token counting active")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
ROOT_FOLDER    = "./university_farhat_abaas"
CHROMA_PATH    = "./chroma_db"
METADATA_PATH  = "./metadata.json"
PROGRESS_DB    = "./pipeline_progress.db"

UNIVERSITY_STRUCTURE_PATH = "./university_structure.json"
ALIASES_PATH              = "./aliases.json"

NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

UNIVERSITY_NAME = "Farhat Abbas University Sétif 1"

EMBED_MODEL      = "intfloat/multilingual-e5-large"
SENT_SPLIT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PASSAGE_PREFIX = "passage: "

EMBED_BATCH        = 64
CHROMA_BATCH       = 100
NEO4J_BATCH        = 50
PROGRESS_FLUSH     = 50
PARSE_WORKERS      = min(8, (os.cpu_count() or 4))

CHUNK_TOKENS_BASE     = 400
OVERLAP_TOKENS        = 80
MIN_CHUNK_CHARS       = 80
MIN_DOC_CHARS         = 100
MAX_BOILERPLATE_RATIO = 0.25

TOPIC_DRIFT_THRESHOLD    = 0.65
RERANK_QUALITY_FLOOR     = -3.0
CROSS_ENCODER_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_ENABLED    = True
SEMANTIC_DEDUP_THRESHOLD = 0.92
SEMANTIC_DEDUP_WINDOW    = 8

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FACULTY LABELS (for folder mapping)
# ─────────────────────────────────────────────────────────────
FACULTY_LABELS: Dict[str, str] = {
    "farhat_abbas_university": "Farhat Abbas University Sétif 1",
    "ftechnologie": "Faculty of Technology",
    "fsciences":    "Faculty of Science",
    "fsnv":         "Faculty of Nature and Life Sciences",
    "feco":         "Faculty of Economics, Business and Management Sciences",
    "fmed":         "Faculty of Medicine",
    "iomp":         "Institute of Optics and Precision Mechanics",
    "iast":         "Institute of Architecture and Earth Sciences",
    "istm":         "Institute of Materials Science and Techniques",
}

# ─────────────────────────────────────────────────────────────
# WORD LISTS & FROZENSETS
# ─────────────────────────────────────────────────────────────
ACADEMIC_INDICATORS = [
    "semester", "module", "course", "exam", "lecture", "syllabus",
    "credits", "prerequisite", "assignment", "curriculum",
    "semestre", "cours", "examen", "licence", "master", "doctorat",
    "formation", "filière", "td", "tp", "contrôle", "ingénieur", "ingenieur",
    "الفصل", "الدراسي", "امتحان", "مقياس", "تخصص", "برنامج",
]
BOILERPLATE_WORDS = [
    "copyright", "all rights reserved", "privacy policy", "terms of use",
    "click here", "read more", "subscribe", "newsletter", "cookie policy",
    "navigation", "footer", "header", "sitemap",
    "home", "menu", "accueil", "الرئيسية", "contact", "about",
    "connexion", "login", "sign in", "se connecter",
    "skip to content", "back to top", "print page",
]
AUTHORITY_SIGNALS = [
    "arrêté", "décret", "décision", "circulaire", "règlement", "official",
    "ministry", "ministère", "وزارة", "مرسوم", "قرار", "مذكرة",
    "journal officiel", "bulletin officiel",
]
EXAM_SIGNALS   = ["exam", "امتحان", "contrôle", "épreuve", "test", "quiz"]
COURSE_SIGNALS = ["cours", "course", "محاضرة", "lecture", "td", "tp", "syllabus"]
ADMIN_SIGNALS  = ["admin", "إدارة", "scolarité", "inscription", "calendrier",
                  "règlement", "décret", "arrêté"]

_BOILERPLATE_SET: FrozenSet[str] = frozenset(BOILERPLATE_WORDS)
_ACADEMIC_SET:    FrozenSet[str] = frozenset(ACADEMIC_INDICATORS)
_AUTHORITY_SET:   FrozenSet[str] = frozenset(AUTHORITY_SIGNALS)
_EXAM_SET:        FrozenSet[str] = frozenset(EXAM_SIGNALS)
_COURSE_SET:      FrozenSet[str] = frozenset(COURSE_SIGNALS)
_ADMIN_SET:       FrozenSet[str] = frozenset(ADMIN_SIGNALS)
_BPLATE_DENOM = float(len(_BOILERPLATE_SET))

# ─────────────────────────────────────────────────────────────
# PRE-COMPILED REGEX
# ─────────────────────────────────────────────────────────────
_RE_CONTROL    = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_TASHKEEL   = re.compile(r"[ًٌٍَُِّْـ]")
_RE_ALEF       = re.compile(r"[أإآٱ]")
_RE_REPEATS    = re.compile(r"(.)\1{2,}")
_RE_APOSTROPHE = re.compile(r"[''`´]")
_RE_QUOTES     = re.compile(r"[«»""„]")
_RE_PAGE_NUM   = re.compile(r"(?m)^\s*\d{1,4}\s*$")
_RE_MULTI_NL   = re.compile(r"\n{3,}")
_RE_SPACES     = re.compile(r"[ \t]+")
_RE_SENT_BOUND = re.compile(r"(?<=[.!?؟])\s+")
_RE_YEAR       = re.compile(r"\b(20[12]\d)\b")
_RE_AR_CHARS   = re.compile(r"[\u0600-\u06FF]")
_RE_FR_WORDS   = re.compile(
    r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
    r"cours|semestre|licence|master|doctorat|filière)\b", re.IGNORECASE)
_RE_URL_GENERAL = re.compile(r"https?://[^\s<>\"')\]]+")
_RE_PDF_URL     = re.compile(
    r"https?://[^\s<>\"')\]]+\.pdf(?:[?#][^\s<>\"')\]]*)?", re.IGNORECASE)
_RE_REL_PDF     = re.compile(r"[\"']([^\"']+\.pdf)[\"']", re.IGNORECASE)
_RE_COURSE_CODE = re.compile(
    r"\b([A-Z]{2,5}\s?\d{3,4}|[A-Z]{3,6}-\d{2,3}|LMD|S[1-6])\b")
_RE_EMAIL       = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
_RE_PHONE       = re.compile(r"(?:\+213|0)[5-7]\d{8}")
_RE_HEADING     = re.compile(
    r"(?m)^(#{1,4}\s.+|[A-ZÀÁÂÄÉÈÊËÎÏÔÙÛÜ][A-ZÀÁÂÄÉÈÊËÎÏÔÙÛÜ\s:,]{8,}|"
    r"[\u0600-\u06FF]{4,}[\s:]+[\u0600-\u06FF].*|Chapitre|Section|Partie|"
    r"Chapter|Part\s+\d|الفصل|القسم|الجزء)\s*$", re.MULTILINE)
_RE_PARA_BREAK = re.compile(r"\n{2,}")


# ══════════════════════════════════════════════════════════════
# ✦ ALIAS RESOLVER — يبحث عن كل الكلمات في aliases.json
# ══════════════════════════════════════════════════════════════

class AliasResolver:
    """
    يبحث عن كل الكلمات الموجودة في aliases.json داخل النص
    ويرجع كل الأسماء القانونية (canonical names) المرتبطة بكل كلمة
    """
    
    def __init__(self, path: str = ALIASES_PATH):
        self.keyword_map: Dict[str, List[str]] = {}
        self._all_keywords_sorted: List[str] = []
        self._load(path)
    
    def _load(self, path: str):
        """تحميل ملف aliases.json وبناء خريطة الكلمات المفتاحية"""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            
            if "keywords" in data:
                for keyword, canonicals in data["keywords"].items():
                    if keyword.startswith("_"):
                        continue
                    
                    keyword_lower = keyword.lower().strip()
                    if isinstance(canonicals, list):
                        self.keyword_map[keyword_lower] = canonicals
                    elif isinstance(canonicals, str):
                        self.keyword_map[keyword_lower] = [canonicals]
            
            self._all_keywords_sorted = sorted(
                self.keyword_map.keys(), 
                key=len, 
                reverse=True
            )
            
            log.info("✅ Aliases loaded: %d keywords", len(self.keyword_map))
            
        except FileNotFoundError:
            log.warning("⚠  Aliases file not found at %s", path)
        except Exception as exc:
            log.warning("⚠  Failed to load aliases: %s", exc)
    
    def find_all_canonical_names(self, text: str) -> Dict[str, List[str]]:
        """
        البحث عن كل الكلمات المفتاحية في النص وإرجاع الأسماء القانونية
        
        Args:
            text: النص المراد البحث فيه
        
        Returns:
            قاموس: {keyword_found: [canonical_name1, canonical_name2, ...]}
        """
        if not text or not self.keyword_map:
            return {}
        
        text_lower = text.lower()
        found_keywords: Dict[str, List[str]] = {}
        matched_positions: List[Tuple[int, int]] = []
        
        for keyword in self._all_keywords_sorted:
            keyword_lower = keyword.lower()
            
            try:
                for match in re.finditer(re.escape(keyword_lower), text_lower):
                    start, end = match.start(), match.end()
                    
                    overlap = False
                    for ms, me in matched_positions:
                        if start >= ms and end <= me:
                            overlap = True
                            break
                    
                    if overlap:
                        continue
                    
                    if len(keyword) <= 3:
                        if start > 0 and (text_lower[start-1].isalnum() and text_lower[start-1] not in '0123456789'):
                            continue
                        if end < len(text_lower) and text_lower[end].isalnum():
                            continue
                    
                    matched_positions.append((start, end))
                    
                    if keyword not in found_keywords:
                        found_keywords[keyword] = self.keyword_map[keyword]
            
            except re.error:
                log.warning("Regex error for keyword: %s", keyword)
                continue
        
        return found_keywords
    
    def get_all_canonical_names_flat(self, text: str) -> List[str]:
        """
        إرجاع قائمة مسطحة بكل الأسماء القانونية الفريدة (بدون تكرار)
        """
        found = self.find_all_canonical_names(text)
        all_names: List[str] = []
        seen: Set[str] = set()
        
        for names in found.values():
            for name in names:
                if name not in seen:
                    seen.add(name)
                    all_names.append(name)
        
        return all_names


# ══════════════════════════════════════════════════════════════
# ✦ UNIVERSITY STRUCTURE LOADER
# ══════════════════════════════════════════════════════════════

def load_university_structure(path: str = UNIVERSITY_STRUCTURE_PATH) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        log.info("✅ University structure loaded from %s", path)
        return data
    except FileNotFoundError:
        log.warning("⚠  University structure file not found at %s", path)
        return {"faculte": "", "departements": []}
    except Exception as exc:
        log.warning("⚠  Failed to load university structure: %s", exc)
        return {"faculte": "", "departements": []}


# ══════════════════════════════════════════════════════════════
# ✦ GRAPH-AWARE INDEX — Built from actual structure.json
# ══════════════════════════════════════════════════════════════

class GraphAwareIndex:
    """
    فهرس مسطح بكل الكيانات الموجودة في university_structure.json
    """
    
    def __init__(self, structure: Dict):
        self.entries: List[Dict] = []
        self._name_index: Dict[str, List[Dict]] = {}
        self._build(structure)
        log.info("✅ GraphAwareIndex built: %d entries, %d unique names", 
                len(self.entries), len(self._name_index))
    
    def _build(self, structure: Dict):
        faculte_name = structure.get("faculte", "")
        
        for dept in structure.get("departements", []):
            dept_name = dept.get("nom", "")
            self._add(dept_name, "departement", faculte=faculte_name, departement=dept_name)
            
            if "niveaux" in dept:
                self._process_niveaux(dept["niveaux"], faculte=faculte_name, departement=dept_name)
            
            if "filieres" in dept:
                for fil in dept["filieres"]:
                    fil_name = fil.get("nom", "")
                    self._add(fil_name, "filiere", faculte=faculte_name, 
                             departement=dept_name, filiere=fil_name)
                    if "niveaux" in fil:
                        self._process_niveaux(fil["niveaux"], faculte=faculte_name,
                                            departement=dept_name, filiere=fil_name)
    
    def _process_niveaux(self, niveaux: List[Dict], **hierarchy):
        for niv in niveaux:
            niv_name = niv.get("nom", "")
            if not niv_name:
                continue
            
            self._add(niv_name, "niveau", niveau=niv_name, **hierarchy)
            
            if "specialites" in niv:
                for spec in niv["specialites"]:
                    self._process_specialite(spec, niv_name, **hierarchy)
            
            if "parcours" in niv:
                for parc in niv["parcours"]:
                    if "specialites" in parc:
                        for spec in parc["specialites"]:
                            self._process_specialite(spec, niv_name, **hierarchy)
    
    def _process_specialite(self, spec: Dict, niveau_name: str, **hierarchy):
        spec_name = spec.get("nom", "")
        if not spec_name:
            return
        
        self._add(spec_name, "specialite", niveau=niveau_name, 
                 specialite=spec_name, **hierarchy)
        
        for an in spec.get("annees", []):
            an_name = an.get("nom", "")
            if an_name:
                self._add(an_name, "annee", niveau=niveau_name,
                         specialite=spec_name, annee=an_name, **hierarchy)
    
    def _add(self, original: str, level: str, **kwargs):
        if not original:
            return
        
        entry = {
            "original":    original,
            "name":        original.lower().strip(),
            "level":       level,
            "faculte":     kwargs.get("faculte"),
            "departement": kwargs.get("departement"),
            "filiere":     kwargs.get("filiere"),
            "niveau":      kwargs.get("niveau"),
            "specialite":  kwargs.get("specialite"),
            "annee":       kwargs.get("annee"),
        }
        
        self.entries.append(entry)
        
        name_key = original.lower().strip()
        if name_key not in self._name_index:
            self._name_index[name_key] = []
        self._name_index[name_key].append(entry)
    
    def find_any(self, name: str) -> List[Dict]:
        """
        البحث عن كيان بالاسم القانوني (في أي مستوى)
        """
        if not name:
            return []
        
        name_lower = name.lower().strip()
        
        if name_lower in self._name_index:
            return self._name_index[name_lower]
        
        results = []
        for entry in self.entries:
            if entry["original"] == name:
                results.append(entry)
        
        return results
    
    def find_by_name(self, name: str, level: str = None) -> List[Dict]:
        name_lower = name.lower().strip()
        return [
            e for e in self.entries
            if (level is None or e["level"] == level) and e["name"] == name_lower
        ]
    
    def get_all_names(self) -> Set[str]:
        return set(self._name_index.keys())


# ══════════════════════════════════════════════════════════════
# ✦ GRAPH-AWARE CLASSIFIER — Uses aliases + structure validation
# ══════════════════════════════════════════════════════════════

class GraphAwareClassifier:
    """
    المصنف الرئيسي - يبحث عن كل الكلمات المفتاحية في النص
    ويربط الـ chunk بكل الكيانات الموجودة في الرسم البياني
    """
    
    def __init__(self, index: GraphAwareIndex, alias_resolver: AliasResolver):
        self._idx = index
        self._aliases = alias_resolver
    
    def classify(self, text: str, title: str = "",
                 faculty_hint: str = "", department_hint: str = "") -> Dict[str, Any]:
        """
        تصنيف النص واستخراج كل الكيانات المرتبطة به
        """
        combined = self._combined_text(text, title, faculty_hint, department_hint)
        
        # Step 1: Find ALL alias keywords and their canonical names
        found_keywords = self._aliases.find_all_canonical_names(combined)
        
        if not found_keywords:
            return self._empty_result()
        
        # Step 2: Collect ALL unique canonical names
        all_canonicals: Set[str] = set()
        for names in found_keywords.values():
            all_canonicals.update(names)
        
        # Step 3: Validate each canonical name against graph structure
        validated_entities: List[Dict] = []
        validated_names: Set[str] = set()
        not_found_names: List[str] = []
        
        for canonical in all_canonicals:
            entries = self._idx.find_any(canonical)
            
            if entries:
                for entry in entries:
                    entity_info = {
                        "canonical": canonical,
                        "level": entry["level"],
                        "name": entry["original"],
                        "faculte": entry.get("faculte"),
                        "departement": entry.get("departement"),
                        "filiere": entry.get("filiere"),
                        "niveau": entry.get("niveau"),
                        "specialite": entry.get("specialite"),
                        "annee": entry.get("annee"),
                    }
                    
                    entity_key = f"{entity_info['level']}:{entity_info['name']}"
                    if entity_key not in validated_names:
                        validated_names.add(entity_key)
                        validated_entities.append(entity_info)
            else:
                not_found_names.append(canonical)
        
        if not_found_names:
            log.debug("Names in aliases but NOT in graph structure: %s", not_found_names)
        
        if not validated_entities:
            return self._empty_result()
        
        # Step 4: Group by level
        by_level = self._group_by_level(validated_entities)
        
        # Step 5: Build final result
        return self._build_final_result(validated_entities, by_level, found_keywords)
    
    def _group_by_level(self, entities: List[Dict]) -> Dict[str, List[str]]:
        grouped: Dict[str, Set[str]] = {
            "annee": set(),
            "specialite": set(),
            "niveau": set(),
            "filiere": set(),
            "departement": set(),
        }
        
        for entity in entities:
            level = entity["level"]
            if level in grouped:
                grouped[level].add(entity["name"])
        
        return {k: sorted(v) for k, v in grouped.items() if v}
    
    def _build_final_result(self, entities: List[Dict], 
                            by_level: Dict[str, List[str]],
                            found_keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        
        match_parts = []
        if by_level.get("annee"): match_parts.append("annee")
        if by_level.get("specialite"): match_parts.append("specialite")
        if by_level.get("niveau"): match_parts.append("niveau")
        if by_level.get("filiere"): match_parts.append("filiere")
        if by_level.get("departement"): match_parts.append("departement")
        
        if len(match_parts) > 1:
            match_method = "alias_multi_" + "_".join(match_parts)
        elif len(match_parts) == 1:
            match_method = f"alias_{match_parts[0]}"
        else:
            match_method = "alias"
        
        primary = self._get_primary_entity(entities, by_level)
        
        return {
            "category": "academic",
            "multi_entity": len(entities) > 1,
            "match_method": match_method,
            "matched_keywords": list(found_keywords.keys()),
            "validated_entities": entities,
            "annees": by_level.get("annee", []),
            "specialites": by_level.get("specialite", []),
            "niveaux": by_level.get("niveau", []),
            "filieres": by_level.get("filiere", []),
            "departements": by_level.get("departement", []),
            "faculte": primary.get("faculte"),
            "departement": primary.get("departement"),
            "filiere": primary.get("filiere"),
            "niveau": primary.get("niveau"),
            "specialite": primary.get("specialite"),
            "annee": by_level.get("annee", [None])[0] if by_level.get("annee") else None,
        }
    
    def _get_primary_entity(self, entities: List[Dict], 
                            by_level: Dict[str, List[str]]) -> Dict:
        priority = ["specialite", "annee", "niveau", "filiere", "departement"]
        
        for level in priority:
            if level in by_level and by_level[level]:
                for entity in entities:
                    if entity["level"] == level:
                        return entity
        
        return entities[0] if entities else {}
    
    def _combined_text(self, text: str, title: str,
                       faculty_hint: str, dept_hint: str) -> str:
        parts = []
        if title:
            parts.append(f"{title} {title}")
        if faculty_hint:
            parts.append(faculty_hint)
        if dept_hint:
            parts.append(dept_hint)
        parts.append(text[:5000])
        return " ".join(parts)
    
    def _empty_result(self) -> Dict[str, Any]:
        return {
            "category": "general",
            "multi_entity": False,
            "match_method": "none",
            "matched_keywords": [],
            "validated_entities": [],
            "annees": [],
            "specialites": [],
            "niveaux": [],
            "filieres": [],
            "departements": [],
            "faculte": None,
            "departement": None,
            "filiere": None,
            "niveau": None,
            "specialite": None,
            "annee": None,
        }


# ══════════════════════════════════════════════════════════════
# TEXT NORMALIZATION
# ══════════════════════════════════════════════════════════════

def normalize_text(text: str, aggressive: bool = False) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _RE_CONTROL.sub("", text)
    text = _RE_TASHKEEL.sub("", text)
    if aggressive:
        text = _RE_ALEF.sub("ا", text)
        text = text.replace("ة", "ه").replace("ى", "ي")
        text = (text
                .replace("é", "e").replace("è", "e").replace("ê", "e")
                .replace("à", "a").replace("â", "a")
                .replace("ù", "u").replace("û", "u")
                .replace("î", "i").replace("ô", "o").replace("ç", "c"))
    text = _RE_REPEATS.sub(r"\1\1", text)
    text = text.replace("œ", "oe").replace("æ", "ae").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = _RE_APOSTROPHE.sub("'", text)
    text = _RE_QUOTES.sub('"', text)
    text = _RE_PAGE_NUM.sub("", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    text = _RE_SPACES.sub(" ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
# LANGUAGE / DOC-TYPE / YEAR DETECTION
# ══════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    s = text[:300]
    if len(_RE_AR_CHARS.findall(s)) > 15:
        return "ar"
    if len(_RE_FR_WORDS.findall(s)) > 3:
        return "fr"
    return "en"


def infer_doc_type(filename: str, title: str, text_sample: str) -> str:
    c = f"{filename} {title} {text_sample[:200]}".lower()
    if any(w in c for w in _EXAM_SET):   return "exam"
    if any(w in c for w in _COURSE_SET): return "course"
    if any(w in c for w in _ADMIN_SET):  return "admin"
    return "general"


def extract_year(text: str) -> Optional[int]:
    m = _RE_YEAR.search(text[:500])
    return int(m.group(1)) if m else None


# ══════════════════════════════════════════════════════════════
# SCORING & METADATA HELPERS
# ══════════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, List[str]]:
    return {
        "course_codes": list(dict.fromkeys(_RE_COURSE_CODE.findall(text)))[:10],
        "emails":       list(dict.fromkeys(_RE_EMAIL.findall(text)))[:5],
        "phones":       list(dict.fromkeys(_RE_PHONE.findall(text)))[:5],
    }


def compute_authority_score(text: str, doc_type: str, source: str) -> float:
    low  = text[:600].lower()
    hits = sum(1 for w in _AUTHORITY_SET if w in low)
    base = min(1.0, hits / 4.0)
    return round(min(1.0, base + (0.2 if source == "pdf" else 0.0) + 
                     (0.1 if doc_type == "admin" else 0.0)), 3)


_ACADEMIC_IDF: Dict[str, float] = {
    "prerequisite": 3.5, "syllabus": 3.2, "curriculum": 3.0, "برنامج": 3.0,
    "تخصص": 2.9, "filière": 2.8, "doctorat": 2.7, "assignment": 2.6,
    "credits": 2.5, "module": 2.0, "مقياس": 2.0, "semester": 1.9,
    "semestre": 1.9, "lecture": 1.8, "master": 1.7, "licence": 1.7,
    "exam": 1.5, "examen": 1.5, "امتحان": 1.5, "td": 1.4, "tp": 1.4,
    "contrôle": 1.4, "cours": 1.1, "course": 1.1, "الفصل": 1.0,
    "الدراسي": 1.0, "formation": 1.0, "ingénieur": 1.7, "ingenieur": 1.7,
}
_DEFAULT_IDF = 1.2


def compute_academic_score(text: str) -> float:
    if not text:
        return 0.0
    low = text.lower()
    words = low.split()
    if not words:
        return 0.0
    n = len(words)
    scores = []
    for term in _ACADEMIC_SET:
        if term not in low:
            continue
        tf = low.count(term) / n
        scores.append(tf * _ACADEMIC_IDF.get(term, _DEFAULT_IDF))
    if not scores:
        return 0.0
    top3 = sorted(scores, reverse=True)[:3]
    raw = sum(top3) / len(top3)
    return round(min(1.0, raw / (raw + 0.05)), 3)


_TOKEN_IDF_TABLE: Dict[str, float] = {
    "the": 0.1, "de": 0.2, "la": 0.2, "le": 0.2, "et": 0.2,
    "is": 0.3, "en": 0.3, "du": 0.3, "un": 0.3, "une": 0.3,
    "dans": 0.4, "sur": 0.4, "pour": 0.4, "avec": 0.4,
    "université": 0.8, "faculté": 0.8, "étudiant": 0.9,
    "syllabus": 3.2, "prerequisite": 3.5, "examen": 1.5,
    "semestre": 1.9, "filière": 2.8, "doctorat": 2.7,
}
_DEFAULT_TOKEN_IDF = 1.0


def compute_keyword_density(text: str) -> Tuple[float, float]:
    words = text.lower().split()
    if not words:
        return 0.0, 0.0
    n = len(words)
    acad = sum(1 for w in words if w in _ACADEMIC_SET)
    avg_idf = sum(_TOKEN_IDF_TABLE.get(w, _DEFAULT_TOKEN_IDF) for w in words) / n
    return round(acad / n, 4), round(avg_idf, 4)


# ══════════════════════════════════════════════════════════════
# ADAPTIVE CHUNK SIZER
# ══════════════════════════════════════════════════════════════

class AdaptiveChunkSizer:
    _TYPE_BASE: Dict[str, int] = {
        "exam": 200, "course": 350, "admin": 300, "general": 500,
    }
    _MIN_TOKENS = 150
    _MAX_TOKENS = 700

    def compute(self, doc_type: str, keyword_density: float, avg_token_idf: float) -> int:
        base = self._TYPE_BASE.get(doc_type, CHUNK_TOKENS_BASE)
        kd_norm = min(1.0, keyword_density / 0.10)
        idf_norm = min(1.0, avg_token_idf / 2.50)
        result = int(base * (1.0 - 0.4 * kd_norm) * (1.0 - 0.2 * idf_norm))
        return max(self._MIN_TOKENS, min(self._MAX_TOKENS, result))


# ══════════════════════════════════════════════════════════════
# TITLE ALIASES & SECTION DETECTION
# ══════════════════════════════════════════════════════════════

def generate_title_aliases(title: str) -> List[str]:
    if not title:
        return []
    aliases: List[str] = []
    alias_norm = normalize_text(title, aggressive=True).lower()
    if alias_norm and alias_norm != title.lower():
        aliases.append(alias_norm)
    latin_only = re.sub(r"[\u0600-\u06FF\s]+", " ", title).strip().lower()
    if latin_only and latin_only not in aliases:
        aliases.append(latin_only)
    arabic_words = " ".join(_RE_AR_CHARS.findall(title))
    if arabic_words and arabic_words not in aliases:
        aliases.append(arabic_words)
    return [a for a in aliases if len(a) > 2]


def detect_sections(text: str, doc_fp: str) -> List[Dict]:
    sections: List[Dict] = []
    matches = list(_RE_HEADING.finditer(text))
    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        start_char = m.start()
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_id = hashlib.sha1(f"{doc_fp}:{heading}".encode()).hexdigest()[:12]
        sections.append({
            "section_id": section_id, "heading": heading,
            "start_char": start_char, "end_char": end_char,
        })
    if not sections:
        sections.append({
            "section_id": hashlib.sha1(f"{doc_fp}:__root__".encode()).hexdigest()[:12],
            "heading": "", "start_char": 0, "end_char": len(text),
        })
    return sections


# ══════════════════════════════════════════════════════════════
# SEMANTIC CHUNKER V2
# ══════════════════════════════════════════════════════════════

class SemanticChunkerV2:
    HINT_MAX_CHARS = 100

    def __init__(self, chunk_tokens: int = CHUNK_TOKENS_BASE, overlap_tokens: int = OVERLAP_TOKENS,
                 min_chars: int = MIN_CHUNK_CHARS, drift_threshold: float = TOPIC_DRIFT_THRESHOLD):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chars = min_chars
        self.drift_threshold = drift_threshold
        self._enc = tiktoken.get_encoding("cl100k_base") if _TIKTOKEN_OK else None

    def split(self, text: str, title: str = "", split_model = None, chunk_tokens: int = 0,
              doc_fp: str = "", sections: List[Dict] = None) -> List[Dict]:
        effective_tokens = chunk_tokens or self.chunk_tokens
        text = normalize_text(text)
        if not text:
            return []
        text = remove_repeated_blocks(text)
        if not text:
            return []
        if sections is None:
            sections = detect_sections(text, doc_fp)

        section_segments = self._split_by_sections(text, sections)
        all_chunks: List[Dict] = []
        global_chunk_idx = 0

        for seg_text, seg_start, section_id in section_segments:
            if not seg_text.strip():
                continue
            paragraphs = self._split_paragraphs(seg_text, base_offset=seg_start)
            raw_chunks = self._pack_paragraphs(paragraphs, effective_tokens, split_model, section_id)
            for chunk_data in raw_chunks:
                body = chunk_data["text"]
                embed_body = f"{title}\n{body}".strip() if title and not body.lower().startswith(title.lower()[:30]) else body
                if len(embed_body) < self.min_chars:
                    continue
                prev_hint = all_chunks[-1]["clean_body"][-self.HINT_MAX_CHARS:] if all_chunks else ""
                full_text = f"[prev: {prev_hint}]\n{embed_body}" if prev_hint else embed_body
                all_chunks.append({
                    "embed_text": embed_body, "text": full_text, "clean_body": body,
                    "token_count": chunk_data["token_count"], "chunk_index": global_chunk_idx,
                    "start_char": chunk_data["start_char"], "section_id": section_id,
                    "prev_hint": prev_hint, "next_hint": "",
                })
                global_chunk_idx += 1

        for i in range(len(all_chunks) - 1):
            hint = all_chunks[i + 1]["clean_body"][:self.HINT_MAX_CHARS]
            all_chunks[i]["next_hint"] = hint
            all_chunks[i]["text"] = f"{all_chunks[i]['text']}\n[next: {hint}]"

        return all_chunks

    def _split_by_sections(self, text: str, sections: List[Dict]) -> List[Tuple[str, int, str]]:
        segments = [(text[s["start_char"]:s["end_char"]], s["start_char"], s["section_id"]) for s in sections]
        return segments if segments else [(text, 0, "")]

    def _split_paragraphs(self, text: str, base_offset: int = 0) -> List[Tuple[str, int]]:
        result, pos = [], 0
        for para in _RE_PARA_BREAK.split(text):
            stripped = para.strip()
            if stripped:
                result.append((stripped, base_offset + pos))
            pos += len(para) + 2
        return result if result else [(text.strip(), base_offset)]

    def _pack_paragraphs(self, paragraphs: List[Tuple[str, int]], chunk_tokens: int,
                          split_model, section_id: str) -> List[Dict]:
        chunks, current_parts, current_tok, current_start = [], [], 0, 0

        for para_text, para_start in paragraphs:
            para_tok = self._token_count(para_text)
            if para_tok > chunk_tokens:
                if current_parts:
                    chunks.append({"text": " ".join(current_parts), "token_count": current_tok, "start_char": current_start})
                    current_parts, current_tok = [], 0
                chunks.extend(self._split_paragraph_sentences(para_text, para_start, chunk_tokens, split_model))
                continue

            if current_tok + para_tok > chunk_tokens and current_parts:
                chunks.append({"text": " ".join(current_parts), "token_count": current_tok, "start_char": current_start})
                overlap_parts, overlap_tok = [], 0
                for prev in reversed(current_parts):
                    pt = self._token_count(prev)
                    if overlap_tok + pt > self.overlap_tokens:
                        break
                    overlap_parts.insert(0, prev)
                    overlap_tok += pt
                current_parts = overlap_parts + [para_text]
                current_tok = overlap_tok + para_tok
                current_start = para_start
            else:
                if not current_parts:
                    current_start = para_start
                current_parts.append(para_text)
                current_tok += para_tok

        if current_parts:
            chunks.append({"text": " ".join(current_parts), "token_count": current_tok, "start_char": current_start})
        return chunks

    def _split_paragraph_sentences(self, para_text: str, base_offset: int, chunk_tokens: int, split_model) -> List[Dict]:
        sentences, sent_offsets = self._split_sentences(para_text)
        if not sentences:
            return []
        drift_positions = self._find_semantic_breaks(sentences, split_model) if split_model and len(sentences) > 8 else set()
        packed = self._pack_sentences(sentences, drift_positions, chunk_tokens)
        return [{"text": " ".join(sents), "token_count": tok, "start_char": base_offset + (sent_offsets[si] if si < len(sent_offsets) else 0)} 
                for sents, tok, si in packed]

    def _token_count(self, text: str) -> int:
        return len(self._enc.encode(text)) if self._enc else len(text) // 4

    def _split_sentences(self, text: str) -> Tuple[List[str], List[int]]:
        sentences, offsets, pos = [], [], 0
        for part in _RE_SENT_BOUND.split(text):
            stripped = part.strip()
            if stripped:
                sentences.append(stripped)
                offsets.append(pos)
            pos += len(part) + 1
        return sentences, offsets

    def _find_semantic_breaks(self, sentences: List[str], split_model) -> Set[int]:
        try:
            embs = split_model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        except Exception:
            return set()
        return {i + 1 for i in range(len(embs) - 1) if float(np.dot(embs[i], embs[i + 1])) < self.drift_threshold}

    def _pack_sentences(self, sentences: List[str], drift_breaks: Set[int], chunk_tokens: int) -> List[Tuple[List[str], int, int]]:
        chunks, current_sents, current_tok, start_idx = [], [], 0, 0
        for idx, sent in enumerate(sentences):
            sent_tok = self._token_count(sent)
            is_break = idx in drift_breaks
            budget_exceeded = current_tok + sent_tok > chunk_tokens and current_sents
            if (is_break or budget_exceeded) and current_sents:
                chunks.append((list(current_sents), current_tok, start_idx))
                if not is_break:
                    ov_sents, ov_tok = [], 0
                    for s in reversed(current_sents):
                        st = self._token_count(s)
                        if ov_tok + st > self.overlap_tokens:
                            break
                        ov_sents.insert(0, s)
                        ov_tok += st
                    current_sents = ov_sents + [sent]
                    current_tok = ov_tok + sent_tok
                    start_idx = idx - len(ov_sents)
                else:
                    current_sents, current_tok, start_idx = [sent], sent_tok, idx
            else:
                current_sents.append(sent)
                current_tok += sent_tok
        if current_sents:
            chunks.append((current_sents, current_tok, start_idx))
        return chunks


TokenChunker = SemanticChunkerV2


# ══════════════════════════════════════════════════════════════
# INGESTION RERANKER
# ══════════════════════════════════════════════════════════════

class IngestionReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        if not CROSS_ENCODER_ENABLED:
            self._model = None
            return
        log.info("Loading cross-encoder: %s", model_name)
        try:
            self._model = CrossEncoder(model_name, max_length=256)
        except Exception as exc:
            log.warning("Cross-encoder failed (%s) — disabled", exc)
            self._model = None

    def filter(self, title: str, chunks: List[Dict], floor: float = RERANK_QUALITY_FLOOR) -> List[Dict]:
        if self._model is None or not chunks or not title:
            return chunks
        results, texts = [], [c["embed_text"] for c in chunks]
        for i in range(0, len(texts), 32):
            batch_texts = texts[i: i + 32]
            try:
                scores = self._model.predict([(title, t[:512]) for t in batch_texts])
            except Exception:
                results.extend(chunks[i: i + 32])
                continue
            for chunk, score in zip(chunks[i: i + 32], scores):
                if float(score) >= floor:
                    results.append(chunk)
        return results


# ══════════════════════════════════════════════════════════════
# QUALITY FILTER
# ══════════════════════════════════════════════════════════════

def quality_and_score(text: str) -> Tuple[bool, str, float]:
    stripped = text.strip()
    n = len(stripped)
    if n < MIN_CHUNK_CHARS:
        return False, f"too_short ({n})", 0.0
    low = stripped.lower()
    bp = sum(1 for w in _BOILERPLATE_SET if w in low)
    if bp / _BPLATE_DENOM > MAX_BOILERPLATE_RATIO:
        return False, f"boilerplate ({bp / _BPLATE_DENOM:.0%})", 0.0
    return True, "ok", compute_academic_score(stripped)


def chunk_fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_text(text, aggressive=True).lower().encode("utf-8")).hexdigest()[:16]


def tokenize_for_bm25(text: str) -> str:
    normed = re.sub(r"[^\w\s]", " ", normalize_text(text, aggressive=True), flags=re.UNICODE)
    return " ".join(t for t in normed.lower().split() if len(t) > 1)


# ══════════════════════════════════════════════════════════════
# JSON PARSERS & TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

def _process_table(tbl: Dict) -> Tuple[str, str, Dict]:
    headers = tbl.get("headers", [])
    rows = tbl.get("rows", [])
    clean_h = [re.sub(r"\s+", " ", str(h)).strip() for h in headers if re.sub(r"\s+", " ", str(h)).strip()]
    if not clean_h:
        return "", "", {}
    md = "| " + " | ".join(clean_h) + " |\n|" + "|".join("---" for _ in clean_h) + "|\n"
    clean_rows = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", str(row.get(h, row.get(orig, "")))).strip() for h, orig in zip(clean_h, headers)]
        if not any(cells):
            continue
        while len(cells) < len(clean_h):
            cells.append("")
        md += "| " + " | ".join(cells) + " |\n"
        clean_rows.append(dict(zip(clean_h, cells)))
    summary = f"Table with {len(clean_rows)} rows. Columns: {', '.join(clean_h[:5])}"
    return md, summary, {"headers": clean_h, "rows": clean_rows, "table_summary": summary}


def table_to_prose(structured: Dict, title: str = "") -> str:
    headers, rows = structured.get("headers", []), structured.get("rows", [])
    if not headers or not rows:
        return ""
    lines = [f"{title + '. ' if title else ''}Table: {', '.join(headers)}."]
    for i, row in enumerate(rows[:20]):
        cells = [f"{h}={row.get(h, '')}" for h in headers if row.get(h)]
        if cells:
            lines.append(f"Row {i + 1}: {', '.join(cells)}.")
    return " ".join(lines)


def extract_links(text: str) -> List[str]:
    return list(dict.fromkeys(_RE_URL_GENERAL.findall(text)))


def extract_pdf_urls(text: str, base_url: str = "") -> List[str]:
    urls = list(dict.fromkeys(_RE_PDF_URL.findall(text)))
    if base_url:
        base = base_url.rstrip("/")
        for rel in _RE_REL_PDF.findall(text):
            if not rel.startswith("http"):
                full = f"{base}/{rel.lstrip('/')}"
                if full not in urls:
                    urls.append(full)
    return urls


def remove_repeated_blocks(text: str) -> str:
    if not text:
        return text
    seen_para, para_out = set(), []
    for para in text.split("\n\n"):
        s = para.strip()
        if not s:
            para_out.append(para)
        elif s not in seen_para:
            seen_para.add(s)
            para_out.append(para)
    text = "\n\n".join(para_out)
    seen_line, line_out = set(), []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            line_out.append(line)
        elif s not in seen_line:
            seen_line.add(s)
            line_out.append(line)
    return "\n".join(line_out)


def parse_json(data: dict) -> dict:
    meta, content = data.get("metadata", {}), data.get("content", {})
    if "page" in meta:
        page, page_url = meta["page"], meta["page"].get("url", "")
        parts = [content.get("text", "")]
        for section in content.get("sections", []):
            if isinstance(section, dict):
                parts.extend([section.get("text", ""), section.get("title", "")])
        raw_text = "\n\n".join(filter(None, parts))
        combined = normalize_text(raw_text)
        tables_structured = [s for tbl in content.get("tables", []) for _, _, s in [_process_table(tbl)] if s]
        return dict(text=combined, raw_length=len(raw_text), clean_length=len(combined),
                    title=page.get("title", ""), url=page_url, file_path="", file_type="web",
                    tables=content.get("tables", []), tables_structured=tables_structured,
                    links=extract_links(raw_text), pdf_urls=extract_pdf_urls(raw_text, page_url), source="scraper")

    file_info = meta.get("file", {})
    file_path, file_type = file_info.get("path", ""), file_info.get("type", "")
    effective_source = "pdf" if file_type.lower() == "pdf" or file_path.lower().endswith(".pdf") else "extractor"
    parts = []
    if content.get("text"):
        parts.append(content["text"])
    for pg in content.get("pages", []):
        if isinstance(pg, dict) and pg.get("text"):
            parts.append(pg["text"])
    for sec in content.get("sections", []):
        if isinstance(sec, dict):
            if sec.get("title"):
                parts.append(sec["title"])
            if sec.get("text"):
                parts.append(sec["text"])
    tables, tables_structured = content.get("tables", []), []
    for tbl in tables:
        md, _, s = _process_table(tbl)
        if md:
            parts.append(md)
        if s:
            tables_structured.append(s)
    raw_text = "\n\n".join(filter(None, parts))
    combined = normalize_text(raw_text)
    title = file_info.get("name", "") or Path(file_path).stem
    return dict(text=combined, raw_length=len(raw_text), clean_length=len(combined),
                title=title, url=file_info.get("url", ""), file_path=file_path, file_type=file_type,
                tables=tables, tables_structured=tables_structured, links=extract_links(raw_text),
                pdf_urls=[], source=effective_source)


# ══════════════════════════════════════════════════════════════
# FILE COLLECTION
# ══════════════════════════════════════════════════════════════

def collect_json_files(root: Path) -> List[Tuple[Path, str, str]]:
    results = []
    for faculty_dir in sorted(root.iterdir()):
        if not faculty_dir.is_dir():
            continue
        fl = FACULTY_LABELS.get(faculty_dir.name.lower(), faculty_dir.name.upper())
        for sub in ("pages", "extracted", "tables"):
            sfolder = faculty_dir / sub
            if not sfolder.exists():
                continue
            base_str, base_len = str(sfolder), len(str(sfolder)) + 1
            for dirpath, _, filenames in os.walk(base_str):
                for fname in filenames:
                    if not fname.endswith(".json"):
                        continue
                    jf = Path(dirpath) / fname
                    rem = str(jf)[base_len:]
                    sp = rem.find(os.sep)
                    dept = (rem[:sp] if sp != -1 else "General").replace("_", " ").replace("-", " ").title()
                    results.append((jf, fl, dept))
        log.info("📂 %s → %d files", faculty_dir.name, sum(1 for r in results if r[1] == fl))
    log.info("🔎 TOTAL JSON FILES: %d", len(results))
    return results


def _doc_fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════
# SQLITE PROGRESS TRACKER
# ══════════════════════════════════════════════════════════════

class ProgressDB:
    def __init__(self, db_path: str = PROGRESS_DB):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-32000")
        self.conn.execute("CREATE TABLE IF NOT EXISTS processed (file_key TEXT PRIMARY KEY, chunks INTEGER, doc_type TEXT, language TEXT, processed_at TEXT)")
        self.conn.commit()
        self._done_set = {row[0] for row in self.conn.execute("SELECT file_key FROM processed")}
        self._pending = []
        log.info("ProgressDB: %d files already processed", len(self._done_set))

    def is_done(self, fk: str) -> bool:
        return fk in self._done_set

    def mark_done(self, fk: str, n: int, dt: str, lang: str):
        self._done_set.add(fk)
        self._pending.append((fk, n, dt, lang, datetime.utcnow().isoformat()))
        if len(self._pending) >= PROGRESS_FLUSH:
            self._flush()

    def _flush(self):
        if self._pending:
            self.conn.executemany("INSERT OR REPLACE INTO processed VALUES (?,?,?,?,?)", self._pending)
            self.conn.commit()
            self._pending.clear()

    def flush_final(self):
        self._flush()

    def reset(self):
        self.conn.execute("DELETE FROM processed")
        self.conn.commit()
        self._done_set.clear()
        self._pending.clear()

    def close(self):
        self.flush_final()
        self.conn.close()


# ══════════════════════════════════════════════════════════════
# ✦ NEO4J — ONLY LINK CHUNKS TO EXISTING GRAPH NODES
# ══════════════════════════════════════════════════════════════

def _create_document_and_chunks(
    tx,
    faculty: str,
    department: str,
    doc_title: str,
    doc_type: str,
    language: str,
    sections: List[Dict],
    chunks_batch: List[Dict],
):
    tx.run("""
        MATCH (f:Faculte {name: $faculty})
        OPTIONAL MATCH (d:Departement {name: $department})
        MERGE (doc:Document {title: $doc_title, faculty: $faculty, department: $department})
        SET doc.doc_type = $doc_type, doc.language = $language
        MERGE (f)-[:HAS_DOCUMENT]->(doc)
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:HAS_DOCUMENT]->(doc)
        )
    """, faculty=faculty, department=department, doc_title=doc_title,
        doc_type=doc_type, language=language)
    
    if sections:
        tx.run("""
            UNWIND $sections AS sec
            MATCH (doc:Document {title: $doc_title})
            MERGE (s:Section {id: sec.section_id})
            SET s.heading = sec.heading, s.start_char = sec.start_char, s.end_char = sec.end_char
            MERGE (doc)-[:HAS_SECTION]->(s)
        """, doc_title=doc_title, sections=sections)
    
    tx.run("""
        UNWIND $chunks AS ch
        MERGE (c:Chunk {id: ch.id})
        SET c.text = ch.text, c.chunk_index = ch.chunk_index,
            c.academic_score = ch.academic_score, c.authority_score = ch.authority_score,
            c.has_tables = ch.has_tables, c.token_count = ch.token_count,
            c.avg_token_idf = ch.avg_token_idf, c.section_id = ch.section_id
        WITH c, ch
        MATCH (doc:Document {title: ch.doc_title})
        MERGE (doc)-[:HAS_CHUNK {order: ch.chunk_index}]->(c)
        WITH c, ch
        OPTIONAL MATCH (s:Section {id: ch.section_id})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:HAS_CHUNK]->(c)
        )
    """, chunks=chunks_batch)


def _link_chunks_to_existing_classification(tx, chunk_ids: List[str], cls: Dict):
    if not chunk_ids:
        return
    
    if cls.get("category") == "general":
        tx.run("""
            UNWIND $ids AS cid
            MATCH (c:Chunk {id: cid})
            MATCH (g:General)
            MERGE (c)-[:BELONGS_TO_GENERAL]->(g)
        """, ids=chunk_ids)
        return
    
    validated_entities = cls.get("validated_entities", [])
    if not validated_entities:
        return
    
    by_level: Dict[str, Set[str]] = {}
    for entity in validated_entities:
        level = entity["level"]
        name = entity["name"]
        if level not in by_level:
            by_level[level] = set()
        by_level[level].add(name)
    
    level_config = {
        "annee":        ("Annee",        "BELONGS_TO_ANNEE"),
        "specialite":   ("Specialite",   "BELONGS_TO_SPECIALITE"),
        "niveau":       ("Niveau",       "BELONGS_TO_NIVEAU"),
        "filiere":      ("Filiere",      "BELONGS_TO_FILIERE"),
        "departement":  ("Departement",  "BELONGS_TO_DEPARTEMENT"),
    }
    
    for level, names in by_level.items():
        if level in level_config:
            label, relationship = level_config[level]
            names_list = list(names)
            
            try:
                tx.run(f"""
                    UNWIND $ids AS cid
                    UNWIND $names AS name
                    MATCH (c:Chunk {{id: cid}})
                    MATCH (n:{label} {{name: name}})
                    MERGE (c)-[:{relationship}]->(n)
                """, ids=chunk_ids, names=names_list)
            except Exception as exc:
                log.warning("Failed to link chunks to %s: %s", level, exc)


def _link_chunks_sequentially(tx, doc_title: str):
    tx.run("""
        MATCH (d:Document {title: $title})-[:HAS_CHUNK]->(c:Chunk)
        WITH c ORDER BY c.chunk_index
        WITH collect(c) AS ordered
        UNWIND range(0, size(ordered)-2) AS i
        WITH ordered[i] AS curr, ordered[i+1] AS nxt
        MERGE (curr)-[:NEXT_CHUNK]->(nxt)
    """, title=doc_title)


def _validate_graph_exists(tx) -> Dict[str, int]:
    result = tx.run("""
        MATCH (n)
        WITH labels(n) AS lbls, count(*) AS cnt
        UNWIND lbls AS lbl
        RETURN lbl, cnt
    """)
    return {record["lbl"]: record["cnt"] for record in result}


# ══════════════════════════════════════════════════════════════
# SEMANTIC DEDUPLICATION
# ══════════════════════════════════════════════════════════════

def apply_semantic_dedup_doc(embeddings: np.ndarray, chunk_texts: List[str],
                             threshold: float = SEMANTIC_DEDUP_THRESHOLD,
                             window: int = SEMANTIC_DEDUP_WINDOW) -> List[int]:
    n, keep_mask = len(embeddings), [True] * len(embeddings)
    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, min(i + window + 1, n)):
            if not keep_mask[j]:
                continue
            if float(np.dot(embeddings[i], embeddings[j])) >= threshold:
                if len(chunk_texts[i]) >= len(chunk_texts[j]):
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    kept = [idx for idx, keep in enumerate(keep_mask) if keep]
    if len(kept) < n:
        log.info("Semantic dedup: %d → %d", n, len(kept))
    return kept


# ══════════════════════════════════════════════════════════════
# PARSE + CHUNK WORKER
# ══════════════════════════════════════════════════════════════

def _parse_and_chunk_file(
    args: Tuple[Path, str, str, SemanticChunkerV2, AdaptiveChunkSizer,
                IngestionReranker, Any, GraphAwareClassifier]
) -> Optional[Dict]:
    jf, faculty, department, chunker, sizer, reranker, split_model, classifier = args
    file_key = f"{faculty}/{department}/{jf.name}"

    try:
        with open(jf, "r", encoding="utf-8") as fh:
            raw = _load_json(fh)
        parsed = parse_json(raw)
    except Exception as exc:
        log.warning("⚠  Parse error [%s]: %s", jf.name, exc)
        return {"fail": True, "file_key": file_key}

    if len(parsed["text"].strip()) < MIN_DOC_CHARS:
        return {"skip": True, "file_key": file_key}

    title = parsed["title"] or jf.stem
    language = detect_language(parsed["text"])
    doc_type = infer_doc_type(jf.name, title, parsed["text"])
    year = extract_year(parsed["text"])
    fp = _doc_fingerprint(parsed["text"])
    auth_score = compute_authority_score(parsed["text"], doc_type, parsed["source"])
    alt_titles = generate_title_aliases(title)

    classification = classifier.classify(
        text=parsed["text"], title=title,
        faculty_hint=faculty, department_hint=department,
    )
    log.debug("Classification for %s: method=%s", jf.name, classification.get("match_method"))

    sample_kd, sample_idf = compute_keyword_density(parsed["text"][:1000])
    adaptive_tokens = sizer.compute(doc_type, sample_kd, sample_idf)
    sections = detect_sections(parsed["text"], fp)

    extra_chunks = []
    for structured in parsed.get("tables_structured", []):
        prose = table_to_prose(structured, title=title)
        if prose and len(prose) >= MIN_CHUNK_CHARS:
            extra_chunks.append({
                "embed_text": prose, "text": prose, "clean_body": prose,
                "token_count": len(prose) // 4, "chunk_index": -1,
                "section_id": sections[0]["section_id"] if sections else "",
                "start_char": 0, "prev_hint": "", "next_hint": "",
            })

    raw_chunks = chunker.split(parsed["text"], title, split_model=split_model,
                               chunk_tokens=adaptive_tokens, doc_fp=fp, sections=sections)
    all_raw = raw_chunks + [{**c, "chunk_index": len(raw_chunks) + i} for i, c in enumerate(extra_chunks)]

    if not all_raw:
        return {"skip": True, "file_key": file_key}

    all_raw = reranker.filter(title, all_raw)
    if not all_raw:
        return {"skip": True, "file_key": file_key}

    quality_counts = {"too_short": 0, "boilerplate": 0, "low_academic": 0}
    good_chunks, seen_chunk_fps = [], set()

    for chunk_dict in all_raw:
        embed_text = chunk_dict.get("embed_text", chunk_dict["text"])
        ok_flag, reason, acad_score = quality_and_score(embed_text)
        if not ok_flag:
            if "too_short" in reason:
                quality_counts["too_short"] += 1
            if "boilerplate" in reason:
                quality_counts["boilerplate"] += 1
            continue
        if acad_score < 0.05:
            quality_counts["low_academic"] += 1

        cfp = chunk_fingerprint(embed_text)
        if cfp in seen_chunk_fps:
            continue
        seen_chunk_fps.add(cfp)

        clean_body = chunk_dict.get("clean_body", embed_text)
        kd, avg_idf = compute_keyword_density(clean_body)
        entities = extract_entities(chunk_dict["text"])
        tokenized_str = tokenize_for_bm25(clean_body)

        chunk_dict.update({
            "academic_score": acad_score, "authority_score": auth_score,
            "tokenized_text": tokenized_str, "keyword_density": kd,
            "avg_token_idf": avg_idf, "entities": entities, "chunk_fp": cfp,
        })
        good_chunks.append(chunk_dict)

    if not good_chunks:
        return {"skip": True, "file_key": file_key}

    return {
        "ok": True, "file_key": file_key, "jf": jf,
        "faculty": faculty, "department": department,
        "title": title, "language": language, "doc_type": doc_type,
        "year": year, "fp": fp, "good_chunks": good_chunks,
        "has_tables": bool(parsed.get("tables")),
        "tables_structured": parsed.get("tables_structured", []),
        "source": parsed["source"], "url": parsed.get("url", ""),
        "file_path": parsed.get("file_path", ""), "links": parsed.get("links", []),
        "pdf_urls": parsed.get("pdf_urls", []),
        "raw_length": parsed.get("raw_length", 0),
        "clean_length": parsed.get("clean_length", 0),
        "authority_score": auth_score, "alternate_titles": alt_titles,
        "sections": sections, "quality_counts": quality_counts,
        "classification": classification,
    }


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def build_pipeline(resume: bool = True, clear_chroma: bool = False):

    log.info("Loading embedding model: %s", EMBED_MODEL)
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"
    log.info("E5 device: %s", _device)
    embed_model = SentenceTransformer(EMBED_MODEL, device=_device)

    log.info("Loading sentence split model: %s (CPU)", SENT_SPLIT_MODEL)
    split_model = SentenceTransformer(SENT_SPLIT_MODEL, device="cpu")

    alias_resolver = AliasResolver(ALIASES_PATH)
    university_structure = load_university_structure(UNIVERSITY_STRUCTURE_PATH)
    graph_index = GraphAwareIndex(university_structure)
    classifier = GraphAwareClassifier(graph_index, alias_resolver)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    if clear_chroma:
        try:
            chroma_client.delete_collection("university_data")
            log.info("🗑️  ChromaDB cleared")
        except Exception:
            pass
    collection = chroma_client.get_or_create_collection(
        name="university_data", metadata={"hnsw:space": "cosine"})

    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with neo4j_driver.session() as validation_session:
        graph_counts = validation_session.execute_write(_validate_graph_exists)
    log.info("📊 Existing graph: %s", graph_counts)
    
    expected = ["Faculte", "Departement", "Niveau", "Specialite", "General"]
    missing = [n for n in expected if graph_counts.get(n, 0) == 0]
    if missing:
        log.warning("⚠️  Missing node types: %s — run structure import first!", missing)

    chunker = SemanticChunkerV2()
    sizer = AdaptiveChunkSizer()
    reranker = IngestionReranker()
    progress = ProgressDB(PROGRESS_DB)
    if not resume:
        log.info("🔥 Fresh start")
        progress.reset()

    root = Path(ROOT_FOLDER)
    all_files = collect_json_files(root)
    json_files = [(jf, f, d) for jf, f, d in all_files
                  if not (resume and progress.is_done(f"{f}/{d}/{jf.name}"))]
    log.info("📂 %d / %d files to process", len(json_files), len(all_files))

    ok = skip = fail = 0
    total_chunks = 0
    quality_stats = {"too_short": 0, "boilerplate": 0, "low_academic": 0}
    metadata_store = []
    seen_doc_fps, seen_chunk_fps, docs_to_link = set(), set(), []
    cls_stats = {}

    log.info("🚀 Parsing & chunking (%d workers) …", PARSE_WORKERS)
    worker_args = [(jf, f, d, chunker, sizer, reranker, split_model, classifier)
                   for jf, f, d in json_files]
    parsed_results = []

    with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as pool:
        futures = {pool.submit(_parse_and_chunk_file, arg): arg for arg in worker_args}
        for future in as_completed(futures):
            res = future.result()
            if res is None or res.get("fail"):
                fail += 1
                continue
            if res.get("skip"):
                skip += 1
                continue
            fp = res.get("fp", "")
            if fp and fp in seen_doc_fps:
                skip += 1
                continue
            if fp:
                seen_doc_fps.add(fp)
            parsed_results.append(res)

    log.info("Parse done: %d docs, %d skipped, %d failed", len(parsed_results), skip, fail)

    flat_embed, flat_text, flat_mapping = [], [], []
    for doc_idx, res in enumerate(parsed_results):
        for local_idx, cd in enumerate(res["good_chunks"]):
            flat_embed.append(cd.get("embed_text", cd["text"]))
            flat_text.append(cd["text"])
            flat_mapping.append((doc_idx, local_idx))

    MAX_CHUNKS = 50_000
    if len(flat_embed) > MAX_CHUNKS:
        log.warning("⚠  Truncating to %d chunks", MAX_CHUNKS)
        flat_embed, flat_text, flat_mapping = flat_embed[:MAX_CHUNKS], flat_text[:MAX_CHUNKS], flat_mapping[:MAX_CHUNKS]

    log.info("🧮 Embedding %d chunks …", len(flat_embed))
    all_embs_np = embed_model.encode(
        [PASSAGE_PREFIX + t for t in flat_embed],
        batch_size=EMBED_BATCH, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True)
    log.info("✅ Embedding complete: %s", str(all_embs_np.shape))

    doc_emb_slices = []
    if flat_mapping:
        current_doc, current_start = flat_mapping[0][0], 0
        for g_idx, (doc_idx, _) in enumerate(flat_mapping):
            if doc_idx != current_doc:
                doc_emb_slices.append((current_start, g_idx))
                current_start, current_doc = g_idx, doc_idx
        doc_emb_slices.append((current_start, len(flat_mapping)))

    with neo4j_driver.session() as neo4j_session:

        for doc_idx, res in enumerate(parsed_results):
            faculty = res["faculty"]
            department = res["department"]
            title = res["title"]
            language = res["language"]
            doc_type = res["doc_type"]
            year = res["year"]
            fp = res["fp"]
            good_chunks = res["good_chunks"]
            has_tables = res["has_tables"]
            source = res["source"]
            url = res.get("url", "")
            file_key = res["file_key"]
            jf = res["jf"]
            file_path = res.get("file_path", "")
            links = res.get("links", [])
            pdf_urls = res.get("pdf_urls", [])
            tables_structured = res.get("tables_structured", [])
            raw_length = res.get("raw_length", 0)
            clean_length = res.get("clean_length", 0)
            auth_score = res.get("authority_score", 0.0)
            alt_titles = res.get("alternate_titles", [])
            sections = res.get("sections", [])
            cls = res.get("classification", {})

            for k, v in res.get("quality_counts", {}).items():
                quality_stats[k] += v
            
            method = cls.get("match_method", "none")
            cls_stats[method] = cls_stats.get(method, 0) + 1

            emb_start, emb_end = doc_emb_slices[doc_idx]
            doc_embs_np = all_embs_np[emb_start:emb_end]

            doc_embed_texts = [cd.get("embed_text", cd["text"]) for cd in good_chunks]
            kept_local = set(apply_semantic_dedup_doc(doc_embs_np, doc_embed_texts))

            chroma_records, doc_meta_rows, neo4j_chunk_buf = [], [], []
            embed_ptr = 0

            for local_idx, chunk_dict in enumerate(good_chunks):
                if local_idx not in kept_local:
                    embed_ptr += 1
                    continue

                cfp = chunk_dict.get("chunk_fp", chunk_fingerprint(chunk_dict.get("embed_text", chunk_dict["text"])))
                if cfp in seen_chunk_fps:
                    embed_ptr += 1
                    continue
                seen_chunk_fps.add(cfp)

                cid = f"{fp}_c{local_idx}"
                embed_text = chunk_dict.get("embed_text", chunk_dict["text"])
                stored_text = chunk_dict["text"]
                clean_body = chunk_dict.get("clean_body", embed_text)
                section_id = chunk_dict.get("section_id", "")
                prev_hint = chunk_dict.get("prev_hint", "")
                next_hint = chunk_dict.get("next_hint", "")
                kd = chunk_dict.get("keyword_density", 0.0)
                avg_idf = chunk_dict.get("avg_token_idf", 0.0)
                entities = chunk_dict.get("entities", {})
                tokenized = chunk_dict.get("tokenized_text", "")
                chunk_has_t = has_tables and ("|" in stored_text or bool(tables_structured))

                chroma_records.append({
                    "id": cid, "text": embed_text, "_emb_ptr": embed_ptr,
                    "metadata": {
                        "faculty": faculty, "department": department, "language": language,
                        "source": source, "doc_type": doc_type, "chunk_index": local_idx,
                        "total_chunks": len(good_chunks), "has_table": chunk_has_t,
                        "academic_score": round(chunk_dict["academic_score"], 3),
                        "authority_score": round(auth_score, 3),
                        "keyword_density": round(kd, 4), "avg_token_idf": round(avg_idf, 4),
                        "year": year or 0, "chunk_len": len(embed_text), "url": url,
                        "has_email": bool(entities.get("emails")),
                        "has_course_code": bool(entities.get("course_codes")),
                        "has_phone": bool(entities.get("phones")),
                        "section_id": section_id,
                        "category": cls.get("category", "general"),
                        "match_method": cls.get("match_method", "none"),
                        "multi_entity": cls.get("multi_entity", False),
                        "annees": ",".join(cls.get("annees", [])),
                        "specialites": ",".join(cls.get("specialites", [])),
                        "niveaux": ",".join(cls.get("niveaux", [])),
                        "filieres": ",".join(cls.get("filieres", [])),
                        "departements": ",".join(cls.get("departements", [])),
                        "matched_keywords": ",".join(cls.get("matched_keywords", [])),
                        "faculte": cls.get("faculte") or "",
                        "departement": cls.get("departement") or "",
                        "filiere": cls.get("filiere") or "",
                        "niveau": cls.get("niveau") or "",
                        "specialite": cls.get("specialite") or "",
                        "annee": cls.get("annee") or "",
                    },
                })

                neo4j_chunk_buf.append({
                    "id": cid, "text": embed_text, "chunk_index": local_idx,
                    "academic_score": round(chunk_dict["academic_score"], 3),
                    "authority_score": round(auth_score, 3), "has_tables": has_tables,
                    "token_count": chunk_dict.get("token_count", 0),
                    "avg_token_idf": round(avg_idf, 4), "section_id": section_id,
                    "doc_title": title,
                })

                doc_meta_rows.append({
                    "chunk_id": cid, "file": jf.name, "faculty": faculty,
                    "department": department, "title": title, "alternate_titles": alt_titles,
                    "source": source, "url": url, "file_path": file_path, "pdf_urls": pdf_urls,
                    "language": language, "doc_type": doc_type, "chunk": stored_text,
                    "embed_text": embed_text, "clean_text": clean_body,
                    "tokenized_text": tokenized, "prev_hint": prev_hint, "next_hint": next_hint,
                    "raw_length": raw_length, "clean_length": clean_length,
                    "academic_score": round(chunk_dict["academic_score"], 3),
                    "authority_score": round(auth_score, 3),
                    "keyword_density": round(kd, 4), "avg_token_idf": round(avg_idf, 4),
                    "chunk_index": local_idx, "year": year, "links": links,
                    "tables_structured": tables_structured, "entities": entities,
                    "section_id": section_id, "classification": cls,
                })
                embed_ptr += 1

            metadata_store.extend(doc_meta_rows)

            surviving_ptrs = [r["_emb_ptr"] for r in chroma_records]
            if surviving_ptrs:
                surviving_embs = doc_embs_np[surviving_ptrs]
                for bs in range(0, len(chroma_records), CHROMA_BATCH):
                    batch = chroma_records[bs: bs + CHROMA_BATCH]
                    lo, hi = bs, bs + len(batch)
                    collection.upsert(
                        ids=[r["id"] for r in batch],
                        documents=[r["text"] for r in batch],
                        embeddings=surviving_embs[lo:hi].tolist(),
                        metadatas=[r["metadata"] for r in batch])

            if neo4j_chunk_buf:
                for i in range(0, len(neo4j_chunk_buf), NEO4J_BATCH):
                    batch = neo4j_chunk_buf[i: i + NEO4J_BATCH]
                    neo4j_session.execute_write(
                        _create_document_and_chunks, faculty, department, title,
                        doc_type, language, sections, batch)

            chunk_ids = [r["id"] for r in chroma_records]
            if chunk_ids:
                neo4j_session.execute_write(_link_chunks_to_existing_classification, chunk_ids, cls)

            docs_to_link.append(title)
            progress.mark_done(file_key, len(doc_meta_rows), doc_type, language)
            total_chunks += len(doc_meta_rows)
            ok += 1

            cls_summary = cls.get("match_method", "none")
            if cls.get("specialite"):
                cls_summary += f":{cls['specialite'][:25]}"
            elif cls.get("niveau"):
                cls_summary += f":{cls['niveau']}"
            elif cls.get("category") == "general":
                cls_summary = "general"
            
            log.info("✅ [%-20s / %-15s] %s → %d chunks (cls=%s)",
                     faculty[:20], department[:15], jf.name, len(doc_meta_rows), cls_summary)

        if docs_to_link:
            log.info("🔗 NEXT_CHUNK edges for %d docs …", len(docs_to_link))
            for t in docs_to_link:
                neo4j_session.execute_write(_link_chunks_sequentially, t)

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata_store, fh, ensure_ascii=False, indent=2)

    progress.close()
    neo4j_driver.close()

    log.info("\n" + "─" * 70)
    log.info("  PIPELINE v6 COMPLETE")
    log.info("   SUCCESS: %d  SKIPPED: %d  FAILED: %d", ok, skip, fail)
    log.info("   CHUNKS: %d", total_chunks)
    log.info("  Quality: too_short=%d boilerplate=%d low_academic=%d",
             quality_stats["too_short"], quality_stats["boilerplate"], quality_stats["low_academic"])
    log.info("  Classification:")
    for m, c in cls_stats.items():
        if c:
            log.info("    %-30s: %d", m, c)
    log.info("─" * 70)
    return ok, skip, fail


if __name__ == "__main__":
    import sys
    fresh = "--fresh" in sys.argv
    clear_chroma = "--clear-chroma" in sys.argv
    if fresh:
        log.info("🔥 --fresh: resetting progress")
    if clear_chroma:
        log.info("🗑️  --clear-chroma: wiping ChromaDB")
    build_pipeline(resume=not fresh, clear_chroma=clear_chroma)