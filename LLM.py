
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Set, Tuple

from groq import Groq

# ── RAG v16 public API ────────────────────────────────────────
import rag as rag
from rag import (
    QueryAnalyzer,
    retrieve_for_llm,
    traverse_graph,
)

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
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_CONTEXT_CHARS  = 8_000
GENERATION_TIMEOUT = 60
TOP_K_RETRIEVE     = 5

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set — generation will fail.")

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class Source:
    title:       str
    url:         str
    pdf_url:     str   = ""
    source_type: str   = "page"
    chunk_score: float = 0.0


@dataclass
class GenerationResult:
    answer:        str
    intent:        str
    language:      str
    sources:       List[Source]
    fallback_used: bool = False


# ─────────────────────────────────────────────────────────────
# LANGUAGE CONFIG
# ─────────────────────────────────────────────────────────────

LANG_CONFIG: Dict[str, Dict[str, str]] = {
    "ar": {
        "name":        "Arabic",
        "instruction": "أجب باللغة العربية فقط.",
        "no_data":     "عذراً، لا توجد معلومات موثوقة حول هذا الموضوع في قاعدة البيانات المتاحة.",
        "graph_intro": "فيما يلي هيكل الكلية / القسم المطلوب:",
    },
    "fr": {
        "name":        "French",
        "instruction": "Répondez UNIQUEMENT en français.",
        "no_data":     "Désolé, aucune information fiable sur ce sujet n'a été trouvée dans les données disponibles.",
        "graph_intro": "Voici la structure de la faculté / département demandé :",
    },
    "en": {
        "name":        "English",
        "instruction": "Respond ONLY in English.",
        "no_data":     "Sorry, no reliable information about this topic was found in the available data.",
        "graph_intro": "Here is the requested faculty / department structure:",
    },
}

_RE_AR = re.compile(r"[\u0600-\u06FF]")
_RE_FR = re.compile(
    r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|est)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    s = text[:200]
    if len(_RE_AR.findall(s)) > 3:
        return "ar"
    if len(_RE_FR.findall(s)) > 2:
        return "fr"
    return "en"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    q = query.strip().strip('"\'«»')
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"[?！？]+$", "", q).strip()
    return q


def truncate_context(llm_context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(llm_context) <= max_chars:
        return llm_context
    truncated = llm_context[:max_chars]
    last_sep  = truncated.rfind("\n─")
    if last_sep > max_chars * 0.5:
        return truncated[:last_sep]
    return truncated + "\n\n[Context truncated...]"


def _format_graph_result(graph_data: Dict, lang: str) -> str:
    """
    Convert the dict returned by rag.traverse_graph() into readable Markdown.
    """
    intro  = LANG_CONFIG.get(lang, LANG_CONFIG["en"])["graph_intro"]
    lines  = [intro, ""]
    nodes  = graph_data.get("graph_results", [])

    def _render(item: Dict, indent: int = 0) -> None:
        prefix = "  " * indent + "- "
        node   = item.get("node", item)
        label  = node.get("label", "")
        name   = node.get("name", "")
        lines.append(f"{prefix}**[{label}]** {name}")
        for child in item.get("children", []):
            _render(child, indent + 1)

    for item in nodes:
        _render(item)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# SOURCE EXTRACTION — updated for RAG v16 flat chunk format
# ─────────────────────────────────────────────────────────────

def _extract_sources(results: List[Dict]) -> List[Source]:
    """
    RAG v16 returns flat dicts.  All fields are top-level:
      chunk["url"], chunk["pdf_url"], chunk["title"], chunk["source_type"], …
    """
    sources:   List[Source] = []
    seen_urls: Set[str]     = set()

    for r in results:
        # RAG v16: fields are top-level (no nested metadata)
        source_type = r.get("source_type", "page")
        pdf_url     = r.get("pdf_url",  "")
        url         = r.get("url",      "")
        title       = r.get("title",    "Document")

        # Skip graph_result synthetic chunks
        if r.get("chunk_id") == "graph_result":
            continue

        canonical = url or pdf_url
        if not canonical or canonical in seen_urls:
            continue

        seen_urls.add(canonical)
        sources.append(Source(
            title       = title,
            url         = url,
            pdf_url     = pdf_url,
            source_type = source_type,
            chunk_score = r.get("score", 0.0),
        ))

    return sources


def _is_document_only(results: List[Dict]) -> Tuple[bool, str, str]:
    """Detect when the top result is a bare PDF link with no extractable text."""
    if not results:
        return False, "", ""
    top         = results[0]
    source_type = top.get("source_type", "")
    text        = top.get("text", "").strip()
    pdf_url     = top.get("pdf_url", "")
    url         = top.get("url",     "")

    if source_type == "pdf" and len(text) < 200 and pdf_url:
        return True, url, pdf_url
    return False, "", ""


def _format_sources_md(sources: List[Source], lang: str) -> str:
    if not sources:
        return ""
    labels = {"ar": "📚 المصادر", "fr": "📚 Sources", "en": "📚 Sources"}
    lines  = [f"\n\n{labels.get(lang, '📚 Sources')}"]
    for s in sources:
        if s.source_type == "pdf" and s.pdf_url:
            if s.url and s.url != s.pdf_url:
                lines.append(f"- [{s.title}]({s.url})  ·  [⬇ PDF]({s.pdf_url})")
            else:
                lines.append(f"- [{s.title}]({s.pdf_url})  ⬇ PDF")
        else:
            display_url = s.url or s.pdf_url
            if display_url:
                lines.append(f"- [{s.title}]({display_url})")
            else:
                lines.append(f"- {s.title}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLARIFICATION ENGINE
# ─────────────────────────────────────────────────────────────

_SLOT_DEFS: Dict[str, Dict] = {
    "person_name": {
        "label": {
            "en": "the person's full name (or partial name / role)",
            "fr": "le nom complet de la personne (ou nom partiel / fonction)",
            "ar": "الاسم الكامل للشخص (أو اسم جزئي / وظيفة)",
        },
        "patterns": [
            r"(?:name(?:\s+is)?|called|named|prof(?:esseur)?\.?\s+|dr\.?\s+|docteur\s+)"
            r"([A-ZÀ-Öa-zà-ö][A-ZÀ-Öa-zà-ö\s\-\.]{2,})",
        ],
        "intents": {"person_lookup"},
    },
    "faculty": {
        "label": {
            "en": "the faculty (e.g. Faculty of Science, Faculty of Technology…)",
            "fr": "la faculté (ex. Faculté des Sciences, Faculté de Technologie…)",
            "ar": "الكلية (مثل كلية العلوم، كلية التكنولوجيا…)",
        },
        "patterns": [
            r"(?:faculty\s+of|faculté\s+(?:de|des?|d')|كلية\s+)"
            r"([A-ZÀ-Öa-zà-öء-ي][A-ZÀ-Öa-zà-öء-ي\s\-\.]{2,})",
        ],
        "intents": {"planning", "course_material", "course_query"},
    },
    "department": {
        "label": {
            "en": "the department (e.g. Computer Science, Physics…)",
            "fr": "le département (ex. Informatique, Physique…)",
            "ar": "القسم (مثل الإعلام الآلي، الفيزياء…)",
        },
        "patterns": [
            r"(?:department\s+of|dept\.?\s+of|département\s+(?:de|d')|قسم\s+)"
            r"([A-ZÀ-Öa-zà-öء-ي][A-ZÀ-Öa-zà-öء-ي\s\-\.]{2,})",
        ],
        "intents": {"planning", "course_material", "course_query"},
    },
    "level": {
        "label": {
            "en": "the academic level (e.g. L1, L2, L3, M1, M2, Doctorate…)",
            "fr": "le niveau académique (ex. L1, L2, L3, M1, M2, Doctorat…)",
            "ar": "المستوى الدراسي (مثل س1، س2، س3، م1، م2، دكتوراه…)",
        },
        "patterns": [
            r"\b(L[123]|M[12]|Doctorate|PhD|Licence\s+[123]|Master\s+[12]"
            r"|ليسانس\s*[١-٣123]|ماستر\s*[١-٢12]|دكتوراه)\b",
        ],
        "intents": {"planning", "course_material", "course_query"},
    },
    "program": {
        "label": {
            "en": "the program / specialty (e.g. Software Engineering, Networks…)",
            "fr": "la spécialité / filière (ex. Génie Logiciel, Réseaux…)",
            "ar": "التخصص / الشعبة (مثل هندسة البرمجيات، الشبكات…)",
        },
        "patterns": [],
        "intents": {"planning", "course_material", "course_query"},
    },
    "year": {
        "label": {
            "en": "the academic year (e.g. 2024-2025)",
            "fr": "l'année universitaire (ex. 2024-2025)",
            "ar": "السنة الجامعية (مثل 2024-2025)",
        },
        "patterns": [
            r"\b(20\d{2}[-/]20\d{2})\b",
            r"\b(20\d{2}[-/]\d{2})\b",
        ],
        "intents": {"planning", "course_material", "course_query"},
    },
    "course_name": {
        "label": {
            "en": "the full course name (e.g. Advanced Algorithms)",
            "fr": "l'intitulé complet du cours (ex. Algorithmes Avancés)",
            "ar": "الاسم الكامل للمقياس (مثل الخوارزميات المتقدمة)",
        },
        "patterns": [
            r"(?:course(?:\s+(?:called|named|is))?|module|مقياس|cours(?:\s+de)?)\s*[:\-]?\s*"
            r"([A-ZÀ-Öa-zà-öء-ي][A-ZÀ-Öa-zà-öء-ي\s\-\(\)]{3,})",
        ],
        "intents": {"course_material"},
    },
    "course_code": {
        "label": {
            "en": "the course code (e.g. INF301, MATH202…)",
            "fr": "le code du cours (ex. INF301, MATH202…)",
            "ar": "رمز المقياس (مثل INF301، MATH202…)",
        },
        "patterns": [
            r"\b([A-Z]{2,5}\s*[-_]?\s*\d{2,4}[A-Z]?)\b",
        ],
        "intents": {"course_material"},
    },
}

REQUIRED_SLOTS: Dict[str, List[str]] = {
    "person_lookup":   ["person_name"],
    "planning":        ["faculty", "department", "level", "program", "year"],
    "course_material": ["faculty", "department", "level", "program", "year",
                        "course_name", "course_code"],
}

_QUESTIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "person_name":  "👤 Who are you looking for? Please provide their full name, partial name, or role (e.g. \"Prof. Benali\" or \"head of the CS department\").",
        "faculty":      "🏛️ Which **faculty** does this relate to? (e.g. *Faculty of Science*, *Faculty of Technology*)",
        "department":   "📂 Which **department**? (e.g. *Computer Science*, *Mathematics*, *Physics*)",
        "level":        "🎓 What **academic level**? (e.g. L1, L2, L3, M1, M2, Doctorate)",
        "program":      "📘 What **program / specialty**? (e.g. *Software Engineering*, *Networks & Telecom*)",
        "year":         "📅 Which **academic year**? (e.g. *2024-2025*)",
        "course_name":  "📖 What is the **full name of the course**? (e.g. *Advanced Algorithms*, *Digital Signal Processing*)",
        "course_code":  "🔢 What is the **course code**? (e.g. *INF301*, *MATH202*) — type `skip` if unknown.",
    },
    "fr": {
        "person_name":  "👤 Qui recherchez-vous ? Donnez le nom complet, partiel ou la fonction (ex. « Prof. Benali » ou « chef du département informatique »).",
        "faculty":      "🏛️ Quelle **faculté** est concernée ? (ex. *Faculté des Sciences*, *Faculté de Technologie*)",
        "department":   "📂 Quel **département** ? (ex. *Informatique*, *Mathématiques*, *Physique*)",
        "level":        "🎓 Quel **niveau académique** ? (ex. L1, L2, L3, M1, M2, Doctorat)",
        "program":      "📘 Quelle **spécialité / filière** ? (ex. *Génie Logiciel*, *Réseaux & Télécoms*)",
        "year":         "📅 Quelle **année universitaire** ? (ex. *2024-2025*)",
        "course_name":  "📖 Quel est l'**intitulé complet du cours** ? (ex. *Algorithmes Avancés*, *Traitement du Signal*)",
        "course_code":  "🔢 Quel est le **code du cours** ? (ex. *INF301*, *MATH202*) — tapez `skip` si inconnu.",
    },
    "ar": {
        "person_name":  "👤 من تبحث عنه؟ يُرجى تقديم الاسم الكامل أو الجزئي أو المنصب (مثل «الأستاذ بن علي» أو «رئيس قسم الإعلام الآلي»).",
        "faculty":      "🏛️ ما **الكلية** المعنية؟ (مثل *كلية العلوم*، *كلية التكنولوجيا*)",
        "department":   "📂 ما **القسم**؟ (مثل *الإعلام الآلي*، *الرياضيات*، *الفيزياء*)",
        "level":        "🎓 ما **المستوى الدراسي**؟ (مثل ل1، ل2، ل3، م1، م2، دكتوراه)",
        "program":      "📘 ما **التخصص / الشعبة**؟ (مثل *هندسة البرمجيات*، *الشبكات والاتصالات*)",
        "year":         "📅 ما **السنة الجامعية**؟ (مثل *2024-2025*)",
        "course_name":  "📖 ما **الاسم الكامل للمقياس**؟ (مثل *الخوارزميات المتقدمة*، *معالجة الإشارات*)",
        "course_code":  "🔢 ما **رمز المقياس**؟ (مثل *INF301*، *MATH202*) — اكتب `skip` إذا كنت لا تعرفه.",
    },
}


@dataclass
class ClarificationState:
    intent:         str
    language:       str
    original_query: str
    slots:          Dict[str, str] = field(default_factory=dict)
    pending:        List[str]      = field(default_factory=list)
    awaiting:       Optional[str]  = None
    done:           bool           = False


class ClarificationEngine:

    @staticmethod
    def init(query: str, intent: str, lang: str) -> ClarificationState:
        required = REQUIRED_SLOTS.get(intent, [])
        state    = ClarificationState(
            intent         = intent,
            language       = lang,
            original_query = query,
            pending        = list(required),
        )
        ClarificationEngine._extract_slots(state, query)
        return state

    @staticmethod
    def needs_clarification(state: ClarificationState) -> bool:
        return bool(state.pending) and not state.done

    @staticmethod
    def next_question(state: ClarificationState) -> str:
        if not state.pending:
            state.done = True
            return ""
        slot           = state.pending[0]
        state.awaiting = slot
        lang           = state.language
        questions      = _QUESTIONS.get(lang, _QUESTIONS["en"])
        return questions.get(slot, f"Please provide: {slot}")

    @staticmethod
    def ingest_reply(state: ClarificationState, reply: str) -> None:
        slot    = state.awaiting
        if slot is None:
            ClarificationEngine._extract_slots(state, reply)
            return
        stripped = reply.strip()
        if stripped.lower() in {"skip", "passer", "تخطي", "تجاوز", "-", "n/a"}:
            state.slots[slot] = ""
        else:
            extracted         = ClarificationEngine._try_extract(slot, stripped)
            state.slots[slot] = extracted or stripped
        if slot in state.pending:
            state.pending.remove(slot)
        state.awaiting = None
        if not state.pending:
            state.done = True

    @staticmethod
    def build_enriched_query(state: ClarificationState) -> str:
        parts  = [state.original_query]
        labels = {
            "person_name": "person",
            "faculty":     "faculty",
            "department":  "department",
            "level":       "level",
            "program":     "program / specialty",
            "year":        "academic year",
            "course_name": "course",
            "course_code": "course code",
        }
        for slot, value in state.slots.items():
            if value:
                parts.append(f"{labels.get(slot, slot)}: {value}")
        return " | ".join(parts)

    @staticmethod
    def _extract_slots(state: ClarificationState, text: str) -> None:
        for slot_key, slot_def in _SLOT_DEFS.items():
            if slot_key not in state.pending:
                continue
            if slot_key in state.slots:
                continue
            if state.intent not in slot_def.get("intents", set()):
                continue
            extracted = ClarificationEngine._try_extract(slot_key, text)
            if extracted:
                state.slots[slot_key] = extracted
                state.pending.remove(slot_key)

    @staticmethod
    def _try_extract(slot_key: str, text: str) -> Optional[str]:
        slot_def = _SLOT_DEFS.get(slot_key, {})
        for pattern in slot_def.get("patterns", []):
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None


# ─────────────────────────────────────────────────────────────
# INTENT-AWARE PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

class PromptBuilder:
    def __init__(self, intent: str, lang: str):
        self.intent = intent
        self.lang   = lang
        self.cfg    = LANG_CONFIG.get(lang, LANG_CONFIG["en"])

    def build(self, query: str, context: str) -> Tuple[str, str]:
        return self._system(), self._user(query, context)

    def _system(self) -> str:
        base = (
            "You are an expert university assistant for Farhat Abbas University Sétif 1.\n"
            "CRITICAL RULES:\n"
            f"1. {self.cfg['instruction']}\n"
            "2. GROUNDING: Base your answer EXACTLY on the provided context. "
            "Do not hallucinate or use outside knowledge.\n"
            "3. FORMAT: Use Markdown (bold, lists, tables where useful).\n"
            "4. LINKS: If a chunk contains 'Source:' or 'PDF:' followed by a URL, "
            "include that link naturally in your answer as a clickable Markdown link."
        )
        extras: Dict[str, str] = {
            "person_lookup": (
                "\n5. Extract: Full Name, Title (Prof/Dr), Department, Faculty, "
                "Email, Office/Bureau, Phone. Format as a profile card. "
                "Omit fields that are missing — do not guess."
            ),
            "course_query": (
                "\n5. Extract: Course Name, Department, Credits, Teacher(s), "
                "Syllabus/Description if available."
            ),
            "course_material": (
                "\n5. Extract: Course Name, Code, Credits, Teacher(s), "
                "Syllabus, required textbooks, and any downloadable resources. "
                "Format clearly with headings."
            ),
            "admin_query": (
                "\n5. Extract precise steps, deadlines, required documents, "
                "and relevant office contacts. Use bullet points for procedures."
            ),
            "planning": (
                "\n5. Present the schedule / planning as a structured table "
                "with columns: Date, Time, Subject/Module, Room, Examiner (if available). "
                "Group by faculty/department/level as requested."
            ),
            "translation": (
                f"\n5. Translate the provided text into {self.cfg['name']}. "
                "Maintain original meaning, tone, and structure."
            ),
            "table_query": (
                "\n5. Format the requested list or table strictly as "
                "a Markdown table or bulleted list."
            ),
            "graph_query": (
                "\n5. The context contains a JSON hierarchy tree. "
                "Format it as a clean Markdown outline: use headings for faculties, "
                "bullet points for departments, sub-bullets for programs/specializations."
            ),
        }
        return base + extras.get(self.intent, "")

    def _user(self, query: str, context: str) -> str:
        cfg = LANG_CONFIG.get(self.lang, LANG_CONFIG["en"])
        if not context or context == "No relevant context available.":
            return (
                f"CONTEXT: Empty\n\n"
                f"QUESTION: {query}\n\n"
                f"ANSWER: {cfg['no_data']}"
            )
        return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"


# ─────────────────────────────────────────────────────────────
# GROQ STREAMING WRAPPER
# ─────────────────────────────────────────────────────────────

class GroqClient:
    def __init__(self):
        self._client   = None
        self.available = False
        if GROQ_API_KEY:
            try:
                self._client = Groq(api_key=GROQ_API_KEY)
                self._client.models.list()
                self.available = True
                log.info("✓ Groq client ready  model=%s", GROQ_MODEL)
            except Exception as exc:
                log.error("Groq initialisation failed: %s", exc)

    def stream(
        self,
        system_prompt: str,
        user_prompt:   str,
        lang:          str,
    ) -> Generator[str, None, None]:
        if not self.available or self._client is None:
            yield LANG_CONFIG[lang]["no_data"]
            return
        try:
            completion = self._client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                stream      = True,
                temperature = 0.2,
                top_p       = 0.1,
                max_tokens  = 2048,
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            log.error("Groq streaming failed: %s", exc)
            yield LANG_CONFIG[lang]["no_data"]


# ─────────────────────────────────────────────────────────────
# MAIN LLM GENERATOR
# ─────────────────────────────────────────────────────────────

class LLMGenerator:
    """
    Orchestrates RAG retrieval → LLM generation.
    Compatible with RAG v16 (rag_pipeline.py).
    """

    def __init__(self):
        self._analyzer      = QueryAnalyzer()   # from rag_pipeline
        self._groq          = GroqClient()
        self._last_sources: List[Source] = []

    # ── Public streaming API ──────────────────────────────────

    def stream(
        self,
        query:       str,
        source_type: Optional[str] = None,   # kept for API compat; not forwarded to RAG v16
        faculty:     Optional[str] = None,
        department:  Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Single-turn streaming.
        Does NOT do clarification — use StreamSession for that.
        """
        self._last_sources = []
        clean_q  = preprocess_query(query)
        analysis = self._analyzer.analyze(clean_q)
        lang     = analysis["language"]
        intent   = analysis["intent"]

        yield from self._generate_from_enriched(clean_q, intent, lang)

    # ── Public synchronous API ────────────────────────────────

    def generate(self, query: str, source_type: Optional[str] = None) -> GenerationResult:
        """Collect all streamed chunks into one GenerationResult."""
        answer_chunks = list(self.stream(query, source_type=source_type))
        full_answer   = "".join(answer_chunks)

        if not full_answer:
            lang = detect_language(query)
            return GenerationResult(
                answer        = LANG_CONFIG[lang]["no_data"],
                intent        = "general_info",
                language      = lang,
                sources       = [],
                fallback_used = True,
            )

        analysis = self._analyzer.analyze(query)
        return GenerationResult(
            answer        = full_answer,
            intent        = analysis["intent"],
            language      = analysis["language"],
            sources       = self._last_sources,
            fallback_used = False,
        )

    # ── Core: retrieve → generate ─────────────────────────────

    def _generate_from_enriched(
        self,
        enriched_query: str,
        intent:         str,
        lang:           str,
    ) -> Generator[str, None, None]:
        """
        Full pipeline:
          1. Handle graph_query without vector retrieval
          2. Retrieve via RAG v16
          3. Handle bare-PDF-only result
          4. Build prompt and stream through Groq
          5. Append source footnotes
        """

        # ── graph_query: direct Neo4j hierarchy, no vectors ───
        if intent == "graph_query":
            try:
                graph_data = traverse_graph(enriched_query)
                context    = _format_graph_result(graph_data, lang)
            except Exception as exc:
                log.error("Graph traversal failed: %s", exc)
                context = ""

            if not context.strip():
                yield LANG_CONFIG[lang]["no_data"]
                return

            system_p, user_p = PromptBuilder(intent, lang).build(enriched_query, context)
            yield from self._groq.stream(system_p, user_p, lang)
            return

        # ── Standard RAG retrieval ────────────────────────────
        try:
            results = retrieve_for_llm(query=enriched_query, top_k=TOP_K_RETRIEVE)
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            yield LANG_CONFIG[lang]["no_data"]
            return

        self._last_sources = _extract_sources(results)

        # ── No results → NO_ANSWER ────────────────────────────
        if not results:
            yield LANG_CONFIG[lang]["no_data"]
            return

        # ── Bare PDF shortcut ─────────────────────────────────
        is_doc, page_url, pdf_url = _is_document_only(results)
        if is_doc:
            title = results[0].get("title", "Document")
            link_map = {
                "ar": (
                    f"يمكنك الاطلاع على الصفحة: [{title}]({page_url})\n"
                    f"أو تحميل الملف مباشرة: [⬇ PDF]({pdf_url})"
                    if page_url and page_url != pdf_url
                    else f"يمكنك تحميل الملف مباشرة: [⬇ {title}]({pdf_url})"
                ),
                "fr": (
                    f"Consultez la page : [{title}]({page_url})\n"
                    f"Ou téléchargez directement : [⬇ PDF]({pdf_url})"
                    if page_url and page_url != pdf_url
                    else f"Vous pouvez télécharger le document directement : [⬇ {title}]({pdf_url})"
                ),
                "en": (
                    f"View the page: [{title}]({page_url})\n"
                    f"Or download directly: [⬇ PDF]({pdf_url})"
                    if page_url and page_url != pdf_url
                    else f"You can download the document directly: [⬇ {title}]({pdf_url})"
                ),
            }
            yield link_map.get(lang, link_map["en"])
            return

        # ── Normal generation ─────────────────────────────────
        llm_context = results[0].get("llm_context", "")
        if not llm_context:
            # Fallback: build context from individual chunk texts
            llm_context = "\n─────\n".join(
                r.get("text", "") for r in results if r.get("text")
            )

        context_str  = truncate_context(llm_context)
        system_p, user_p = PromptBuilder(intent, lang).build(enriched_query, context_str)

        yield from self._groq.stream(system_p, user_p, lang)

        sources_md = _format_sources_md(self._last_sources, lang)
        if sources_md:
            yield sources_md


# ─────────────────────────────────────────────────────────────
# STREAM SESSION — multi-turn with pre-flight clarification
# ─────────────────────────────────────────────────────────────

class StreamSession:
    """
    Drives a multi-turn conversation with optional slot-filling.

    Usage
    ─────
        session = StreamSession()

        for chunk in session.chat("emploi du temps L2 informatique"):
            print(chunk, end="", flush=True)

        # If clarification was needed, the next call feeds the reply:
        for chunk in session.chat("Faculté des Sciences"):
            print(chunk, end="", flush=True)

        # Once session.finished is True, call session.reset() to reuse.
    """

    def __init__(self, source_type: Optional[str] = None):
        self._generator  = LLMGenerator()
        self._analyzer   = QueryAnalyzer()
        self.source_type = source_type   # kept for API compat
        self._state:  Optional[ClarificationState] = None
        self._intent: str  = "general_info"
        self._lang:   str  = "en"
        self.finished: bool = False

    # ── Public entry point ────────────────────────────────────

    def chat(self, user_input: str) -> Generator[str, None, None]:
        clean = preprocess_query(user_input)

        # ── First turn ────────────────────────────────────────
        if self._state is None:
            analysis     = self._analyzer.analyze(clean)
            self._lang   = analysis["language"]
            self._intent = analysis["intent"]

            self._state = ClarificationEngine.init(clean, self._intent, self._lang)

            if ClarificationEngine.needs_clarification(self._state):
                yield ClarificationEngine.next_question(self._state)
                return

            yield from self._run_generation()
            return

        # ── Reply to clarification question ───────────────────
        ClarificationEngine.ingest_reply(self._state, clean)

        if ClarificationEngine.needs_clarification(self._state):
            yield ClarificationEngine.next_question(self._state)
            return

        yield from self._run_generation()

    # ── Internal generation trigger ───────────────────────────

    def _run_generation(self) -> Generator[str, None, None]:
        enriched = ClarificationEngine.build_enriched_query(self._state)
        log.info("Enriched query: %s", enriched)

        yield from self._generator._generate_from_enriched(
            enriched_query = enriched,
            intent         = self._intent,
            lang           = self._lang,
        )
        self.finished = True

    # ── Reset for next question ───────────────────────────────

    def reset(self) -> None:
        self._state   = None
        self._intent  = "general_info"
        self._lang    = "en"
        self.finished = False


# ─────────────────────────────────────────────────────────────
# CLI — interactive multi-turn
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args      = sys.argv[1:]
    q_parts:  List[str]     = []
    src_type: Optional[str] = None

    i = 0
    while i < len(args):
        if args[i] == "--source-type" and i + 1 < len(args):
            src_type = args[i + 1]; i += 2
        else:
            q_parts.append(args[i]); i += 1

    initial_query = " ".join(q_parts) if q_parts else None

    print(f"Model : {GROQ_MODEL}")
    print("Type 'exit' or 'quit' to stop.\n" + "=" * 60)

    session = StreamSession(source_type=src_type)

    user_msg = initial_query if initial_query else input("You: ").strip()

    while True:
        if user_msg.lower() in {"exit", "quit", "خروج", "quitter"}:
            print("Bye!")
            break

        print("\nAssistant: ", end="", flush=True)
        start = time.time()

        for chunk in session.chat(user_msg):
            print(chunk, end="", flush=True)

        elapsed = time.time() - start
        print(f"\n[{elapsed:.2f}s]\n" + "─" * 60)

        if session.finished:
            cont = input("\nAnother question? (y/n): ").strip().lower()
            if cont != "y":
                break
            session.reset()

        user_msg = input("\nYou: ").strip()
