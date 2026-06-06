#!/usr/bin/env python3
"""
LLM Generation Layer v5.0 — Farhat Abbas University Sétif 1
============================================================
Works with RAG v13.0 (Neo4j-native, no ChromaDB / metadata.json).

Changes vs v4.0 (Gemini)
────────────────────────
  • Replaced Google Gemini with Groq API (faster, generous free tier)
  • GROQ_API_KEY      — set via env var
  • GROQ_MODEL        — defaults to "llama3-70b-8192"
  • Uses official 'groq' Python library (pip install groq)
  • Streaming via Groq's chat completion API
"""

import os
import re
import sys
import time
import logging
from typing import Dict, Generator, List, Optional, Tuple
from dataclasses import dataclass, field

from groq import Groq

import rag
from rag import QueryUnderstanding

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")   # other options: mixtral-8x7b-32768, gemma2-9b-it
MAX_CONTEXT_CHARS  = 8000
GENERATION_TIMEOUT = 60

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set. Generation will fail.")

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class Source:
    title:       str
    url:         str             # page url (human-readable)
    pdf_url:     str = ""        # direct pdf download link (may be same as url)
    source_type: str = "page"
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

LANG_CONFIG = {
    "ar": {
        "name":        "Arabic",
        "instruction": "أجب باللغة العربية فقط.",
        "no_data":     "عذراً، لا توجد معلومات موثوقة حول هذا الموضوع في قاعدة البيانات المتاحة.",
    },
    "fr": {
        "name":        "French",
        "instruction": "Répondez UNIQUEMENT en français.",
        "no_data":     "Désolé, aucune information fiable sur ce sujet n'a été trouvée dans les données disponibles.",
    },
    "en": {
        "name":        "English",
        "instruction": "Respond ONLY in English.",
        "no_data":     "Sorry, no reliable information about this topic was found in the available data.",
    },
}

_RE_AR = re.compile(r"[\u0600-\u06FF]")
_RE_FR = re.compile(
    r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|est)\b",
    re.IGNORECASE,
)

def detect_language(text: str) -> str:
    s = text[:200]
    if len(_RE_AR.findall(s)) > 3: return "ar"
    if len(_RE_FR.findall(s)) > 2: return "fr"
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
            "4. LINKS: If a chunk ends with 'Source: [URL]' or 'PDF: [URL]', "
            "include that link naturally in your answer as a clickable Markdown link."
        )
        extras = {
            "person_lookup": (
                "\n5. Extract: Full Name, Title (Prof/Dr), Department, Faculty, "
                "Email, Office/Bureau, Phone. Format as a profile card. "
                "Omit fields that are missing — do not guess."
            ),
            "course_query": (
                "\n5. Extract: Course Name, Department, Credits, Teacher(s), "
                "Syllabus/Description if available."
            ),
            "admin_query": (
                "\n5. Extract precise steps, deadlines, required documents, "
                "and relevant office contacts. Use bullet points for procedures."
            ),
            "translation": (
                f"\n5. Translate the provided text into {self.cfg['name']}. "
                "Maintain original meaning, tone, and structure."
            ),
            "table_query": (
                "\n5. Format the requested list or table strictly as "
                "a Markdown table or bulleted list."
            ),
        }
        return base + extras.get(self.intent, "")

    def _user(self, query: str, context: str) -> str:
        if not context or context == "No relevant context available.":
            return (
                f"CONTEXT: Empty\n\n"
                f"QUESTION: {query}\n\n"
                f"ANSWER: {self.cfg['no_data']}"
            )
        return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"


# ─────────────────────────────────────────────────────────────
# SOURCE FORMATTER
# ─────────────────────────────────────────────────────────────

def _format_sources_md(sources: List[Source], lang: str) -> str:
    """Return a Markdown sources block to append after the LLM answer."""
    if not sources:
        return ""
    labels = {
        "ar": "📚 المصادر",
        "fr": "📚 Sources",
        "en": "📚 Sources",
    }
    lines = [f"\n\n{labels.get(lang, '📚 Sources')}"]
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
# MAIN LLM GENERATOR (GROQ)
# ─────────────────────────────────────────────────────────────

class LLMGenerator:
    def __init__(self):
        self.qu            = QueryUnderstanding()
        self.model_name    = GROQ_MODEL
        self.available     = self._check_groq()
        self._last_sources: List[Source] = []

        if self.available:
            self._client = Groq(api_key=GROQ_API_KEY)
        else:
            self._client = None
            log.error("Groq client not available. Check your GROQ_API_KEY.")

    # ── Availability check ─────────────────────────────────────

    def _check_groq(self) -> bool:
        if not GROQ_API_KEY:
            return False
        try:
            # Simple test: list models (lightweight)
            client = Groq(api_key=GROQ_API_KEY)
            client.models.list()
            return True
        except Exception as exc:
            log.error("Groq initialisation failed: %s", exc)
            return False

    # ── Source extraction ──────────────────────────────────────

    def _extract_sources(self, results: List[Dict]) -> List[Source]:
        sources:   List[Source] = []
        seen_urls: set           = set()

        for r in results:
            meta        = r.get("metadata", {})
            source_type = r.get("source_type") or meta.get("source_type", "page")
            pdf_url     = r.get("pdf_url")  or meta.get("pdf_url",  "")
            page_url    = r.get("page_url") or meta.get("page_url", "")
            direct_url  = r.get("url")      or meta.get("url",      "")

            canonical = page_url or direct_url or pdf_url
            if not canonical or canonical in seen_urls:
                continue

            seen_urls.add(canonical)
            sources.append(Source(
                title       = r.get("title") or meta.get("title", "Document"),
                url         = page_url or direct_url,
                pdf_url     = pdf_url,
                source_type = source_type,
                chunk_score = r.get("score", 0.0),
            ))

        return sources

    # ── Document-only shortcut ─────────────────────────────────

    def _is_document_only(self, results: List[Dict]) -> Tuple[bool, str, str]:
        if not results:
            return False, "", ""
        top         = results[0]
        source_type = top.get("source_type") or top.get("metadata", {}).get("source_type", "")
        text        = top.get("text", "").strip()
        pdf_url     = top.get("pdf_url")  or top.get("metadata", {}).get("pdf_url",  "")
        page_url    = top.get("page_url") or top.get("metadata", {}).get("page_url", "")

        if source_type == "pdf" and len(text) < 200 and pdf_url:
            return True, page_url, pdf_url
        return False, "", ""

    # ── Groq streaming call ────────────────────────────────────

    def _stream_groq(self, system_prompt: str,
                     user_prompt: str,
                     lang: str) -> Generator[str, None, None]:
        """
        Call Groq chat completions with streaming.
        """
        if not self.available or self._client is None:
            yield LANG_CONFIG[lang]["no_data"]
            return

        try:
            # Groq expects messages array (system + user)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]

            stream = self._client.chat.completions.create(
                model    = self.model_name,
                messages = messages,
                stream   = True,
                temperature = 0.2,
                top_p       = 0.1,
                max_tokens  = 2048,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as exc:
            log.error("Groq streaming failed: %s", exc)
            yield LANG_CONFIG[lang]["no_data"]

    # ── Public API ─────────────────────────────────────────────

    def generate(self, query: str,
                 source_type: Optional[str] = None) -> GenerationResult:
        """Synchronous: collect all streamed chunks into one string."""
        answer_chunks = list(self.stream(query, source_type=source_type))
        full_answer   = "".join(answer_chunks)

        if not full_answer:
            lang = detect_language(query)
            return GenerationResult(
                answer       = LANG_CONFIG[lang]["no_data"],
                intent       = "general_info",
                language     = lang,
                sources      = [],
                fallback_used= True,
            )

        analysis = self.qu.analyze(query)
        return GenerationResult(
            answer        = full_answer,
            intent        = analysis["intent"],
            language      = analysis["language"],
            sources       = self._last_sources,
            fallback_used = False,
        )

    def stream(self, query: str,
               source_type: Optional[str] = None,
               faculty:     Optional[str] = None,   # kept for backward compat
               department:  Optional[str] = None,   # kept for backward compat
               ) -> Generator[str, None, None]:
        """Streaming generator — yields string chunks of the LLM answer."""
        self._last_sources = []

        # 1. Analyze
        clean_q  = preprocess_query(query)
        analysis = self.qu.analyze(clean_q)
        lang     = analysis["language"]
        intent   = analysis["intent"]

        # 2. Retrieve
        try:
            results = rag.retrieve_for_llm(
                query       = clean_q,
                top_k       = 5,
                source_type = source_type,
            )
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            yield LANG_CONFIG[lang]["no_data"]
            return

        self._last_sources = self._extract_sources(results)

        # 3. No context
        if not results:
            yield LANG_CONFIG[lang]["no_data"]
            return

        # 4. Document-only shortcut
        is_doc, page_url, pdf_url = self._is_document_only(results)
        if is_doc:
            title = results[0].get("title", "Document")
            if page_url and page_url != pdf_url:
                link_map = {
                    "ar": (f"يمكنك الاطلاع على الصفحة: [{title}]({page_url})\n"
                           f"أو تحميل الملف مباشرة: [⬇ PDF]({pdf_url})"),
                    "fr": (f"Consultez la page : [{title}]({page_url})\n"
                           f"Ou téléchargez directement : [⬇ PDF]({pdf_url})"),
                    "en": (f"View the page: [{title}]({page_url})\n"
                           f"Or download directly: [⬇ PDF]({pdf_url})"),
                }
            else:
                link_map = {
                    "ar": f"يمكنك تحميل الملف مباشرة: [⬇ {title}]({pdf_url})",
                    "fr": f"Vous pouvez télécharger le document directement : [⬇ {title}]({pdf_url})",
                    "en": f"You can download the document directly: [⬇ {title}]({pdf_url})",
                }
            yield link_map.get(lang, link_map["en"])
            return

        # 5. Build prompt
        context_str                = truncate_context(results[0].get("llm_context", ""))
        system_prompt, user_prompt = PromptBuilder(intent, lang).build(clean_q, context_str)

        # 6. Stream from Groq
        yield from self._stream_groq(system_prompt, user_prompt, lang)

        # 7. Append formatted sources after the answer
        sources_md = _format_sources_md(self._last_sources, lang)
        if sources_md:
            yield sources_md


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python llm.py "Your question" [--source-type pdf|page]')
        sys.exit(1)

    args      = sys.argv[1:]
    q_parts:  List[str]      = []
    src_type: Optional[str]  = None

    i = 0
    while i < len(args):
        if args[i] == "--source-type" and i + 1 < len(args):
            src_type = args[i + 1]; i += 2
        else:
            q_parts.append(args[i]); i += 1

    user_query = " ".join(q_parts)
    if not user_query:
        print("Error: No query provided.")
        sys.exit(1)

    print(f"Query     : {user_query}")
    print(f"Model     : {GROQ_MODEL}")
    print("=" * 60)

    generator  = LLMGenerator()
    start_time = time.time()

    for chunk in generator.stream(user_query, source_type=src_type):
        print(chunk, end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n\n{'─' * 60}")
    print(f"⏱️  Generated in {elapsed:.2f}s")