#!/usr/bin/env python3
"""
LLM Generation Layer v10.0 — Farhat Abbas University Sétif 1
============================================================
Compatible with RAG v16.5 (rag.py) — Stable + Bidirectional + Soft Temporal.

Changes vs v9.0
───────────────
  • Compatible with RAG v16.5 (no clarification, no context API changes)
  • Temporal hints passed to LLM prompt for time-aware answers
  • Only shows sources that were actually USED (score > 0.50)
  • PDF file awareness with clear messaging
  • StreamSession with multi-turn conversation + context tracking
  • Cleaner source extraction — no sources from unused chunks
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
from rag import QueryAnalyzer

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
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_CONTEXT_CHARS  = 10000  # Increased for expanded chunks
GENERATION_TIMEOUT = 60
TOP_K_RETRIEVE = 5

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY is not set. Generation will fail.")

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class Source:
    title:       str
    url:         str
    pdf_url:     str = ""
    source_type: str = "page"
    chunk_score: float = 0.0
    is_pdf_only: bool = False


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
        "pdf_notice":  "📄 هذه المعلومات من ملف PDF. يمكنك تحميله من:",
    },
    "fr": {
        "name":        "French",
        "instruction": "Répondez UNIQUEMENT en français.",
        "no_data":     "Désolé, aucune information fiable sur ce sujet n'a été trouvée dans les données disponibles.",
        "graph_intro": "Voici la structure de la faculté / département demandé :",
        "pdf_notice":  "📄 Ces informations proviennent d'un fichier PDF. Téléchargez-le depuis :",
    },
    "en": {
        "name":        "English",
        "instruction": "Respond ONLY in English.",
        "no_data":     "Sorry, no reliable information about this topic was found in the available data.",
        "graph_intro": "Here is the requested faculty / department structure:",
        "pdf_notice":  "📄 This information is from a PDF file. Download it from:",
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
    """Convert the dict returned by rag.traverse_graph() into readable Markdown."""
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
# SOURCE EXTRACTION — only from used chunks (score > 0.45)
# ─────────────────────────────────────────────────────────────

def _extract_sources(results: List[Dict]) -> List[Source]:
    """
    Extract sources ONLY from chunks with score > 0.45.
    Slightly lower threshold to include borderline relevant sources.
    """
    sources:   List[Source] = []
    seen_urls: set          = set()

    for r in results:
        score = r.get("score", 0.0)
        if score < 0.45:
            continue
        
        source_type = r.get("source_type", "page")
        pdf_url     = r.get("pdf_url", "")
        url         = r.get("url", "")
        title       = r.get("title", "Document")
        is_pdf_only = r.get("is_pdf_only", False)

        canonical = url or pdf_url
        if not canonical or canonical in seen_urls:
            continue

        seen_urls.add(canonical)
        sources.append(Source(
            title       = title,
            url         = url,
            pdf_url     = pdf_url,
            source_type = source_type,
            chunk_score = score,
            is_pdf_only = is_pdf_only,
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
    url         = top.get("url", "")

    if source_type == "pdf" and len(text) < 200 and pdf_url:
        return True, url, pdf_url
    return False, "", ""


def _format_sources_md(sources: List[Source], lang: str) -> str:
    """Return a Markdown sources block. Only includes sources that were actually used."""
    if not sources:
        return ""
    
    usable_sources = [s for s in sources if s.chunk_score > 0.45]
    if not usable_sources:
        return ""
    
    labels = {"ar": "📚 المصادر", "fr": "📚 Sources", "en": "📚 Sources"}
    lines  = [f"\n\n{labels.get(lang, '📚 Sources')}"]
    
    for s in usable_sources:
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
# INTENT-AWARE PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

class PromptBuilder:
    def __init__(self, intent: str, lang: str):
        self.intent = intent
        self.lang   = lang
        self.cfg    = LANG_CONFIG.get(lang, LANG_CONFIG["en"])

    def build(self, query: str, context: str, has_time_ref: bool = False) -> Tuple[str, str]:
        return self._system(has_time_ref), self._user(query, context)

    def _system(self, has_time_ref: bool = False) -> str:
        base = (
            "You are an expert university assistant for Farhat Abbas University Sétif 1.\n"
            "CRITICAL RULES:\n"
            f"1. {self.cfg['instruction']}\n"
            "2. GROUNDING: Base your answer EXACTLY on the provided context. "
            "Do not hallucinate or use outside knowledge.\n"
            "3. FORMAT: Use Markdown (bold, lists, tables where useful).\n"
            "4. LINKS: If a chunk contains 'Source:' or 'PDF:' followed by a URL, "
            "include that link naturally in your answer as a clickable Markdown link.\n"
            "5. QUALITY: Each chunk has a [score=X.XX]. "
            "ONLY use chunks with score > 0.45. "
            "If multiple chunks contradict, prefer the one with HIGHER score.\n"
            "6. PDF FILES: If a chunk says '📄 Source PDF' or '📄 This information is from a PDF file', "
            "mention this clearly in your answer and provide the download link.\n"
            "7. EXPANDED CHUNKS: The chunks may contain EXPANDED content (previous and next sections "
            "merged together). Read the FULL text of each chunk carefully - names may appear "
            "earlier in the merged text while titles/positions appear later.\n"
        )
        
        # ✨ Temporal hint
        if has_time_ref:
            base += (
                "8. TIME AWARENESS: The query contains time-related words (current, latest, maintenant, حالي, etc.). "
                "Look for the MOST RECENT dates in the context (2024, 2025, 2026). "
                "The chunks are already sorted with recent content prioritized. "
                "Mention dates clearly when available.\n"
            )
            base += "9. If NO chunk has score > 0.45, respond with the no_data message."
        else:
            base += "8. If NO chunk has score > 0.45, respond with the no_data message."
        
        extras: Dict[str, str] = {
            "person_lookup": (
                "\n\nADDITIONAL INSTRUCTIONS:\n"
                "• Extract: Full Name, Title (Prof/Dr), Department, Faculty, "
                "Email, Office/Bureau, Phone. Format as a profile card.\n"
                "• Omit fields that are missing — do not guess.\n"
                "• IMPORTANT: Names may be in a different part of the merged chunk "
                "than the position/title. Read the ENTIRE chunk text to find all details.\n"
                "• If the query asks about 'current' or 'حالي' or 'maintenant', "
                "look for the MOST RECENT information (latest dates like 2024, 2025, 2026)."
            ),
            "course_query": (
                "\n\nADDITIONAL INSTRUCTIONS:\n"
                "• Extract: Course Name, Department, Credits, Teacher(s), "
                "Syllabus/Description if available."
            ),
            "admin_query": (
                "\n\nADDITIONAL INSTRUCTIONS:\n"
                "• Extract precise steps, deadlines, required documents, "
                "and relevant office contacts. Use bullet points for procedures.\n"
                "• Pay attention to DATES in the context — deadlines are important!"
            ),
            "node_query": (
                "\n\nADDITIONAL INSTRUCTIONS:\n"
                "• List the relevant specializations, programs, or departments clearly.\n"
                "• Include any years/levels mentioned (L1, L2, L3, M1, M2)."
            ),
            "table_query": (
                "\n\nADDITIONAL INSTRUCTIONS:\n"
                "• Format the requested list or table strictly as "
                "a Markdown table or bulleted list."
            ),
            "graph_query": (
                "\n\nADDITIONAL INSTRUCTIONS:\n"
                "• The context contains a JSON hierarchy tree. "
                "Format it as a clean Markdown outline."
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
    Compatible with RAG v16.5 (rag.py).
    """

    def __init__(self):
        self._analyzer      = QueryAnalyzer()
        self._groq          = GroqClient()
        self._last_sources: List[Source] = []

    def stream(
        self,
        query:       str,
        source_type: Optional[str] = None,
        faculty:     Optional[str] = None,
        department:  Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Single-turn streaming."""
        self._last_sources = []
        clean_q  = preprocess_query(query)
        analysis = self._analyzer.analyze(clean_q)
        lang     = analysis["language"]
        intent   = analysis["intent"]
        has_time = analysis.get("has_time_ref", False)

        yield from self._generate_from_enriched(clean_q, intent, lang, has_time)

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

    def _generate_from_enriched(
        self,
        enriched_query: str,
        intent:         str,
        lang:           str,
        has_time:       bool = False,
    ) -> Generator[str, None, None]:
        """
        Full pipeline:
          1. Handle graph_query
          2. Retrieve via RAG v16.5
          3. Build prompt and stream through Groq
          4. Append source footnotes
        """

        # ── graph_query: direct Neo4j hierarchy ───
        if intent == "graph_query":
            try:
                graph_data = rag.traverse_graph(enriched_query)
                context    = _format_graph_result(graph_data, lang)
            except Exception as exc:
                log.error("Graph traversal failed: %s", exc)
                context = ""

            if not context.strip():
                yield LANG_CONFIG[lang]["no_data"]
                return

            system_p, user_p = PromptBuilder(intent, lang).build(enriched_query, context, has_time)
            yield from self._groq.stream(system_p, user_p, lang)
            return

        # ── Standard RAG retrieval ────────────────────────────
        try:
            results = rag.retrieve_for_llm(
                query=enriched_query,
                top_k=TOP_K_RETRIEVE,
            )
        except Exception as exc:
            log.error("Retrieval failed: %s", exc)
            yield LANG_CONFIG[lang]["no_data"]
            return

        # ✨ Extract sources ONLY from used chunks
        self._last_sources = _extract_sources(results)

        # ── No results → NO_ANSWER ────────────────────────────
        if not results:
            yield LANG_CONFIG[lang]["no_data"]
            return

        # ── Bare PDF shortcut ─────────────────────────────────
        is_doc, page_url, pdf_url = _is_document_only(results)
        if is_doc:
            title = results[0].get("title", "Document")
            cfg = LANG_CONFIG.get(lang, LANG_CONFIG["en"])
            if page_url and page_url != pdf_url:
                yield f"{cfg['pdf_notice']} [{title}]({pdf_url})\n\n🔗 Page web: [{title}]({page_url})"
            else:
                yield f"{cfg['pdf_notice']} [{title}]({pdf_url})"
            return

        # ── Normal generation ─────────────────────────────────
        llm_context = results[0].get("llm_context", "")
        if not llm_context:
            llm_context = "\n─────\n".join(
                r.get("text", "") for r in results if r.get("text") and r.get("score", 0) > 0.45
            )

        context_str  = truncate_context(llm_context)
        system_p, user_p = PromptBuilder(intent, lang).build(enriched_query, context_str, has_time)

        yield from self._groq.stream(system_p, user_p, lang)

        # ✨ Append sources (only from used chunks)
        sources_md = _format_sources_md(self._last_sources, lang)
        if sources_md:
            yield sources_md


# ─────────────────────────────────────────────────────────────
# STREAM SESSION — multi-turn with context
# ─────────────────────────────────────────────────────────────

class StreamSession:
    """
    Drives a multi-turn conversation with context tracking.
    """

    def __init__(self, source_type: Optional[str] = None):
        self._generator  = LLMGenerator()
        self._analyzer   = QueryAnalyzer()
        self.source_type = source_type
        self.finished: bool = False

    def chat(self, user_input: str) -> Generator[str, None, None]:
        clean = preprocess_query(user_input)
        yield from self._generator.stream(clean)
        self.finished = True

    def reset(self) -> None:
        rag.reset_context() if hasattr(rag, 'reset_context') else None
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
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    session = StreamSession(source_type=src_type)

    user_msg = initial_query if initial_query else input("\nYou: ").strip()

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
