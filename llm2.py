#!/usr/bin/env python3
"""
LLM Answer Generation Layer v1.0 — Farhat Abbas University Sétif 1
==================================================================
Production-ready Ollama-based answer generation integrated with
rag.py's retrieve_for_llm() pipeline.

Responsibilities
----------------
  1. Query preprocessing & optimization for retrieval
  2. Language detection & answer language locking
  3. Retrieval call via retrieve_for_llm()
  4. Prompt construction with strict context grounding
  5. Ollama API call with deterministic parameters
  6. Answer post-processing (link extraction, translation, formatting)
  7. File/link detection logic for direct-document answers
  8. Hallucination guard — "no answer" preferred over fabrication

Install
-------
  pip install requests
"""

# ─────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────
import json
import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────
import requests

# ─────────────────────────────────────────────────────────────
# RAG IMPORT
# ─────────────────────────────────────────────────────────────
try:
    from rag import retrieve_for_llm, QueryUnderstanding
except ImportError:
    # Fallback: if rag.py is in the same directory but import path differs
    import importlib.util
    _rag_path = Path(__file__).with_name("rag.py")
    if _rag_path.exists():
        _spec = importlib.util.spec_from_file_location("rag", _rag_path)
        _rag_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_rag_mod)
        retrieve_for_llm = _rag_mod.retrieve_for_llm
        QueryUnderstanding = _rag_mod.QueryUnderstanding
    else:
        raise ImportError("rag.py not found in the same directory as llm.py")

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Generation parameters — strictly factual, low temperature
LLM_TEMPERATURE    = 0.25
LLM_TOP_P          = 0.90
LLM_REPEAT_PENALTY = 1.15
LLM_NUM_PREDICT    = 1024

# Timeout for Ollama generation calls (seconds)
OLLAMA_TIMEOUT = 45

# Maximum context length to send to LLM (characters)
MAX_CONTEXT_CHARS = 12000

# Answer length thresholds
MIN_ANSWER_CHARS = 20
MAX_ANSWER_CHARS = 4000

# Link detection regex
_URL_RE = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class LLMResult:
    """Structured result from the LLM answer generation pipeline."""
    answer:          str
    language:        str
    intent:          str
    sources:         List[Dict] = field(default_factory=list)
    has_document:    bool = False
    document_links:  List[str] = field(default_factory=list)
    confidence:      str = "medium"  # low | medium | high
    raw_chunks_used: int = 0


# ══════════════════════════════════════════════════════════════
# LAYER 1 — QUERY PREPROCESSING
# ══════════════════════════════════════════════════════════════

class QueryPreprocessor:
    """
    Cleans and optimizes user queries for retrieval without semantic drift.

    Rules:
      - Strip leading/trailing whitespace and common filler phrases
      - Normalize Unicode (NFC)
      - Remove excessive punctuation
      - Preserve original meaning exactly
      - Do NOT expand with synonyms or unrelated terms
      - Detect and flag bare-name queries for person_lookup routing
    """

    # Phrases that add noise but no semantic value
    _NOISE_PREFIXES = re.compile(
        r"^(?i)(?:"
        r"please\s+(?:tell\s+me|give\s+me|find|search|look\s+up|help\s+me\s+(?:find|with))\s*"
        r"|s'il\s+vous\s+pla[iî]t\s+(?:donne[rz]?-moi|trouve[rz]?|cherche[rz]?|aidez-moi\s+(?:à\s+)?)"
        r"|من\s+فضلك\s+(?:أخبرني|أعطني|ابحث|جد|ساعدني)"
        r"|أرجو\s+(?:أن\s+)?(?:تخبرني|تعطيني|تبحث|تجد)"
        r"|je\s+voudrais\s+(?:savoir|trouver|trouve[rz]?)"
        r"|i\s+would\s+like\s+(?:to\s+know|to\s+find)"
        r"|can\s+you\s+(?:tell\s+me|find|give\s+me|help\s+me\s+(?:find|with))"
        r"|could\s+you\s+(?:tell\s+me|find|give\s+me)"
        r"|peux-tu\s+(?:me\s+dire|me\s+donner|trouver|chercher)"
        r"|pourriez-vous\s+(?:me\s+dire|me\s+donner|trouver)"
        r"|هل\s+يمكنك\s+(?:أن\s+)?(?:تخبرني|تعطيني|تبحث|تجد)"
        r"| qu'est-ce que | what is | que signifie | what does \s+mean"
        r")\s*[:\-]?\s*",
        re.UNICODE,
    )

    # Excessive punctuation cleanup
    _PUNCT_DUPE = re.compile(r"([!?.,:;])\1{2,}")
    _MULTI_SPACE = re.compile(r"\s+")

    # Bare name heuristic (same as rag.py but duplicated here for independence)
    _BARE_NAME_RE = re.compile(
        r"^[A-ZÀ-Ö][a-zà-ö]+$|^\u0600-\u06FF]{2,}$", re.UNICODE
    )
    _NAME_TRIGGERS = re.compile(
        r"\b(dr|pr|prof|professeur|docteur|mr|mme|mlle|"
        r"أستاذ|دكتور|أ\.د|د\.)\s*\.?\s*",
        re.IGNORECASE | re.UNICODE,
    )

    def preprocess(self, query: str) -> str:
        """Return cleaned query string optimized for retrieval."""
        if not query or not query.strip():
            return ""

        # 1. Unicode normalization
        text = unicodedata.normalize("NFC", query.strip())

        # 2. Strip common noise prefixes
        text = self._NOISE_PREFIXES.sub("", text)

        # 3. Collapse excessive punctuation
        text = self._PUNCT_DUPE.sub(r"\1", text)

        # 4. Normalize whitespace
        text = self._MULTI_SPACE.sub(" ", text).strip()

        # 5. Ensure query ends with proper punctuation if it's a question
        if text and text[-1] not in "?.!؟":
            # Detect question words to auto-append ?
            q_markers = re.search(
                r"\b(what|where|when|why|how|which|who|whom|whose|"
                r"quel|quelle|quels|quelles|qui|que|quoi|comment|où|quand|"
                r"ماذا|أين|متى|كيف|أي|من|هل|ما)\b",
                text, re.IGNORECASE,
            )
            if q_markers:
                text += "?"

        return text.strip()

    def is_bare_name(self, query: str) -> bool:
        """Check if query is essentially just a proper name."""
        tokens = query.strip().split()
        if not (1 <= len(tokens) <= 3):
            return False
        # Strip triggers
        clean = self._NAME_TRIGGERS.sub("", query).strip()
        tokens_clean = clean.split()
        return sum(
            1 for t in tokens_clean if self._BARE_NAME_RE.match(t)
        ) >= max(1, len(tokens_clean) - 1)


# ══════════════════════════════════════════════════════════════
# LAYER 2 — LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════

class LanguageDetector:
    """
    Fast language detection using the same heuristics as rag.py.
    Returns ISO-style codes: 'ar', 'fr', 'en'.
    """

    _RE_AR = re.compile(r"[\u0600-\u06FF]")
    _RE_FR = re.compile(
        r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|par|est|"
        r"comment|quoi|qui|quel|quelle|mon|ton|son|leur|nos|vos)\b",
        re.IGNORECASE,
    )

    def detect(self, text: str) -> str:
        s = text[:300]
        ar_chars = len(self._RE_AR.findall(s))
        ar_threshold = 1 if len(s.strip()) <= 15 else 5
        if ar_chars >= ar_threshold:
            return "ar"
        if len(self._RE_FR.findall(s)) > 2:
            return "fr"
        return "en"


# ══════════════════════════════════════════════════════════════
# LAYER 3 — RETRIEVAL INTEGRATION
# ══════════════════════════════════════════════════════════════

def fetch_context(
    query: str,
    top_k: int = 8,
    faculty: Optional[str] = None,
    department: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Call rag.py retrieve_for_llm() and return (chunks, llm_context).

    Returns:
        chunks:      List of dicts from retrieve_for_llm()
        llm_context: Pre-formatted context string (from first chunk) or None
    """
    try:
        chunks = retrieve_for_llm(
            query=query,
            top_k=top_k,
            faculty=faculty,
            department=department,
        )
    except Exception as exc:
        log.error("Retrieval failed: %s", exc)
        return [], None

    llm_context = chunks[0].get("llm_context") if chunks else None
    return chunks, llm_context


# ══════════════════════════════════════════════════════════════
# LAYER 4 — PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════

class PromptBuilder:
    """
    Builds deterministic, language-locked system prompts for Ollama.

    Design principles:
      - System prompt is the primary control mechanism
      - User prompt contains ONLY query + context (no instructions)
      - Language is strictly locked to query language
      - Grounding rules are explicit and non-negotiable
    """

    # Language-specific system prompts
    _SYSTEM_PROMPTS = {
        "ar": (
            "أنت مساعد ذكي متخصص في جامعة فرحات عباس - سطيف 1.\n"
            "قواعد صارمة:\n"
            "1. أجب دائماً باللغة العربية فقط.\n"
            "2. استند فقط على المعلومات المذكورة في السياق المقدم.\n"
            "3. إذا كانت الإجابة وثيقة أو ملف PDF، اذكر الرابط مباشرة.\n"
            "4. لا تختلق معلومات. إذا لم تجد إجابة، قل: 'لم يتم العثور على معلومات موثوقة.'\n"
            "5. حافظ على الإجابة واضحة ومنظمة ودقيقة.\n"
            "6. لا تضف معلومات خارج السياق."
        ),
        "fr": (
            "Vous êtes un assistant intelligent spécialisé pour l'Université Farhat Abbas — Sétif 1.\n"
            "Règles strictes :\n"
            "1. Répondez TOUJOURS en français uniquement.\n"
            "2. Basez-vous UNIQUEMENT sur les informations fournies dans le contexte.\n"
            "3. Si la réponse est un document ou un PDF, donnez le lien directement.\n"
            "4. N'inventez aucune information. Si vous ne trouvez pas de réponse, dites : 'Aucune information fiable trouvée.'\n"
            "5. Gardez la réponse claire, structurée et précise.\n"
            "6. N'ajoutez aucune information hors contexte."
        ),
        "en": (
            "You are an intelligent assistant specialized for Farhat Abbas University — Sétif 1.\n"
            "Strict rules:\n"
            "1. ALWAYS respond in English only.\n"
            "2. Base your answer ONLY on the information provided in the context.\n"
            "3. If the answer is a document or PDF, provide the direct link.\n"
            "4. Do NOT invent information. If no answer is found, say: 'No reliable information found.'\n"
            "5. Keep the answer clear, structured, and precise.\n"
            "6. Do NOT add any information outside the provided context."
        ),
    }

    # Answer format instructions per language
    _FORMAT_INSTRUCTIONS = {
        "ar": (
            "\n\nتعليمات التنسيق:\n"
            "- إذا كانت الإجابة تحتوي على رابط وثيقة، اذكر الرابط بوضوح في نهاية الإجابة.\n"
            "- استخدم تنسيق Markdown للعناوين والقوائم عند الضرورة.\n"
            "- تجنب الإجابات الطويلة غير الضرورية."
        ),
        "fr": (
            "\n\nInstructions de format :\n"
            "- Si la réponse contient un lien vers un document, mentionnez-le clairement à la fin.\n"
            "- Utilisez le format Markdown pour les titres et listes si nécessaire.\n"
            "- Évitez les réponses longues et inutiles."
        ),
        "en": (
            "\n\nFormat instructions:\n"
            "- If the answer contains a document link, mention it clearly at the end.\n"
            "- Use Markdown for headings and lists when appropriate.\n"
            "- Avoid unnecessarily long answers."
        ),
    }

    def build(
        self,
        query: str,
        context: str,
        lang: str,
        is_document_answer: bool = False,
    ) -> Dict[str, str]:
        """
        Build the complete prompt payload for Ollama.

        Returns dict with 'system' and 'user' keys.
        """
        system = self._SYSTEM_PROMPTS.get(lang, self._SYSTEM_PROMPTS["en"])
        system += self._FORMAT_INSTRUCTIONS.get(lang, self._FORMAT_INSTRUCTIONS["en"])

        # Truncate context if too long
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

        if is_document_answer:
            user = (
                f"Query: {query}\n\n"
                f"The retrieved result indicates a document. "
                f"Provide the direct link and a very brief description.\n\n"
                f"Context:\n{context}"
            )
        else:
            user = (
                f"Query: {query}\n\n"
                f"Based ONLY on the following context, provide a clear and accurate answer.\n\n"
                f"Context:\n{context}"
            )

        return {"system": system, "user": user}


# ══════════════════════════════════════════════════════════════
# LAYER 5 — OLLAMA API CLIENT
# ══════════════════════════════════════════════════════════════

class OllamaClient:
    """
    Robust Ollama API client with retry logic and error handling.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
    ):
        self._url = base_url.rstrip("/") + "/api/chat"
        self._model = model
        self._tags_url = base_url.rstrip("/") + "/api/tags"

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            r = requests.get(self._tags_url, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        repeat_penalty: float = LLM_REPEAT_PENALTY,
        num_predict: int = LLM_NUM_PREDICT,
    ) -> str:
        """
        Generate text via Ollama chat API.

        Returns generated text string. Raises on failure.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_predict": num_predict,
            },
        }

        try:
            r = requests.post(self._url, json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "").strip()
            return content
        except requests.exceptions.Timeout:
            log.error("Ollama request timed out after %ds", OLLAMA_TIMEOUT)
            raise
        except requests.exceptions.ConnectionError:
            log.error("Cannot connect to Ollama at %s", self._url)
            raise
        except Exception as exc:
            log.error("Ollama generation failed: %s", exc)
            raise


# ══════════════════════════════════════════════════════════════
# LAYER 6 — ANSWER POST-PROCESSING
# ══════════════════════════════════════════════════════════════

class AnswerPostProcessor:
    """
    Cleans, validates, and formats LLM-generated answers.

    Responsibilities:
      - Strip markdown code blocks if accidentally wrapped
      - Validate minimum length
      - Detect hallucination patterns ("I don't know" when context exists)
      - Extract document links
      - Ensure language consistency
    """

    # Patterns that indicate the model is refusing despite having context
    _REFUSAL_PATTERNS = {
        "ar": re.compile(
            r"(لا\s+أعرف|لا\s+أستطيع|لا\s+يوجد\s+معلومات|"
            r"غير\s+متأكد|لا\s+أملك\s+معلومات|ليس\s+لدي\s+إجابة)",
            re.IGNORECASE,
        ),
        "fr": re.compile(
            r"(je\s+ne\s+sais\s+pas|je\s+ne\s+peux\s+pas|"
            r"aucune\s+information|pas\s+d'information|"
            r"je\s+n'ai\s+pas\s+d'information)",
            re.IGNORECASE,
        ),
        "en": re.compile(
            r"(i\s+don't\s+know|i\s+can't|no\s+information|"
            r"i\s+don't\s+have|i\s+am\s+not\s+sure|"
            r"i\s+do\s+not\s+know)",
            re.IGNORECASE,
        ),
    }

    # Generic "no answer" phrases to detect and standardize
    _NO_ANSWER_PHRASES = {
        "ar": "لم يتم العثور على معلومات موثوقة.",
        "fr": "Aucune information fiable trouvée.",
        "en": "No reliable information found.",
    }

    def clean(self, text: str) -> str:
        """Remove accidental markdown code fences and normalize whitespace."""
        # Strip code fences
        text = re.sub(r"^```[\w]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def validate(
        self,
        text: str,
        lang: str,
        has_context: bool,
    ) -> Tuple[str, bool]:
        """
        Validate answer quality.

        Returns (cleaned_text, is_valid).
        If invalid, returns standardized no-answer phrase.
        """
        text = self.clean(text)

        # Too short
        if len(text) < MIN_ANSWER_CHARS:
            return self._NO_ANSWER_PHRASES.get(lang, self._NO_ANSWER_PHRASES["en"]), False

        # Too long (possible runaway generation)
        if len(text) > MAX_ANSWER_CHARS:
            text = text[:MAX_ANSWER_CHARS] + "\n...[truncated]"

        # If we have context but model is refusing, that's a failure
        if has_context:
            refusal_pat = self._REFUSAL_PATTERNS.get(lang, self._REFUSAL_PATTERNS["en"])
            if refusal_pat.search(text):
                log.warning("Model refused despite having context — possible misalignment")
                return self._NO_ANSWER_PHRASES.get(lang, self._NO_ANSWER_PHRASES["en"]), False

        return text, True

    def extract_links(self, text: str) -> List[str]:
        """Extract all HTTP(S) URLs from the answer."""
        return _URL_RE.findall(text)

    def standardize_no_answer(self, lang: str) -> str:
        """Return the standardized "no answer" phrase for the given language."""
        return self._NO_ANSWER_PHRASES.get(lang, self._NO_ANSWER_PHRASES["en"])


# ══════════════════════════════════════════════════════════════
# LAYER 7 — DOCUMENT / LINK DETECTION
# ══════════════════════════════════════════════════════════════

def detect_document_answer(chunks: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Determine if the top retrieved result is primarily a document/link answer.

    Returns:
        is_document: True if answer should be link-centric
        links:       List of unique document URLs to present
    """
    if not chunks:
        return False, []

    links: List[str] = []
    for chunk in chunks[:3]:  # Check top 3
        pdf_url = chunk.get("pdf_url", "")
        url = chunk.get("url", "")
        if pdf_url and pdf_url.strip():
            links.append(pdf_url.strip())
        if url and url.strip() and url.strip() not in links:
            links.append(url.strip())

    # Heuristic: if top chunk has very short text but has a link, it's a doc
    top_text = chunks[0].get("text", "")
    is_short = len(top_text.strip()) < 200 if top_text else True

    # Or if title suggests a document
    title = (chunks[0].get("title") or "").lower()
    doc_indicators = [".pdf", "document", " fichier", "ملف", "وثيقة", "pdf"]
    is_doc_title = any(ind in title for ind in doc_indicators)

    is_document = bool(links) and (is_short or is_doc_title)
    return is_document, links


def build_document_answer(
    query: str,
    chunks: List[Dict],
    links: List[str],
    lang: str,
) -> str:
    """
    Build a concise document-centric answer.

    Returns formatted answer string with links.
    """
    title = chunks[0].get("title", "Document") if chunks else "Document"

    if lang == "ar":
        lines = [
            f"تم العثور على الوثيقة التالية: **{title}**",
            "",
            "يمكنك الوصول إليها من خلال الروابط التالية:",
        ]
        for link in links:
            lines.append(f"- {link}")
        if len(chunks) > 1 and chunks[0].get("text"):
            lines += ["", "ملخص:", chunks[0]["text"][:300] + "..."]
        return "\n".join(lines)

    elif lang == "fr":
        lines = [
            f"Document trouvé : **{title}**",
            "",
            "Vous pouvez y accéder via les liens suivants :",
        ]
        for link in links:
            lines.append(f"- {link}")
        if len(chunks) > 1 and chunks[0].get("text"):
            lines += ["", "Résumé :", chunks[0]["text"][:300] + "..."]
        return "\n".join(lines)

    else:  # en
        lines = [
            f"Document found: **{title}**",
            "",
            "You can access it through the following links:",
        ]
        for link in links:
            lines.append(f"- {link}")
        if len(chunks) > 1 and chunks[0].get("text"):
            lines += ["", "Summary:", chunks[0]["text"][:300] + "..."]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# LAYER 8 — MAIN ANSWER GENERATION PIPELINE
# ══════════════════════════════════════════════════════════════

class LLMAnswerEngine:
    """
    End-to-end answer generation pipeline.

    Orchestrates:
      1. Query preprocessing
      2. Language detection
      3. Retrieval
      4. Document vs. text answer routing
      5. Prompt building
      6. LLM generation
      7. Post-processing & validation
    """

    def __init__(self):
        self._preprocessor = QueryPreprocessor()
        self._lang_detector = LanguageDetector()
        self._prompt_builder = PromptBuilder()
        self._ollama = OllamaClient()
        self._post_processor = AnswerPostProcessor()

    def answer(
        self,
        query: str,
        top_k: int = 8,
        faculty: Optional[str] = None,
        department: Optional[str] = None,
    ) -> LLMResult:
        """
        Main entry point. Generate an answer for the given query.

        Args:
            query:      Raw user query string
            top_k:      Number of chunks to retrieve
            faculty:    Optional faculty filter
            department: Optional department filter

        Returns:
            LLMResult with answer, sources, metadata
        """
        # ── Step 1: Preprocess query ──────────────────────────
        original_query = query.strip()
        clean_query = self._preprocessor.preprocess(original_query)
        if not clean_query:
            log.warning("Empty query after preprocessing")
            lang = self._lang_detector.detect(original_query) if original_query else "en"
            return LLMResult(
                answer=self._post_processor.standardize_no_answer(lang),
                language=lang,
                intent="empty_query",
                confidence="low",
            )

        # ── Step 2: Detect language ───────────────────────────
        lang = self._lang_detector.detect(clean_query)
        log.info("Detected language: %s", lang)

        # ── Step 3: Retrieve context ──────────────────────────
        chunks, llm_context = fetch_context(
            query=clean_query,
            top_k=top_k,
            faculty=faculty,
            department=department,
        )

        if not chunks:
            log.warning("No chunks retrieved for query: %s", clean_query)
            return LLMResult(
                answer=self._post_processor.standardize_no_answer(lang),
                language=lang,
                intent="no_context",
                confidence="low",
                raw_chunks_used=0,
            )

        # ── Step 4: Route to document or text answer ──────────
        is_document, doc_links = detect_document_answer(chunks)

        if is_document:
            answer_text = build_document_answer(clean_query, chunks, doc_links, lang)
            return LLMResult(
                answer=answer_text,
                language=lang,
                intent="document_lookup",
                sources=chunks,
                has_document=True,
                document_links=doc_links,
                confidence="high",
                raw_chunks_used=len(chunks),
            )

        # ── Step 5: Build prompt for text answer ──────────────
        context_to_use = llm_context if llm_context else _build_fallback_context(chunks)
        prompt = self._prompt_builder.build(
            query=clean_query,
            context=context_to_use,
            lang=lang,
        )

        # ── Step 6: Generate with Ollama ──────────────────────
        if not self._ollama.is_available():
            log.error("Ollama is not available")
            return LLMResult(
                answer=self._post_processor.standardize_no_answer(lang),
                language=lang,
                intent="llm_unavailable",
                sources=chunks,
                confidence="low",
                raw_chunks_used=len(chunks),
            )

        try:
            raw_answer = self._ollama.generate(
                system=prompt["system"],
                user=prompt["user"],
            )
        except Exception as exc:
            log.error("LLM generation failed: %s", exc)
            return LLMResult(
                answer=self._post_processor.standardize_no_answer(lang),
                language=lang,
                intent="generation_error",
                sources=chunks,
                confidence="low",
                raw_chunks_used=len(chunks),
            )

        # ── Step 7: Post-process & validate ───────────────────
        validated_answer, is_valid = self._post_processor.validate(
            raw_answer, lang, has_context=True,
        )

        extracted_links = self._post_processor.extract_links(validated_answer)

        confidence = "high" if is_valid else "low"
        # Boost confidence if rerank_cal is high on top chunk
        if chunks and chunks[0].get("rerank_cal", 0) > 0.7:
            confidence = "high"

        return LLMResult(
            answer=validated_answer,
            language=lang,
            intent="text_answer",
            sources=chunks,
            has_document=bool(extracted_links),
            document_links=extracted_links,
            confidence=confidence,
            raw_chunks_used=len(chunks),
        )


def _build_fallback_context(chunks: List[Dict]) -> str:
    """Build a simple context string when llm_context is not available."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Unknown")
        text = chunk.get("text", "")
        parts.append(f"[{i}] {title}\n{text.strip()}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

_engine: Optional[LLMAnswerEngine] = None


def _get_engine() -> LLMAnswerEngine:
    global _engine
    if _engine is None:
        _engine = LLMAnswerEngine()
    return _engine


def ask(
    query: str,
    top_k: int = 8,
    faculty: Optional[str] = None,
    department: Optional[str] = None,
) -> Dict:
    """
    Primary public API for getting an LLM-generated answer.

    Returns a dict with:
        answer:          str  — the generated answer
        language:        str  — detected query language
        confidence:      str  — low | medium | high
        has_document:    bool — whether answer references a document
        document_links:  List[str] — extracted URLs
        sources_count:   int  — number of chunks used
        sources:         List[Dict] — raw chunk data (optional, for debugging)
    """
    result = _get_engine().answer(
        query=query,
        top_k=top_k,
        faculty=faculty,
        department=department,
    )

    return {
        "answer": result.answer,
        "language": result.language,
        "confidence": result.confidence,
        "has_document": result.has_document,
        "document_links": result.document_links,
        "sources_count": result.raw_chunks_used,
        "sources": result.sources,
    }


def ask_simple(query: str) -> str:
    """
    Simplified API that returns just the answer string.
    """
    return ask(query).get("answer", "")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python llm.py "your question here" '
              '[--faculty FAC] [--department DEPT] [--top-k N] [--debug]')
        sys.exit(1)

    query = sys.argv[1]
    faculty = None
    department = None
    top_k = 8

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--faculty" and i + 1 < len(sys.argv):
            faculty = sys.argv[i + 1]
            i += 2
        elif arg == "--department" and i + 1 < len(sys.argv):
            department = sys.argv[i + 1]
            i += 2
        elif arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
            i += 2
        elif arg == "--debug":
            logging.getLogger().setLevel(logging.DEBUG)
            i += 1
        else:
            i += 1

    result = ask(query, top_k=top_k, faculty=faculty, department=department)

    print("\n" + "=" * 64)
    print(f"QUERY      : {query}")
    print(f"LANGUAGE   : {result['language']}")
    print(f"CONFIDENCE : {result['confidence']}")
    print(f"SOURCES    : {result['sources_count']}")
    if result['has_document']:
        print(f"DOCUMENTS  : {', '.join(result['document_links'])}")
    print("=" * 64)
    print("\nANSWER:\n")
    print(result["answer"])
    print("\n" + "=" * 64)