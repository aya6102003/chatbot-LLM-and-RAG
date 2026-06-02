#!/usr/bin/env python3
"""
LLM Generation Layer v2.0 — Farhat Abbas University Sétif 1
============================================================
Advanced, production-ready generation layer with streaming,
intent-aware prompting, and strict context grounding.
"""

import os
import re
import sys
import time
import logging
import requests
from typing import Dict, List, Optional, Generator, Tuple, Any
from dataclasses import dataclass, field

# Import the RAG pipeline and its query analyzer for intent-aware generation
import rag
from rag import QueryUnderstanding

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Llama 3.1 8B context window is 128k tokens. 
# We limit context to ~8000 chars to ensure fast, focused generation without getting lost.
MAX_CONTEXT_CHARS = 8000 
GENERATION_TIMEOUT = 120  # seconds

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────
@dataclass
class Source:
    title: str
    url: str
    doc_type: str = ""
    chunk_score: float = 0.0

@dataclass
class GenerationResult:
    answer: str
    intent: str
    language: str
    sources: List[Source]
    fallback_used: bool = False

# ─────────────────────────────────────────────────────────────
# LANGUAGE DETECTION & MAPPING
# ─────────────────────────────────────────────────────────────
_RE_AR = re.compile(r"[\u0600-\u06FF]")
_RE_FR = re.compile(r"\b(le|la|les|de|du|des|et|en|un|une|pour|avec|dans|sur|est)\b", re.IGNORECASE)

LANG_CONFIG = {
    "ar": {"name": "Arabic", "instruction": "أجب باللغة العربية فقط.", "no_data": "عذراً، لا توجد معلومات موثوقة حول هذا الموضوع في قاعدة البيانات المتاحة."},
    "fr": {"name": "French", "instruction": "Répondez UNIQUEMENT en français.", "no_data": "Désolé, aucune information fiable sur ce sujet n'a été trouvée dans les données disponibles."},
    "en": {"name": "English", "instruction": "Respond ONLY in English.", "no_data": "Sorry, no reliable information about this topic was found in the available data."}
}

def detect_language(text: str) -> str:
    s = text[:200]
    if len(_RE_AR.findall(s)) > 3: return "ar"
    if len(_RE_FR.findall(s)) > 2: return "fr"
    return "en"

# ─────────────────────────────────────────────────────────────
# QUERY PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_query(query: str) -> str:
    q = query.strip().strip('"\'«»')
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"[?！？]+$", "", q).strip()
    return q

# ─────────────────────────────────────────────────────────────
# CONTEXT TRUNCATION (Prevents Token Overflow)
# ─────────────────────────────────────────────────────────────
def truncate_context(llm_context: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(llm_context) <= max_chars:
        return llm_context
    
    # Find a safe cut-off point (end of a chunk block)
    truncated = llm_context[:max_chars]
    # The separator in rag.py is "─" * 55
    last_sep = truncated.rfind("\n─")
    if last_sep > max_chars * 0.5: # Only cut at separator if it doesn't remove too much
        return truncated[:last_sep]
    return truncated + "\n\n[Context truncated...]"

# ─────────────────────────────────────────────────────────────
# INTENT-AWARE PROMPT ENGINEERING
# ─────────────────────────────────────────────────────────────
class PromptBuilder:
    def __init__(self, intent: str, lang: str):
        self.intent = intent
        self.lang = lang
        self.cfg = LANG_CONFIG.get(lang, LANG_CONFIG["en"])

    def build(self, query: str, context: str) -> Tuple[str, str]:
        system = self._system()
        user = self._user(query, context)
        return system, user

    def _system(self) -> str:
        base = f"""You are an expert university assistant for Farhat Abbas University Sétif 1.
CRITICAL RULES:
1. {self.cfg['instruction']}
2. GROUNDING: Base your answer EXACTLY on the provided context. Do not hallucinate or use outside knowledge.
3. FORMAT: Use Markdown for formatting (bolding, lists, tables if necessary).
4. SOURCE INTEGRATION: If a chunk ends with "Source: [URL]", include that link naturally in your response."""
        
        intent_rules = {
            "person_lookup": """5. INTENT (Person Lookup): Extract specific details: Full Name, Title (Prof/Dr), Department, Faculty, Email, Office/Bureau, and Phone. Format as a clean profile card. If info is missing, omit that field—do not guess.""",
            
            "course_query": """5. INTENT (Course Info): Extract Course Name, Department, Credits, Teacher(s), and Syllabus/Description if available. Structure logically.""",
            
            "admin_query": """5. INTENT (Administrative): Extract precise steps, deadlines, required documents, and specific offices/links. Use bullet points for procedures.""",
            
            "translation": f"""5. INTENT (Translation): Translate the provided text into {self.cfg['name']}. Maintain the original meaning, tone, and structure perfectly.""",
            
            "table_query": """5. INTENT (Table/List): Extract the requested list or table from the context. Format strictly as a Markdown table or bulleted list."""
        }
        
        return base + "\n" + intent_rules.get(self.intent, "")

    def _user(self, query: str, context: str) -> str:
        if not context or context == "No relevant context available.":
            return f"CONTEXT: Empty\n\nQUESTION: {query}\n\nANSWER: {self.cfg['no_data']}"
        
        return f"""CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

# ─────────────────────────────────────────────────────────────
# MAIN LLM GENERATOR CLASS
# ─────────────────────────────────────────────────────────────
class LLMGenerator:
    def __init__(self):
        self.qu = QueryUnderstanding()
        self.model = OLLAMA_MODEL
        self.available = self._check_ollama()
        if not self.available:
            log.error("Ollama is not reachable. LLM generation will fail.")

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _extract_sources(self, results: List[Dict]) -> List[Source]:
        sources = []
        seen_urls = set()
        for r in results:
            url = r.get("pdf_url") or r.get("url")
            if url and url not in seen_urls:
                sources.append(Source(
                    title=r.get("title", "Untitled Document"),
                    url=url,
                    doc_type=r.get("metadata", {}).get("doc_type", ""),
                    chunk_score=r.get("score", 0.0)
                ))
                seen_urls.add(url)
        return sources

    def _is_document_only(self, results: List[Dict]) -> Tuple[bool, str]:
        """Deterministic check for Case 3: Answer is just a file."""
        if not results: return False, ""
        top = results[0]
        meta = top.get("metadata", {})
        text = top.get("text", "").strip()
        url = meta.get("pdf_url") or meta.get("url")
        
        is_doc = "pdf" in meta.get("doc_type", "").lower() or "document" in meta.get("doc_type", "").lower()
        if is_doc and len(text) < 200 and url:
            return True, url
        return False, ""

    def generate(self, query: str, faculty: Optional[str] = None, department: Optional[str] = None) -> GenerationResult:
        """Synchronous generation wrapper."""
        answer_chunks = list(self.stream(query, faculty, department))
        full_answer = "".join(answer_chunks)
        
        # Fallback if streaming failed or yielded nothing
        if not full_answer:
            lang = detect_language(query)
            return GenerationResult(
                answer=LANG_CONFIG[lang]["no_data"],
                intent="general_info", language=lang, sources=[], fallback_used=True
            )
            
        # Re-parse intent for the final result object (since stream populates it)
        analysis = self.qu.analyze(query)
        return GenerationResult(
            answer=full_answer,
            intent=analysis["intent"],
            language=analysis["language"],
            sources=self._last_sources,
            fallback_used=False
        )

    def stream(self, query: str, faculty: Optional[str] = None, department: Optional[str] = None) -> Generator[str, None, None]:
        """Streaming generator. Yields string chunks of the answer."""
        self._last_sources = [] # Hack to pass sources from stream to generate()
        
        # 1. Analyze Query
        clean_q = preprocess_query(query)
        analysis = self.qu.analyze(clean_q)
        lang = analysis["language"]
        intent = analysis["intent"]

        # 2. Retrieve Context
        try:
            results = rag.retrieve_for_llm(
                query=clean_q, top_k=5, faculty=faculty, department=department
            )
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            yield LANG_CONFIG[lang]["no_data"]
            return

        self._last_sources = self._extract_sources(results)

        # 3. Handle No Context (Case 5)
        if not results:
            yield LANG_CONFIG[lang]["no_data"]
            return

        # 4. Handle Document-Only Results (Case 3)
        is_doc, doc_url = self._is_document_only(results)
        if is_doc:
            title = results[0].get("title", "Document")
            link_map = {
                "ar": f"يمكنك تحميل أو الوصول إلى المستند ('{title}') مباشرة من هنا: {doc_url}",
                "fr": f"Vous pouvez télécharger ou accéder au document ('{title}') directement ici : {doc_url}",
                "en": f"You can download or access the document ('{title}') directly here: {doc_url}"
            }
            yield link_map.get(lang, link_map["en"])
            return

        # 5. Build Prompt
        context_str = truncate_context(results[0].get("llm_context", ""))
        builder = PromptBuilder(intent=intent, lang=lang)
        system_prompt, user_prompt = builder.build(clean_q, context_str)

        # 6. Stream from Ollama
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": True,
            "options": {
                "temperature": 0.2,  # Low temp for factual grounding
                "top_p": 0.85
            }
        }

        try:
            with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=GENERATION_TIMEOUT) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line: continue
                    chunk = line.decode("utf-8")
                    # Ollama streams JSON objects
                    if chunk.startswith("data: "):
                        json_str = chunk[6:]
                        try:
                            data = json.loads(json_str)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.Timeout:
            log.error("Ollama streaming timed out")
            yield "\n\n[Error: Generation timed out]"
        except Exception as e:
            log.error(f"Ollama streaming failed: {e}")
            yield LANG_CONFIG[lang]["no_data"]

# ─────────────────────────────────────────────────────────────
# CLI INTERFACE (Streaming Output)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python llm.py "Your question here" [--faculty FAC] [--department DEPT]')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:]) # Simple joining to handle quotes nicely
    fac, dept = None, None
    
    # Parse args properly
    args = sys.argv[1:]
    i = 0
    q_parts = []
    while i < len(args):
        if args[i] == "--faculty" and i + 1 < len(args):
            fac = args[i+1]; i += 2
        elif args[i] == "--department" and i + 1 < len(args):
            dept = args[i+1]; i += 2
        else:
            q_parts.append(args[i]); i += 1
            
    user_query = " ".join(q_parts)

    if not user_query:
        print("Error: No query provided.")
        sys.exit(1)

    print(f"Query: {user_query}")
    print("=" * 60)
    
    generator = LLMGenerator()
    
    start_time = time.time()
    # Stream directly to stdout for beautiful CLI UX
    for chunk in generator.stream(user_query, faculty=fac, department=dept):
        print(chunk, end="", flush=True)
        
    elapsed = time.time() - start_time
    
    # Print sources nicely at the bottom
    if generator._last_sources:
        print("\n\n" + "─" * 60)
        print("📚 Sources:")
        for src in generator._last_sources:
            print(f"  • {src.title}: {src.url}")
            
    print("\n" + "─" * 60)
    print(f"⏱️  Generated in {elapsed:.2f}s")