import json
import shutil
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path("./university_farhat_abaas/fsciences")

PAGES_DIR  = BASE_DIR / "pages"
DOCS_DIR   = BASE_DIR / "docs"
PDFS_DIR   = BASE_DIR / "pdfs"
IMAGES_DIR = BASE_DIR / "images"
TABLES_DIR = BASE_DIR / "tables"

OUTPUT_DIR = BASE_DIR / "clean_dataset"

OUT_PAGES  = OUTPUT_DIR / "pages"
OUT_DOCS   = OUTPUT_DIR / "docs"
OUT_PDFS   = OUTPUT_DIR / "pdfs"
OUT_IMAGES = OUTPUT_DIR / "images"
OUT_TABLES = OUTPUT_DIR / "tables"
OUT_LOGS   = OUTPUT_DIR / "logs"

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BLOCKED_URL = "/production_pedagogiques"

MIN_TEXT_CHARS = 80

# ─────────────────────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unquote(text)
    text = text.lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    text = re.sub(r"[_\.\-/\\]+", " ", text)

    text = re.sub(r"([a-z])([A-Z])", r"\\1 \\2", text)

    text = re.sub(r"(\\d)([a-z])", r"\\1 \\2", text)
    text = re.sub(r"([a-z])(\\d)", r"\\1 \\2", text)

    text = re.sub(r"\\s+", " ", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────
# COURSE KEYWORDS
# ─────────────────────────────────────────────────────────────

COURSE_KEYWORDS = {
    "cours",
    "course",
    "lecture",
    "lesson",
    "chapter",
    "chapitre",
    "module",
    "support",
    "book",
    "ebook",
    "manuel",
    "notes",
    "slides",
    "presentation",
    "resume",
    "summary",
    "polycopie",
    "polycopy",
    "methode",
    "technique",
    "td",
    "tp",
    "tutorial",
    "exercise",
    "exercises",
    "exercice",
    "exercices",
    "serie",
    "series",
    "worksheet",
    "problem",
    "lab",
    "practical",
    "exam",
    "test",
    "quiz",
    "controle",
    "interrogation",
    "midterm",
    "final",
    "rattrapage",
    "sujet",
    "corrige",
    "solution",
    "annale",
    "annales",
    "devoir",
    "assignment",
    "homework",
    "project",
    "projet",
    "rapport",
    "محاضرات",
    "دروس",
    "تمارين",
    "اعمال",
}

COMMON_SUBSTRINGS = [
    "cour",
    "chap",
    "td",
    "tp",
    "exo",
    "corr",
    "rattr",
    "annal",
    "interro",
    "devoir",
    "sujet",
    "serie",
    "book",
    "exam",
    "quiz",
    "poly",
    "methode",
    "technique",
    "projet",
    "rapport",
    "manuel",
    "support",
]

NUMBERED_KEYWORD_RE = re.compile(
    r"\\b(td|tp|serie|series|exam|chapitre|chap|seance|module|partie|part|niveau)\\s*\\d+\\b"
)

# ─────────────────────────────────────────────────────────────
# SCHEDULE DETECTION
# ─────────────────────────────────────────────────────────────

SCHEDULE_WORDS = re.compile(
    r"(emploi|horaire|timetable|planning|schedule|"
    r"emploi du temps|emploi des examens|creneau|seance)",
    re.IGNORECASE,
)

DAY_WORDS = re.compile(
    r"(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"الاحد|الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعة|السبت)",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"\\b\\d{1,2}[:h]\\d{0,2}\\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def has_course_keywords(text: str, file_name: str = ""):
    combined = normalize_text(f"{file_name} {text}")

    for kw in COURSE_KEYWORDS:
        if kw in combined:
            return True

    for sub in COMMON_SUBSTRINGS:
        if sub in combined:
            return True

    if NUMBERED_KEYWORD_RE.search(combined):
        return True

    return False


def is_schedule(text: str, tables: list, file_name: str = ""):
    combined = normalize_text(f"{file_name} {text}")

    has_schedule_word = bool(SCHEDULE_WORDS.search(combined))
    has_days          = bool(DAY_WORDS.search(combined))
    has_times         = bool(TIME_PATTERN.search(combined))
    has_tables        = bool(tables)

    score = 0

    if has_schedule_word:
        score += 4

    if has_days:
        score += 2

    if has_times:
        score += 2

    if has_tables:
        score += 1

    return score >= 4


def should_keep_document(text, tables, file_name=""):

    if is_schedule(text, tables, file_name):
        return True

    if has_course_keywords(text, file_name):
        return False

    if len(normalize_text(text)) < MIN_TEXT_CHARS:
        return False

    return False


# ─────────────────────────────────────────────────────────────
# IMAGE FILTER
# ─────────────────────────────────────────────────────────────

def image_has_text(image_meta):
    """
    SIMPLE VERSION:
    keep images only if alt/desc contain text.
    
    Later you can upgrade with OCR.
    """

    alt  = image_meta.get("alt", "")
    desc = image_meta.get("desc", "")

    text = normalize_text(f"{alt} {desc}")

    return len(text) > 5


# ─────────────────────────────────────────────────────────────
# OUTPUT FOLDERS
# ─────────────────────────────────────────────────────────────

for folder in [
    OUT_PAGES,
    OUT_DOCS,
    OUT_PDFS,
    OUT_IMAGES,
    OUT_TABLES,
    OUT_LOGS,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# TRACK DROPPED FILES
# ─────────────────────────────────────────────────────────────

dropped_files = set()

# ─────────────────────────────────────────────────────────────
# PROCESS PAGES
# ─────────────────────────────────────────────────────────────

json_files = list(PAGES_DIR.glob("*.json"))

print(f"FOUND {len(json_files)} JSON FILES")

for json_file in json_files:

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_url = (
            data.get("metadata", {})
                .get("page", {})
                .get("url", "")
        )

        content_text = (
            data.get("content", {})
                .get("text", "")
        )

        resources = data.get("resources", {})

        tables = resources.get("tables", [])

        # ─────────────────────────────────────────
        # DROP production pedagogiques pages
        # ─────────────────────────────────────────

        if BLOCKED_URL in page_url:

            print(f"DROP PAGE: {json_file.name}")

            # track docs
            for doc in resources.get("documents", []):
                local_file = doc.get("local_file")

                if local_file:
                    dropped_files.add(Path(local_file).name)

            # track tables
            for table in tables:
                table_file = table.get("file")

                if table_file:
                    dropped_files.add(Path(table_file).name)

            continue

        # ─────────────────────────────────────────
        # CLEAN IMAGES
        # ─────────────────────────────────────────

        clean_images = []

        for image in resources.get("images", []):

            if image_has_text(image):
                clean_images.append(image)

        resources["images"] = clean_images

        # ─────────────────────────────────────────
        # CLEAN DOCUMENTS
        # ─────────────────────────────────────────

        clean_docs = []

        for doc in resources.get("documents", []):

            title = doc.get("title", "")

            if should_keep_document(
                text=content_text,
                tables=tables,
                file_name=title,
            ):
                clean_docs.append(doc)
            else:
                local_file = doc.get("local_file")

                if local_file:
                    dropped_files.add(Path(local_file).name)

        resources["documents"] = clean_docs

        # ─────────────────────────────────────────
        # SAVE CLEAN PAGE
        # ─────────────────────────────────────────

        output_json = OUT_PAGES / json_file.name

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"ERROR: {json_file.name} -> {e}")

# ─────────────────────────────────────────────────────────────
# COPY VALID FILES
# ─────────────────────────────────────────────────────────────

def copy_valid_files(src_dir, dst_dir):

    if not src_dir.exists():
        return

    for file in src_dir.iterdir():

        if file.name in dropped_files:
            print(f"DROP FILE: {file.name}")
            continue

        shutil.copy2(file, dst_dir / file.name)


copy_valid_files(DOCS_DIR, OUT_DOCS)
copy_valid_files(PDFS_DIR, OUT_PDFS)
copy_valid_files(IMAGES_DIR, OUT_IMAGES)
copy_valid_files(TABLES_DIR, OUT_TABLES)

# ─────────────────────────────────────────────────────────────
# SAVE LOG
# ─────────────────────────────────────────────────────────────

log_file = OUT_LOGS / "dropped_files.txt"

with open(log_file, "w", encoding="utf-8") as f:

    for item in sorted(dropped_files):
        f.write(item + "\\n")

print("\\nFILTERING FINISHED")
print(f"DROPPED FILES: {len(dropped_files)}")
print(f"CLEAN DATASET SAVED TO: {OUTPUT_DIR}")
