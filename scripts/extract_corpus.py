#!/usr/bin/env python3
"""
Extract text from Rebel project documents (Word, PDF, PPT).
Output: data/corpus.jsonl with text + metadata per document.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib

# Document extraction libraries
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None


# === Configuration ===

SOURCE_DIR = Path(r"C:\Users\David.Olmer\Rebelgroup\REP - Documents\1. Projecten")
OUTPUT_DIR = Path(__file__).parent.parent / "data"
FINAL_PATTERN = re.compile(r'(?i)(final|definitief|eindrapport|rapport.*v\d|deliverable)')
PROJECT_PATTERN = re.compile(r'^P(\d+)')


# === Text Extraction ===

def extract_docx(path: Path) -> Optional[str]:
    """Extract text from Word document."""
    if DocxDocument is None:
        return None
    try:
        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"  WARN: Failed to extract {path.name}: {e}", file=sys.stderr)
        return None


def extract_pdf(path: Path) -> Optional[str]:
    """Extract text from PDF."""
    if fitz is None:
        return None
    try:
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"  WARN: Failed to extract {path.name}: {e}", file=sys.stderr)
        return None


def extract_pptx(path: Path) -> Optional[str]:
    """Extract text from PowerPoint."""
    if Presentation is None:
        return None
    try:
        prs = Presentation(str(path))
        text_parts = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_parts.append(shape.text)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"  WARN: Failed to extract {path.name}: {e}", file=sys.stderr)
        return None


def extract_text(path: Path) -> Optional[str]:
    """Extract text based on file extension."""
    ext = path.suffix.lower()
    if ext == ".docx":
        return extract_docx(path)
    elif ext == ".pdf":
        return extract_pdf(path)
    elif ext == ".pptx":
        return extract_pptx(path)
    return None


# === Metadata Extraction ===

def extract_project_number(path: Path) -> Optional[str]:
    """Extract project number from path (P12345 format)."""
    for part in path.parts:
        match = PROJECT_PATTERN.match(part)
        if match:
            return f"P{match.group(1)}"
    return None


def extract_project_name(path: Path) -> Optional[str]:
    """Extract project name from folder."""
    for part in path.parts:
        if PROJECT_PATTERN.match(part):
            # Format: "P12345 - Project Name"
            if " - " in part:
                return part.split(" - ", 1)[1]
            return part
    return None


def infer_document_type(path: Path, text: str) -> str:
    """Infer document type from filename and content."""
    name_lower = path.name.lower()

    if "mkba" in name_lower or "kosten" in name_lower and "baten" in name_lower:
        return "mkba"
    elif "evaluatie" in name_lower:
        return "evaluatie"
    elif "rapport" in name_lower or "eindrapport" in name_lower:
        return "rapport"
    elif "presentatie" in name_lower or path.suffix.lower() == ".pptx":
        return "presentatie"
    elif "offerte" in name_lower:
        return "offerte"
    elif "notitie" in name_lower or "memo" in name_lower:
        return "notitie"
    else:
        return "overig"


def compute_hash(text: str) -> str:
    """Compute MD5 hash of text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# === Main Pipeline ===

def find_final_documents(source_dir: Path) -> list[Path]:
    """Find all final/definitief documents."""
    documents = []

    for ext in ["*.docx", "*.pdf", "*.pptx"]:
        for path in source_dir.rglob(ext):
            if FINAL_PATTERN.search(path.name):
                documents.append(path)

    return documents


def process_documents(documents: list[Path]) -> list[dict]:
    """Process documents and extract text + metadata."""
    results = []
    seen_hashes = set()

    for i, path in enumerate(documents):
        print(f"[{i+1}/{len(documents)}] Processing: {path.name}")

        text = extract_text(path)
        if not text or len(text.strip()) < 100:
            print(f"  SKIP: Too short or failed extraction")
            continue

        # Deduplicate
        text_hash = compute_hash(text)
        if text_hash in seen_hashes:
            print(f"  SKIP: Duplicate content")
            continue
        seen_hashes.add(text_hash)

        # Build record
        record = {
            "id": text_hash,
            "filename": path.name,
            "path": str(path),
            "project_number": extract_project_number(path),
            "project_name": extract_project_name(path),
            "document_type": infer_document_type(path, text),
            "format": path.suffix.lower()[1:],
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()),
            "extracted_at": datetime.now().isoformat(),
        }

        results.append(record)
        print(f"  OK: {record['word_count']} words, type={record['document_type']}")

    return results


def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("Rebel Corpus Extraction")
    print("=" * 60)

    # Check dependencies
    missing = []
    if DocxDocument is None:
        missing.append("python-docx")
    if fitz is None:
        missing.append("PyMuPDF")
    if Presentation is None:
        missing.append("python-pptx")

    if missing:
        print(f"WARNING: Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))

    # Find documents
    print(f"\nSearching in: {SOURCE_DIR}")
    documents = find_final_documents(SOURCE_DIR)
    print(f"Found {len(documents)} final documents")

    if not documents:
        print("No documents found. Check SOURCE_DIR path.")
        sys.exit(1)

    # Process
    print(f"\nExtracting text...")
    results = process_documents(documents)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "corpus.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n" + "=" * 60)
    print(f"Extracted {len(results)} documents")
    print(f"Output: {output_path}")

    # Stats
    by_type = {}
    for r in results:
        t = r["document_type"]
        by_type[t] = by_type.get(t, 0) + 1

    print(f"\nBy type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    total_words = sum(r["word_count"] for r in results)
    print(f"\nTotal words: {total_words:,}")


if __name__ == "__main__":
    main()
