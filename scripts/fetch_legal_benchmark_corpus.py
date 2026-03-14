#!/usr/bin/env python3
"""
Download and normalize a public legal-document corpus for RAG benchmarking.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import ssl
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import certifi

from app.documents.processing import process_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


class TextExtractor(HTMLParser):
    """Minimal HTML to text converter for legal policy pages."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self.parts)


def infer_suffix(url: str, content_type: str | None) -> str:
    """Infer a file suffix from URL or HTTP content type."""
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".pdf", ".docx", ".doc", ".html", ".htm", ".txt"}:
        return suffix

    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
    if guessed in {".pdf", ".docx", ".doc", ".html", ".htm", ".txt"}:
        return guessed

    return ".html"


def strip_html_to_text(raw_html: bytes) -> str:
    """Convert HTML bytes to plain text."""
    parser = TextExtractor()
    parser.feed(raw_html.decode("utf-8", errors="ignore"))
    text = parser.get_text()
    return re.sub(r"\n{3,}", "\n\n", text)


def download(entry: dict, raw_dir: Path, normalized_dir: Path) -> dict:
    """Download a single entry and save normalized text."""
    request = urllib.request.Request(
        entry["url"],
        headers={"User-Agent": "legal-doc-analyzer-benchmark/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")

    suffix = infer_suffix(entry["url"], content_type)
    raw_path = raw_dir / f"{entry['id']}{suffix}"
    raw_path.write_bytes(payload)

    if suffix == ".txt":
        normalized_text = payload.decode("utf-8", errors="ignore")
    elif suffix in {".html", ".htm"}:
        normalized_text = strip_html_to_text(payload)
    else:
        normalized_text, _, error = process_document(str(raw_path), suffix.lstrip("."))
        if error:
            raise RuntimeError(error)

    normalized_path = normalized_dir / f"{entry['id']}.txt"
    normalized_path.write_text(normalized_text, encoding="utf-8")

    logger.info("Downloaded %s -> %s", entry["title"], normalized_path)

    return {
        "id": entry["id"],
        "title": entry["title"],
        "source_url": entry["url"],
        "tags": entry.get("tags", []),
        "raw_path": str(raw_path),
        "text_path": str(normalized_path),
        "source_format": suffix.lstrip("."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default="benchmark_data/legal_sources.json",
        help="JSON file containing the source URL list.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_data/legal_corpus",
        help="Directory where raw and normalized files should be stored.",
    )
    args = parser.parse_args()

    sources_path = Path(args.sources)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads(sources_path.read_text(encoding="utf-8"))
    manifest = []
    failures = []

    for entry in entries:
        try:
            manifest.append(download(entry, raw_dir, normalized_dir))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", entry["title"], exc)
            failures.append({"id": entry["id"], "title": entry["title"], "error": str(exc)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote normalized manifest to %s", manifest_path)

    if failures:
        failures_path = output_dir / "failures.json"
        failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        logger.warning("Some downloads failed. See %s", failures_path)


if __name__ == "__main__":
    main()
