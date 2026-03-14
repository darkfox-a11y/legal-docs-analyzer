#!/usr/bin/env python3
"""
Run a multi-model RAG benchmark across a public legal-document corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import google.generativeai as genai

from app.config import settings
from app.rag.benchmark import (
    aggregate_results,
    build_markdown_report,
    load_corpus_manifest,
    load_questions,
    score_benchmark_case,
)
from app.rag.chunking import smart_chunking
from app.rag.embeddings import generate_embeddings
from app.rag.qa import answer_query
from app.rag.vector_store import (
    create_collection_if_not_exists,
    delete_document_chunks,
    store_document_chunks,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmark_data/legal_corpus/manifest.json")
    parser.add_argument("--questions", default="benchmark_data/general_questions.json")
    parser.add_argument("--collection-name", default="legal_documents_benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-documents", type=int, default=12)
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--output-dir", default="benchmark_results")
    return parser.parse_args()


def ingest_documents(manifest: list[dict], collection_name: str) -> None:
    """Chunk, embed, and store the normalized corpus in Qdrant."""
    create_collection_if_not_exists(collection_name)

    for index, document in enumerate(manifest, start=1):
        text = Path(document["text_path"]).read_text(encoding="utf-8")
        chunks = smart_chunking(text, document_type="legal", chunk_size=900, overlap_size=150)
        embeddings = generate_embeddings(chunks, model_name="default", show_progress=False)

        try:
            delete_document_chunks(index, collection_name=collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping pre-delete for document %s in %s: %s", index, collection_name, exc)
        store_document_chunks(
            document_id=index,
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
        )
        document["document_id"] = index
        logger.info("Indexed %s with %s chunks", document["title"], len(chunks))


def clean_json_response(text: str) -> dict:
    """Parse a JSON object even if the model wraps it in fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def judge_answer(question: dict, rag_response: dict, judge_model: str) -> dict:
    """Use Gemini as a judge for groundedness and usefulness."""
    genai.configure(api_key=settings.gemini_api_key)
    context = "\n\n".join(
        source.get("text_preview", source.get("text", ""))
        for source in rag_response.get("sources", [])
    )

    prompt = f"""
You are grading a legal-document RAG answer. Score each item from 1 to 5.
Return JSON only with keys:
groundedness_score, completeness_score, legal_reasoning_score, clarity_score, hallucination_risk_score, summary

Question:
{question["prompt"]}

Expected focus terms:
{", ".join(question.get("expected_keywords", []))}

Retrieved context:
{context}

Answer:
{rag_response.get("answer", "")}
"""
    response = genai.GenerativeModel(judge_model).generate_content(prompt)
    return clean_json_response(response.text)


def main() -> None:
    args = parse_args()
    manifest = load_corpus_manifest(args.manifest)[: args.max_documents]
    questions = load_questions(args.questions)[: args.max_questions]

    ingest_documents(manifest, args.collection_name)

    case_results = []
    for model_name in args.models:
        logger.info("Running benchmark for %s", model_name)
        for document in manifest:
            for question in questions:
                started = time.perf_counter()
                rag_response = answer_query(
                    query=question["prompt"],
                    document_id=document["document_id"],
                    top_k=args.top_k,
                    detail_level="detailed",
                    model_name=model_name,
                    collection_name=args.collection_name,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0

                judge_scores = {}
                try:
                    judge_scores = judge_answer(question, rag_response, args.judge_model)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Judge failed for %s / %s / %s: %s",
                        model_name,
                        document["id"],
                        question["id"],
                        exc,
                    )

                case_results.append(
                    score_benchmark_case(
                        model_name=model_name,
                        document_id=document["id"],
                        document_title=document["title"],
                        question=question,
                        rag_response=rag_response,
                        latency_ms=latency_ms,
                        judge_scores=judge_scores,
                    )
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_payload = aggregate_results(case_results)
    report_json = output_dir / "rag_benchmark_report.json"
    report_md = output_dir / "rag_benchmark_report.md"
    report_json.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown_report(aggregate_payload), encoding="utf-8")

    logger.info("Wrote JSON report to %s", report_json)
    logger.info("Wrote Markdown report to %s", report_md)


if __name__ == "__main__":
    main()
