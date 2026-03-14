# Judicial Benchmark Summary

This run evaluates the Legal Document Analyzer on a separate judicial corpus made from 10 official U.S. Supreme Court PDFs, including merits opinions and opinions relating to orders.

## Corpus

- Source list: `benchmark_data/judicial_sources_scotus.json`
- Downloaded corpus: `benchmark_data/judicial_scotus_corpus/`
- Question set: `benchmark_data/judicial_questions.json`
- Model: `gemini-2.5-flash`
- Top K: `3`
- Total benchmark cases: `50`

## Aggregate Results

- Average latency: `3085.29 ms`
- Average composite score: `0.4165`
- Average retrieval score: `0.4003`
- Average retrieval keyword hit rate: `0.1520`
- Average groundedness: `0.2081`
- Average answer keyword coverage: `0.3000`
- Average question alignment: `0.4420`
- Generic answer rate: `0.0000`
- Average judge score: `2.58 / 5`

## By Question Type

- `case_identity`: latency `2866.05 ms`, composite `0.5451`
- `procedural_posture`: latency `3138.03 ms`, composite `0.3528`
- `legal_issue`: latency `3302.78 ms`, composite `0.4598`
- `holding_or_disposition`: latency `2885.72 ms`, composite `0.3806`
- `relief_and_next_steps`: latency `3233.89 ms`, composite `0.3442`

## Interpretation

The system remains responsive on real court documents, but judicial reasoning tasks are materially harder than the earlier policy/contract-style corpus. Case-name extraction performed best, while procedural posture and relief-oriented questions were weaker. This suggests the benchmark is now much closer to a true legal-document workload and should be treated as the stronger experimental evidence for court-style RAG evaluation.
