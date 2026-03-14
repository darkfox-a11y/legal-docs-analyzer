# Adaptive Judicial Mode vs Judicial Baseline

## Runs

- Baseline judicial run: `benchmark_results/judicial_scotus_10docs_gemini25flash_2026-03-14/`
- Adaptive judicial run: `benchmark_results/judicial_scotus_10docs_gemini25flash_adaptive_2026-03-14/`

## Aggregate Comparison

- Average latency: `3085.29 ms` -> `3092.11 ms`
- Average composite score: `0.4165` -> `0.4193`
- Average retrieval score: `0.4003` -> `0.4003`
- Average retrieval keyword hit rate: `0.1520` -> `0.1520`
- Average groundedness: `0.2081` -> `0.2047`
- Average answer keyword coverage: `0.3000` -> `0.2960`
- Average question alignment: `0.4420` -> `0.4374`
- Average judge score: `2.58 / 5` -> `2.74 / 5`

## Interpretation

The adaptive judicial mode produced only a small aggregate improvement. The strongest positive change was the Gemini-as-judge score, which rose from `2.58` to `2.74`. Procedural-posture questions also improved modestly. However, the global composite score moved only slightly, latency was effectively unchanged, and some other question types did not improve.

This suggests that prompt specialization alone helps a bit, but the main remaining bottleneck for court decisions is still retrieval quality and section targeting rather than generic answer phrasing.
