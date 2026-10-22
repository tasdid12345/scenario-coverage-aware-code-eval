# Scenario-Coverage Gradient Boosting vs. Per-Scenario Aggregate Baseline on Adult

Motivation: Scenario-based evaluation, as in "Sceneval: A benchmark for scenario-based evaluation of code generation" (2024), argues that aggregate scores can hide scenarios where a model is under-served. This experiment asks whether a model that explicitly reasons about scenario coverage can surface such scenarios better than a standard aggregate report.

Question: Can a class-balanced, cross-validated gradient-boosting model that explicitly models scenario coverage identify under-served scenarios better than a standard per-scenario aggregate baseline on a public stand-in dataset?

Dataset: UCI Adult income (via the `ucimlrepo` package). This is a stand-in, not the paper's own dataset. Rows are test cases; scenarios are built by a deterministic rule on education-level bands and sex. A stratified 70/30 split uses seed 42.

Baseline vs. improved: The baseline is a cross-validated logistic regression that reports per-scenario aggregate macro-F1. The improved model is a class-balanced, CV-tuned LightGBM with scenario-coverage features (scenario size and per-scenario class balance), averaged with the baseline using a CV-selected weight. The main metric is worst-scenario macro-F1 on the test set.

How to run:
```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts saved to `figures/`).

## Results (latest run)

On the public UCI Adult Income stand-in, a logistic regression baseline reached worst-scenario macro-F1 of 59% on the held-out test set, while my class-balanced LightGBM with scenario-coverage features reached 70% (single split).

| Metric | Value |
|---|---|
| baseline_score | 0.5943 |
| improved_score | 0.6966 |
| improvement | 0.1023 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| baseline_f1 | 0.7827 |
| improved_f1 | 0.8179 |
| baseline_n_below_threshold | 3 |
| improved_n_below_threshold | 1 |
| n_scenarios_evaluated | 6 |
| under_served_threshold | 0.7 |
| ensemble_weight_lgbm | 0.5 |
| runtime_seconds | 90.86 |

Run on 2026-10-22. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Scenarios are a constructed rule on Adult attributes, not a real code-generation benchmark; the paper's reported result is not reproduced. Only one train/test split; seed-based std covers model randomness only. Threshold for under-served scenarios is an arbitrary choice. Small scenarios (<50 test rows) are excluded from the main metric.

## Reference

This project is motivated by the following paper:

> DG Paul, H Zhu, I Bayley (2024). *Sceneval: A benchmark for scenario-based evaluation of code generation*. 2024 IEEE International Conference. https://ieeexplore.ieee.org/abstract/document/10685215/

Paper author: Prof. Hong Zhu, Oxford Brookes University (hzhu@brookes.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
