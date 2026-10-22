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

## Reference

This project is motivated by the following paper:

> DG Paul, H Zhu, I Bayley (2024). *Sceneval: A benchmark for scenario-based evaluation of code generation*. 2024 IEEE International Conference. https://ieeexplore.ieee.org/abstract/document/10685215/

Paper author: Prof. Hong Zhu, Oxford Brookes University (hzhu@brookes.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
