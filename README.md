# Equal Tokens, Unequal Evidence

**Separating Evidence Loss from Evidence Use in Multilingual LLMs**

![Status: In Development](https://img.shields.io/badge/Status-In%20Development-orange)
![Target: ACL or EMNLP](https://img.shields.io/badge/Target-ACL%20%2F%20EMNLP-blue)

---

## Project objective

This project investigates how fixed input-token budgets affect multilingual reading comprehension. It measures how often truncation excludes the relevant passage and how model performance changes when that passage remains available. The study compares equal token budgets with matched retained content, using relevant-passage-only baselines to account for differences in task performance across languages.

## Datasets

- **Belebele** provides aligned reading comprehension in English, Bangla, Hindi, Urdu, Nepali, Sinhala, Southern Pashto, and Tamil for controlled passage-packing experiments.
- **MLQA** provides paired English and Hindi questions with annotated answer spans for natural-context evaluation.
- **TyDiQA-GoldP** provides independently authored Bangla and English questions for a natural-context replication within each language.

## Models

Planned checkpoints, subject to pilot validation:

- **Qwen2.5-3B-Instruct**
- **Gemma 3 4B IT**
- **Aya Expanse 8B**

## Status

The project is under active development, targeting a future ACL or EMNLP submission.
