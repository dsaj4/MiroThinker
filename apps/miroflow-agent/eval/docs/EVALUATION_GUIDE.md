# MiroSearch Evaluation Guide

## 1. Scope
This document defines the reproducible evaluation workflow for four answer providers:
- `mirosearch` (this project)
- `kimi`
- `minimax`
- `glm`

It keeps two evaluation standards in parallel:
- **Quality Judge**: relevance / information density / model alignment
- **Ground-Truth Hit Judge**: HIT / PARTIAL / MISS and hit score

## 2. Evaluation Material Generation Workflow
The material pipeline is fixed as follows:
1. Use `materials/prompts/prompt_dataset.md` to instruct an external agent to generate the evaluation dataset.
2. Use `materials/prompts/prompt_search.md` to instruct external search agents to produce comparison answer sets.
3. Run the local evaluation scripts to score all answer sets under the same judge standards.

This matches the required process:
- `prompt_dataset` -> external agent produces dataset
- `prompt_search` -> external agent produces comparison answers
- local evaluation system performs scoring

## 3. Canonical Input Assets
- Raw dataset: `materials/datasets/eval_dataset_raw.json`
- Working dataset: `materials/datasets/eval_dataset_working.json`
- Questions only: `materials/questions/eval_questions_only.json`
- Ground truth map: `materials/ground_truth/eval_ground_truth_map.json`
- Cleaned answer sets:
  - `materials/answers_cleaned/model_mirosearch_answers_cleaned.json`
  - `materials/answers_cleaned/model_kimi_answers_cleaned.json`
  - `materials/answers_cleaned/model_minimax_answers_cleaned.json`
  - `materials/answers_cleaned/model_glm_answers_cleaned.json`

## 4. Local Scoring Process
### 4.1 Quality Judge
- Script: `eval/judge_existing_results.py`
- Output location: `results/quality_judge/`
- Metrics:
  - `relevance_score`
  - `density_score`
  - `alignment_score`

### 4.2 Ground-Truth Hit Judge
- Script: `eval/judge_existing_results_hit.py`
- Output location: `results/hitrate_judge/`
- Metrics:
  - `hit_label` (`HIT`, `PARTIAL`, `MISS`)
  - `hit_score` (0-10)
  - `hit_reason`

## 5. Final Published Results
- Per-model quality results: `results/quality_judge/*.csv`
- Per-model hit-rate results: `results/hitrate_judge/*.csv`
- Cross-model dual-standard overview: `results/overview/model_overview_dual_eval_20260307.csv`

## 6. Fairness Constraints
To keep evaluation fair:
- Same question set for all models.
- Same judge model family and scoring rubric for all models.
- Same parsing and post-processing rules for all outputs.
- No model-specific prompt tweaks during scoring.

## 7. Archive Policy
Historical temporary outputs are moved to `archive/legacy_outputs/`.
Published assets remain in `materials/` and `results/`.
