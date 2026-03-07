# Eval Directory (Published Layout)

## 目录说明
- `materials/`: 评测材料（数据集、问题集、ground truth、prompt、清理后的回答集）
- `results/quality_judge/`: 三维质量评测结果
- `results/hitrate_judge/`: 基于 ground truth 命中率评测结果
- `results/overview/`: 四模型双标准总览表
- `docs/`: 评测说明文档
- `archive/legacy_outputs/`: 历史临时产物（保留追溯，不参与发布主引用）

## 发布时建议主引用
1. `docs/EVALUATION_GUIDE.md`
2. `results/overview/model_overview_dual_eval_20260307.csv`
3. `materials/datasets/eval_dataset_raw.json`
4. `materials/questions/eval_questions_only.json`
5. `materials/answers_cleaned/*.json`
