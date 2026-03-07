# Eval Workspace（VisionTree / MiroSearch）

## 目录定位
`eval/` 是本项目的发布级评测工作区，用于管理：
- 评测材料（数据集、问题集、ground truth、外部 prompt、四模型清洗答案）
- 评测结果（质量维度 + 命中率维度）
- 评测说明与代表性案例
- 历史临时产物归档

## 目录结构
- `materials/`: 评测输入材料
- `results/quality_judge/`: 三维质量评测结果
- `results/hitrate_judge/`: ground truth 命中率评测结果
- `results/overview/`: 四模型双标准总览表
- `docs/`: 评测方法与案例文档
- `archive/legacy_outputs/`: 历史临时文件（不参与发布主引用）

## 发布主引用清单
1. `docs/EVALUATION_GUIDE.md`
2. `results/overview/model_overview_dual_eval_20260307.csv`
3. `materials/datasets/eval_dataset_raw.json`
4. `materials/questions/eval_questions_only.json`
5. `materials/answers_cleaned/*.json`
6. `docs/REPRESENTATIVE_CASE_A009.md`

## 一键复现实用建议
- 先用 `materials/*` 作为唯一输入。
- 再运行评测脚本输出到 `results/*`。
- 最后仅发布 `results/overview/* + docs/*` 作为对外口径。
