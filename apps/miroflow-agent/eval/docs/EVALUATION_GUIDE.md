# MiroSearch 评测说明（VisionTree）

## 1. 评测目标
本评测用于衡量四类答案提供方在“思维模型参数题”上的真实能力：
- `mirosearch`（本项目）
- `kimi`
- `minimax`
- `glm`

采用双标准并行评估：
- 质量维度评测：`relevance / information_density / model_alignment`
- 命中率评测：`HIT / PARTIAL / MISS + hit_score`

## 2. 评测材料生成流程（固定）
1. 使用 `materials/prompts/prompt_dataset.md` 交付外部智能体生成评测数据集。
2. 使用 `materials/prompts/prompt_search.md` 交付外部智能体生成对比答案集。
3. 使用本地评测系统在同一标准下统一打分。

即：
- `prompt_dataset -> 外部智能体生成数据集`
- `prompt_search -> 外部智能体生成对比答案`
- `本地评测系统统一评分`

## 3. 标准输入资产
- 原始数据集：`materials/datasets/eval_dataset_raw.json`
- 工作数据集：`materials/datasets/eval_dataset_working.json`
- 纯问题集：`materials/questions/eval_questions_only.json`
- Ground Truth 映射：`materials/ground_truth/eval_ground_truth_map.json`
- 四模型清洗答案集：`materials/answers_cleaned/*.json`

## 4. 评测执行
### 4.1 质量维度
- 脚本：`eval/judge_existing_results.py`
- 输出：`results/quality_judge/*.csv`
- 指标：
  - `relevance_score`
  - `density_score`
  - `alignment_score`

### 4.2 命中率维度
- 脚本：`eval/judge_existing_results_hit.py`
- 输出：`results/hitrate_judge/*.csv`
- 指标：
  - `hit_label`（`HIT`, `PARTIAL`, `MISS`）
  - `hit_score`（0-10）
  - `hit_reason`

## 5. 最终发布结果
- 分模型质量评测：`results/quality_judge/*.csv`
- 分模型命中率评测：`results/hitrate_judge/*.csv`
- 四模型双标准总览：`results/overview/model_overview_dual_eval_20260307.csv`

## 6. 公平性约束
- 同一问题集。
- 同一评测模型与评分规则。
- 同一结果抽取与后处理策略。
- 评分阶段不做模型特定 prompt 特调。

## 7. 归档策略
历史中间产物统一放入 `archive/legacy_outputs/`。  
发布文档与结果仅引用 `materials/`、`results/`、`docs/`。
