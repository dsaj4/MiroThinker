# MEMORY

## Project Memory
- This file stores long-term project-level constraints and persistent preferences.
- Task-specific execution traces should be stored in daily logs under `daily/`.

## VisionTree Project Core Preferences
- Enable deep aggregation of multi-modal fragmented information
- Support automatic mapping of information into mental models
- Maintain persistent memory of workflow and analysis parameters
- Provide semantic retrieval for previous analyses

## Models Initialized
### 网络模型 (Network Models)
- Applicable Fields: 生活管理, 行为改变, 时间管理, 自我复盘
- Default Parameters:
  - nodes: ["熬夜", "拖延", "饮食不规律", "运动不持续", "房间杂乱"]
  - edges: ["熬夜导致拖延", "饮食不规律影响运动状态", "房间杂乱影响心情导致拖延", "熬夜影响运动状态", "饮食不规律导致状态差"]
  - weight: [1,2,3,4,5]
- Model Risks:
  - 枢纽错觉：关键节点可能只是信息汇聚点
  - 降维盲区：关系简化可能掩盖心理阻力
  - 静态失效：节点性质随环境变化可能失效
  - 闭环强化：初始权重偏见可能强化既有认知
  - 边界误判：分析快照受限，可能遗漏隐藏因素

### SWOT分析 (Structured Framework)
- Applicable Fields: 企业战略, 产品管理, 市场分析
- Default Parameters:
  - S: []
  - W: []
  - O: []
  - T: []
  - source: ["行业报告", "官方文档", "竞争分析", "新闻资讯"]
- Analysis Preferences:
  - 强调内部-外部交叉策略生成
  - 输出边界明确，提供风险提示
  - 支持后续任务迭代和策略更新