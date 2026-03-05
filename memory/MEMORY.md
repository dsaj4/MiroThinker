# MEMORY

## Project Memory
- This file stores long-term project-level constraints and persistent preferences.
- Task-specific execution traces should be stored in daily logs under `daily/`.

## Network Analysis Scenario & Model Context
**1. 情景 (Scenario)**
你是一名自由职业者或项目负责人，近期感觉自己工作效率低下，项目进展缓慢。你意识到可能是多种习惯和状态问题交织导致的，但不确定从哪里入手解决最有效。你向系统描述：“我最近项目效率很低，经常熬夜赶工，白天又拖延，饮食不规律，也很久没运动了，整个人很疲惫。”

**2. 模型 (Model)：网络模型 (Network Models)**
- **模型类型：** 结构化框架模型 (Structured Framework)
- **核心思维逻辑：** 网络模型的核心洞察是：节点的价值和行为不仅取决于自身属性，更取决于其在网络中的位置和连接。这一模型帮助我们理解为什么“认识谁”有时比“是谁”更重要，以及如何通过改变关键连接来撬动整个系统。
- **关键概念：**
  - **节点 (Node)：** 影响效率的各个因素（如睡眠、拖延）。
  - **边 (Edge)：** 节点之间的连接关系，即“互相影响/连带改善”关系。
  - **中心性 (Centrality)：** 衡量节点在网络中重要程度的指标。
    - **度中心性 (Degree Centrality)：** 节点拥有的连接数量。对应寻找“一旦改善，能连带改善最多其他因素”的抓手。
    - **介数中心性 (Betweenness Centrality)：** 节点担任“桥梁”的程度。对应寻找“连接身体状态与工作表现/情绪”的关键桥梁。

**3. 基础节点与边定义 (Base Node & Edge Definition)**
- **节点 (Nodes):** 睡眠质量、拖延程度、饮食健康、运动频率、精力水平、环境整洁度、工作压力、社交互动。
- **边 (Edges):** “互相影响/连带改善”关系。例如，睡眠改善 ->精力的提升 -> 拖延的减少。
