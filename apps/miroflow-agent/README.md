# MiroSearch（VisionTree 搜索与参数引擎）

> 本文档聚焦“主流程与评测结论”。详细评测材料见 [`eval/`](./eval/README.md)。

## 一、核心定位与技术底座：思维模型的数据引擎

市面上的套壳 AI 多在解决“搜什么”的表层问题，而本项目从架构起点就是为 VisionTree 的核心业务“思维模型”设计：它不是独立搜索框，而是驱动“参数级推演”的数据引擎。

### 1) 无缝对接主流程：寻找真数据、深数据

主业务链路是：`选择模型 -> 定义概念 -> 确定参数`。  
最难环节是“确定参数”（如收益矩阵、节点权重、约束强度）。MiroSearch 的设计目标就是补这一断点：自动拆解模型骨架，执行深搜与长文抓取，为参数提供真实、可追溯、可复核的数据支撑。

### 2) 搭载全项目记忆系统：灵魂统一的进化

系统内置持久记忆（`memory/SOUL.md`、`memory/MEMORY.md`、`memory/daily/*`），并支持语义检索回忆。  
每次任务都会沉淀“可复用检索经验 + 新证据链”，使策略随使用迭代，形成与 VisionTree 业务逻辑一致的长期演进能力。

### 3) 高容错调度大脑（Orchestrator）

在死链、空结果、格式偏差、反爬拦截等场景下，调度器会自动触发回滚、改写查询、二次检索或降级路径，避免死循环，保障长链路任务稳定收敛。

---

## 二、评测基准设计：如何检验参数级搜索能力

我们构建的是“强制模型推演与逆向生成流”评测，不是闲聊问答集。每个问题都满足以下标准：

- 真实性：答案依据来自近 1-2 年真实研报、法规、技术文档或财务数据。
- 模型绑定性：要求按特定思维模型视角推理（如结构洞、收益矩阵），禁止泛化叙述。
- 参数断点填补：要求检索补出常识无法直接推出的关键参数。

评测流程：
1. 使用 `prompt_dataset` 生成评测数据集。
2. 使用 `prompt_search` 生成外部模型对比答案。
3. 由本地评测系统执行双标准打分（质量维度 + ground truth 命中率）。

---

## 三、效能、成本与体验透视：系统真实账本

在高难度“思维模型参数题”数据集上的账本如下：

| 评测维度 | 指标定义 | 实测平均数据（自研） | 业务透视与核心价值 |
|---|---|---:|---|
| 时间响应 | 复杂问题响应时间 | **88.83 秒** | 最低 28 秒。 |
| 算力成本 | 单次查询平均 Token | **34,822 个** | 输入大体量资料，输出参数化结论。 |
| 算力成本 | 单次查询预估费用 | **0.0985 元** | 单次深度研报级抽取成本低于 1 毛。 |
| 系统稳定性 | 深度抓取成功率 | **84.21%** | 主要失败来自强反爬与严重超时。 |
| 召回能力 | 平均有效网页源数量 | **13.46 个** | 单次最多深读 34 篇长文，支撑参数推演。 |

---

## 四、核心竞争力：多维对比评测（vs 国内一线模型）

同数据集、同评测流程下四模型对比：

| 模型 | Relevance | Information Density | Model Alignment | Hit Score | Full Hit Rate | Partial Hit Rate | Miss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kimi | 6.20 | **7.15** | **5.50** | 2.95 | 10.00% | 15.00% | 75.00% |
| Minimax | 6.30 | 5.65 | 3.75 | 3.20 | 5.00% | 25.00% | 70.00% |
| GLM | **7.75** | 6.95 | 5.30 | **4.50** | 5.00% | **45.00%** | **50.00%** |
| VisionTree（本项目） | 6.80 | 5.60 | 4.10 | 4.35 | **20.00%** | 25.00% | 55.00% |

优势定位：
- 综合事实召回与命中表现进入第一梯队。
- 完全命中率领先，体现“深度抓取 + 证据闭环”能力。
- 结构化行文风格仍有持续优化空间。

---

## 五、深度案例剖析：A_009（结构洞推演）

案例文档：[`eval/docs/REPRESENTATIVE_CASE_A009.md`](./eval/docs/REPRESENTATIVE_CASE_A009.md)

在“全球数据合规网络结构洞”问题中，本项目相对 Kimi / Minimax / GLM 命中 3 个独有专业来源：

- https://rouse.com/insights/news/2025/data-localisation-and-transfer-issues-in-southeast-asia-what-businesses-need-to-know
- https://www.lexology.com/library/detail.aspx?g=c1ad463a-8af6-4637-a958-224ccfe0c18b
- https://bigid.com/blog/complying-with-the-doj-rule-on-cross-border-data-transfers/

该案例体现了主流程的三段能力：
1. 先建模（节点/边权/结构洞定义）；
2. 再锚定关键协定（EU-Singapore DTA）；
3. 最后深潜专业信源补全边权参数并收敛答案。

---

## 六、演进路线

下一阶段重点：

1. 提升运行成功率与稳定性  
围绕反爬、长尾超时、工具格式鲁棒性继续做工程加固。

2. 强化长链路上下文一致性  
进一步降低多轮搜索中的注意力漂移，增强“思维模型约束”在长上下文中的稳定执行。

---

## 附录：快速运行

### 1) 单任务运行

```bash
uv run python main.py llm=qwen-3 agent=mirothinker_v1.5_keep5_max200 llm.base_url=http://localhost:61002/v1
```

### 2) 启动 Mirosearch 服务

```bash
uv run python mirosearch_server.py
```

### 3) Windows 下使用远程 Milvus（可选）

```bash
set MILVUS_URI=http://<milvus-host>:19530
set MILVUS_TOKEN=<optional-token>
uv run python mirosearch_server.py
```

### 4) 评测文档入口

- 评测说明：[`eval/docs/EVALUATION_GUIDE.md`](./eval/docs/EVALUATION_GUIDE.md)
- 四模型双标准总览：[`eval/results/overview/model_overview_dual_eval_20260307.csv`](./eval/results/overview/model_overview_dual_eval_20260307.csv)
