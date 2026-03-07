# VisionTree / MiroSearch

本仓库当前聚焦于 `VisionTree` 的参数级搜索与证据引擎实现，核心交付是 `MiroSearch`：一个面向思维模型推演场景的深度检索服务。它不是通用聊天外壳，而是为“选择模型 -> 定义概念 -> 确定参数”这条业务主链路提供真实数据、深层证据和可复用记忆。

## 核心能力

- 参数级搜索：围绕结构洞、SWOT、收益矩阵等思维模型检索关键参数，而不是泛泛摘要。
- 深度抓取：优先读取法规、财报、技术文档和专业研报，而不是停留在新闻标题层。
- 记忆闭环：通过 `memory/` 中的 `SOUL.md`、`MEMORY.md` 和 `daily/` 形成持续强化的检索策略。
- 鲁棒调度：`Orchestrator` 负责多轮推理、工具调用、重复查询治理、回滚与收敛。
- 可评测：仓库内置完整评测工作区，可对本项目与外部模型进行统一对比。

## 仓库结构

- [apps/miroflow-agent](/d:/Pyproject/MiroThinker/apps/miroflow-agent): 主服务与主流程代码。
- [apps/miroflow-agent/eval](/d:/Pyproject/MiroThinker/apps/miroflow-agent/eval): 发布版评测材料、结果与文档。
- [memory](/d:/Pyproject/MiroThinker/memory): 持久记忆、灵魂文件与每日沉淀。
- [logs](/d:/Pyproject/MiroThinker/logs): 任务运行日志与调试轨迹。
- [libs/miroflow-tools](/d:/Pyproject/MiroThinker/libs/miroflow-tools): MCP 工具与工具管理层。
- `memsearch-main/`: 参考接入的上游记忆系统源码，不属于当前发布主文档口径。

## 快速入口

运行主流程：

```bash
uv run python apps/miroflow-agent/main.py llm=qwen-3 agent=mirothinker_v1.5_keep5_max200 llm.base_url=http://localhost:61002/v1
```

启动 MiroSearch 服务：

```bash
uv run python apps/miroflow-agent/mirosearch_server.py
```

查看评测说明：

- [apps/miroflow-agent/eval/docs/EVALUATION_GUIDE.md](/d:/Pyproject/MiroThinker/apps/miroflow-agent/eval/docs/EVALUATION_GUIDE.md)
- [apps/miroflow-agent/eval/results/overview/model_overview_dual_eval_20260307.csv](/d:/Pyproject/MiroThinker/apps/miroflow-agent/eval/results/overview/model_overview_dual_eval_20260307.csv)

## 推荐阅读顺序

1. [apps/miroflow-agent/README.md](/d:/Pyproject/MiroThinker/apps/miroflow-agent/README.md)
2. [apps/miroflow-agent/PROJECT_STRUCTURE.md](/d:/Pyproject/MiroThinker/apps/miroflow-agent/PROJECT_STRUCTURE.md)
3. [apps/miroflow-agent/eval/docs/EVALUATION_GUIDE.md](/d:/Pyproject/MiroThinker/apps/miroflow-agent/eval/docs/EVALUATION_GUIDE.md)
4. [apps/miroflow-agent/eval/docs/REPRESENTATIVE_CASE_A009.md](/d:/Pyproject/MiroThinker/apps/miroflow-agent/eval/docs/REPRESENTATIVE_CASE_A009.md)
5. [memory/INIT_MEMORY.md](/d:/Pyproject/MiroThinker/memory/INIT_MEMORY.md)

## 当前定位

这个仓库目前最成熟的部分不是训练框架展示，而是：

- 一个可运行的参数级搜索主流程
- 一套可复现的四模型双标准评测体系
- 一个带持久记忆和语义检索回路的搜索规划系统

如果对外发布，建议以 `apps/miroflow-agent/` 和 `apps/miroflow-agent/eval/` 作为核心展示面，其余目录按“依赖层 / 参考实现 / 辅助工具”处理。
