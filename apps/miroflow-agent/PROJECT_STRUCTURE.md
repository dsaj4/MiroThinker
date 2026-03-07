# Project Structure (VisionTree MiroSearch Service)

## 1) Runtime Entry
- `main.py`: CLI 单任务入口（Hydra 配置 + 主流程执行）。
- `api_server.py`: FastAPI 服务入口（`/run`）。
- `run_network_search.py`: 数据集批量执行入口。
- `mirosearch_server.py`: 记忆语义检索服务入口（`/health`, `/index`, `/search`）。

## 2) Core Execution Path
- `src/core/pipeline.py`: 组件装配与任务级 pipeline。
- `src/core/orchestrator.py`: 多轮调度大脑（LLM 调用、工具循环、回滚与收敛）。
- `src/core/tool_executor.py`: 工具调用参数修复、结果后处理、重复查询处理。
- `src/core/answer_generator.py`: 最终答案生成、重试、失败经验总结。

## 3) Cross-Cutting Modules
- `src/llm/*`: 多模型客户端与工厂。
- `src/memory/manager.py`: 持久记忆编排（SOUL/MEMORY/daily）。
- `src/memory/mirosearch_service.py`: 语义检索服务封装（memsearch + 本地回退）。
- `src/io/*`: 输入处理与输出格式化（含 miro 模式）。
- `src/logging/task_logger.py`: 结构化任务日志。
- `src/utils/*`: prompt、解析与通用工具。

## 4) Config and Tests
- `conf/`: Hydra 配置（`llm/`, `agent/`）。
- `tests/`: pytest 单测。
- `eval/`: 发布版评测工作区：
  - `materials/`: 数据集、问题集、ground truth、外部 prompt、清洗后答案集
  - `results/quality_judge/`: 三维质量评测输出
  - `results/hitrate_judge/`: 基于 ground truth 的命中率评测输出
  - `results/overview/`: 四模型总览表
  - `docs/`: 评测方法与案例文档
  - `archive/legacy_outputs/`: 历史产物归档

## 5) Persistent Data Boundary
建议长期保留：
- `memory/`：项目记忆与灵魂迭代日志
- `logs/`：运行轨迹与调试日志
- `apps/miroflow-agent/data/`：运行数据集

可安全再生成：
- 覆盖率报告、缓存目录（`htmlcov`, `.pytest_cache`, `__pycache__`）

## 6) Release Rule of Thumb
- 先清理临时结果，再发布源代码与文档。
- 发布时优先引用 `apps/miroflow-agent/README.md` 与 `eval/docs/EVALUATION_GUIDE.md`。
- 不删除 `conf/`, `src/`, `tests/`，除非明确进行重构迁移。
