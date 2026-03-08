# Project Structure (VisionTree / MiroSearch)

## 1) Production Entry
- `api_server.py`: 主 API 服务入口。对外接口包括 `/run`、`/v1/search`、`/memory/search`。
- `DEPLOY_LINUX.md`: Linux 部署说明。
- `deploy/linux/mirosearch-api.service`: systemd 示例模板。

## 2) Development Entry
- `main.py`: CLI 单任务入口，支持通过 Hydra 参数传入任务。
- `run_network_search.py`: 数据集批量执行入口。
- `mirosearch_server.py`: 独立记忆检索服务入口，仅在需要单独暴露 memory 检索时使用。

## 3) Core Execution Path
- `src/core/pipeline.py`: 任务级 pipeline，负责装配组件并执行主流程。
- `src/core/orchestrator.py`: 多轮调度大脑，负责 LLM 调用、工具循环、回滚与收敛。
- `src/core/tool_executor.py`: 工具调用参数修复、结果后处理与重复查询治理。
- `src/core/answer_generator.py`: 最终答案生成、失败总结和收尾逻辑。

## 4) Service Modules
- `src/io/output_formatter.py`: 输出格式化与 `miro` 结构化结果构建。
- `src/memory/manager.py`: 持久记忆编排（SOUL / MEMORY / daily）。
- `src/memory/mirosearch_service.py`: 语义检索服务封装（memsearch + 本地回退）。
- `src/llm/`: 多模型客户端与工厂。
- `src/logging/task_logger.py`: 结构化任务日志。
- `src/utils/`: prompt、解析和通用工具函数。

## 5) Configuration, Tests, Evaluation
- `conf/`: Hydra 配置（`llm/`, `agent/`）。
- `tests/`: pytest 单测。
- `eval/`: 发布版评测工作区。

## 6) Persistent vs Regenerable
建议保留：
- `memory/`
- `logs/`
- `eval/materials/`
- `eval/results/`

可安全再生成：
- `__pycache__/`
- `.pytest_cache/`
- `*.stdout.log`
- `*.stderr.log`

## 7) Release Rule
- 对外部署优先使用 `api_server.py`。
- 对外集成优先使用 `POST /v1/search`，因为它返回稳定 JSON。
- 发布时保留 `src/`, `conf/`, `deploy/`, `README.md`, `API_SERVICE.md`, `DEPLOY_LINUX.md`。
