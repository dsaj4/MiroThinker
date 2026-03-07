# 记忆系统初始化指南（VisionTree / MiroSearch）

## 1. 目的
`memory/` 目录用于保存本项目的长期记忆与检索策略演进：
- `SOUL.md`: 规划师身份与策略迭代日志
- `MEMORY.md`: 项目长期偏好、约束和规则
- `daily/`: 按天落盘的任务记忆
- `external/`: 外部附件留存
- `index/`: 语义检索索引缓存
- `soul_versions/`: `SOUL.md` 历史版本备份

## 2. 初始化状态（建议）
- `SOUL.md`：保留当前身份设定，不建议频繁人工覆盖。
- `MEMORY.md`：保留项目级偏好条目。
- `daily/`, `external/`, `index/`, `soul_versions/`：可按需清空后再启动。

## 3. 最小配置步骤
1. 在 `apps/miroflow-agent/conf/config.yaml` 确认：
   - `memory.enabled: true`
   - `memory.semantic_recall: true`
   - `memory.root_dir: ../../memory`
2. 可选配置 Milvus（语义检索增强）：
   - 设置 `MILVUS_URI`
   - 如需鉴权，设置 `MILVUS_TOKEN`
3. 运行一次主流程任务完成初始化：
   - `uv run python apps/miroflow-agent/main.py llm=qwen-3 agent=mirothinker_v1.5_keep5_max200`

## 4. 首次运行后会发生什么
- 生成或更新 `daily/YYYY-MM-DD.md`。
- `SOUL.md` 追加新的策略强化条目。
- 语义索引完成 warmup/reindex（memsearch 或 local_tfidf 回退）。

## 5. 健康检查
检查 `logs/debug/task_*.json` 是否包含：
- `Mirosearch | Init`
- `Memory | Warmup`
- `Memory | Prompt Injection`
- `Memory | Persist`

若后端显示 `local_tfidf`，说明仍可正常运行；
如需切换到 memsearch，请提供可用 Milvus 端点。

## 6. 注意事项
- `SOUL.md` 与 `MEMORY.md` 属于运行时核心资产，发布时建议保留并做版本备份。
- 若出现编码异常，优先修复文件编码为 UTF-8，不要直接清空历史记忆。
