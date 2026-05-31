# AutoMerger — 父子层级自动合并节点

## 概述

`AutoMerger` 实现 **Auto-Merging Retrieval** 策略：当多个子块（叶子层 L3）命中同一父块（L2/L1）时，将其替换为父块，扩大上下文范围，提升回答完整性。

**路径**：`nodes/retrieval/AutoMerger.py`

## 核心原理

检索召回的是细粒度子块（text_chunk、table_chunk），但回答时需要更完整的上下文。当同一父 section 下有 ≥ N 个子块被召回时，用父 section chunk 替换这些子块。

```
召回 hits: [L3_chunk_A, L3_chunk_B, L3_chunk_C, L3_chunk_D, ...]

如果 L3_chunk_A 和 L3_chunk_B 属于同一父块 L2_parent_X:
    满足 threshold >= 2 → 替换为 L2_parent_X

如果 L2_parent_X 和其他 L2 块又属于同一 L1_parent:
    继续向上合并
```

## 合并层级

默认配置 `merge_levels: [3, 2, 1]`，最多 2 步：

```
Step 1: L3 → L2（叶子块 → 父块）
Step 2: L2 → L1（父块 → 祖父块）
```

## 配置

`config/auto_merge_config.json`：

```json
{
  "enabled": true,
  "merge_threshold": 2,
  "max_merge_steps": 2,
  "merge_levels": [3, 2, 1]
}
```

| 参数 | 说明 |
| --- | --- |
| `enabled` | 是否启用 |
| `merge_threshold` | 子块命中数 ≥ 此值触发合并 |
| `max_merge_steps` | 最大合并步数 |
| `merge_levels` | 合并层级顺序 |

## 核心函数

```python
auto_merge_hits(hits, top_k, merge_threshold, max_steps, merge_levels) → (merged_hits, meta)
integrate_auto_merge_to_retrieval(retrieval_result, ...) → dict
```

## 注意

- 当前 `ChromaRetriever` 的检索流程中并未默认启用 AutoMerger；它作为可选后处理节点存在
- AutoMerger 需要 chunk 中有 `parent_chunk_id` 和 `chunk_level` 元数据——当前 `HybridMarkdownChunk` 的 metadata 中有 `parent_id`（通过 `parent_context.parent_section_id`），但未使用 `parent_chunk_id` 和 `chunk_level` 字段名
- 如需启用，需调整 metadata 字段映射

## CLI 使用

```powershell
.\.venv\Scripts\python.exe nodes\retrieval\AutoMerger.py retrieval_result.json --threshold 2 --output merged.json
```
