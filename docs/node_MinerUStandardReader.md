# MinerUStandardReader — PDF 解析节点

## 概述

`MinerUStandardReader` 是 NodeRAG 知识库入库链路的**第一个节点**，负责将本地 PDF 文件通过 MinerU API 解析为 Markdown 文档。它是整个 RAG pipeline 的数据源头。

**路径**：`nodes/documents/MinerUStandardReader.py`

## 在链路中的位置

```
MinerUStandardReader → DataClean → HybridMarkdownChunk → ChromaBailianEmbedding
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| PDF 解析 | 调用 MinerU API (`/api/v4/file-urls/batch`) 进行精准解析 |
| 内容缓存 | 基于文件 SHA256 + 解析参数生成缓存 key，避免重复上传解析 |
| 批量处理 | 支持 `read_data_dir()` 批量处理 `data/` 目录下所有 PDF |
| 结果隔离 | 每个 PDF 输出到 `MinerUResult/<文档名>_<hash>/` 独立目录 |
| 可恢复 | 缓存命中时自动恢复完整解析结果到 MinerUResult |

## 输入 / 输出

### 输入

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `file_path` | `str` | (必传) | 本地 PDF 文件路径 |
| `model_version` | `str` | `vlm` | MinerU 模型版本 |
| `language` | `str` | `ch` | 文档语言 |
| `is_ocr` | `bool` | `False` | 是否强制 OCR |
| `enable_table` | `bool` | `True` | 是否启用表格识别 |
| `enable_formula` | `bool` | `True` | 是否启用公式识别 |
| `page_ranges` | `str \| None` | `None` | 指定解析页码范围 |
| `force_reparse` | `bool` | `False` | 忽略缓存，强制重新解析 |
| `confirm_reparse` | `bool` | `False` | 命中缓存时询问用户是否重新解析 |

### 输出（写入文件系统）

```text
MinerUResult/<文档名>_<hash>/
├── full.md              # 解析后的完整 Markdown
├── mineru_meta.json     # 解析元数据（task_id, file_hash, 解析参数等）
├── mineru_result.zip    # MinerU 返回的原始 zip
└── images/              # 文档中提取的图片
```

缓存目录 `.mineru_cache/<cache_key>/` 保存相同的完整结果，用于后续命中恢复。

## 核心类

### `MinerUStandardReader`

```
class MinerUStandardReader:
    __init__(token, timeout, interval, cache_dir, result_dir)
    read_file(file_path, ...)  → str          # 单文件解析
    read_data_dir(data_dir, ...) → dict[str,str] # 批量解析
```

### 内部方法链

```
read_file()
  ├── _hash_file()            # 计算文件 SHA256
  ├── _make_cache_key()       # 生成缓存 key
  ├── _create_file_task()     # 创建 MinerU 任务，获取上传 URL
  ├── _upload_file()          # PUT 上传文件到签名 URL
  ├── _poll_zip_url()         # 轮询等待解析完成
  ├── _download_full_md()     # 下载 zip，读取 full.md
  └── _write_result_artifacts() # 同步到 MinerUResult
```

## 环境变量依赖

| 变量 | 说明 |
| --- | --- |
| `MINERU_API_TOKEN` | MinerU API 鉴权 Token（必填） |

## CLI 使用

```powershell
# 批量解析 data/ 下所有 PDF，需要输入 "确认"
.\.venv\Scripts\python.exe nodes\documents\MinerUStandardReader.py
```

## 缓存策略

1. 计算文件内容 SHA256（分块读取，支持大文件）
2. 将 `file_hash` + 解析参数（不含 `file_name`）组成 `cache_identity`
3. 对 `cache_identity` 做 SHA256 得到 `cache_key`
4. 如果 `.mineru_cache/<cache_key>.md` 已存在 → 直接返回缓存
5. 否则上传解析 → 写入缓存

**注意**：同名不同内容的文件会被视为不同缓存，避免文件名碰撞。

## 上下游契约

- **下游消费者**：`DataClean.py` 读取 `MinerUResult/*/full.md`
- **输出的 full.md** 是经过 MinerU 解析的原始 OCR Markdown，可能包含噪声，需要后续清洗
