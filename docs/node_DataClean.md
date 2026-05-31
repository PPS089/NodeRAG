# DataClean — Markdown 清洗节点

## 概述

`DataClean` 是 NodeRAG 知识库入库链路的**第二个节点**，负责清洗 MinerU 解析产生的 Markdown 文本，去除 OCR/PDF 噪声，规范化格式，为后续分片做好准备。

**路径**：`nodes/documents/DataClean.py`

## 在链路中的位置

```
MinerUStandardReader → DataClean → HybridMarkdownChunk → ChromaBailianEmbedding
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 不可见字符清理 | 删除零宽字符、控制字符、全角空格归一化 |
| 目录删除 | 识别并删除 `# 目录` / `Table of Contents` / TOC 区块 |
| YAML front matter 删除 | 删除 `---` 包裹的 YAML 元数据 |
| HTML 标签清理 | 删除 `<div>`, `<span>`, `<font>`, `<center>`, `<p>`, `<br>` 等非表格标签 |
| HTML 表格保护 | `<table>...</table>` 先保护再恢复，保留表格结构 |
| 代码块保护 | ` ``` ` 和 `~~~` 代码块先保护再恢复 |
| 页码/页脚噪声删除 | 识别并删除 `第X页`、`Page X`、`- 123 -` 等页码形态 |
| 空表格行删除 | 删除 `| | | |` 形式的空行 |
| 格式规范化 | 标题 `###标题`→`### 标题`；列表 `*`/`+`→`-`；有序列表规范化；表格分隔线规范化 |
| 去重 | 删除连续重复的相邻行 |
| 空白压缩 | 普通文本的多余空格压缩，3+ 空行压为 2 行 |

## 输入 / 输出

### 输入

从 `MinerUResult/*/full.md` 读取 MinerU 解析的原始 Markdown。

### 输出

写回同目录的 `DataCleaned.md`：

```text
MinerUResult/<文档名>_<hash>/
├── full.md              # 输入（MinerU 原始）
├── DataCleaned.md       # 输出（清洗后）
└── ...
```

## 核心函数

### `clean_markdown(text: str) → str`

主清洗函数，处理顺序：

1. 统一换行符 `\r\n` / `\r` → `\n`
2. `normalize_invisible_chars()` — 清理不可见字符
3. `protect_code_blocks()` — 保护代码块
4. `protect_html_tables()` — 保护 HTML 表格
5. `remove_toc()` — 删除目录
6. `remove_front_matter()` — 删除 YAML front matter
7. 删除 HTML 注释 `<!-- ... -->`
8. `remove_common_html_tags()` — 删除非表格标签
9. `remove_redundant_separators()` — 删除无意义分隔线
10. `remove_page_noise()` — 删除页码噪声
11. `remove_empty_table_rows()` — 删除空表格行
12. 逐行清洗（`normalize_heading`, `normalize_unordered_list`, `normalize_ordered_list`, 空格压缩, 去重）
13. `normalize_table_separator()` — 规范表格分隔行
14. `restore_html_tables()` — 恢复 HTML 表格
15. `restore_code_blocks()` — 恢复代码块

### 批量入口

```python
clean_mineru_result_dir(result_dir, output_name="DataCleaned.md") → list[Path]
clean_md_file(input_path, output_path) → Path
```

## CLI 使用

```powershell
# 批量清洗 MinerUResult 下所有 full.md，输出 DataCleaned.md
.\.venv\Scripts\python.exe nodes\documents\DataClean.py
```

## 配置阈值

无需额外配置。所有规则硬编码在模块中，针对 MinerU 输出特点调优：

| 规则 | 策略 |
| --- | --- |
| 页码识别 | 仅删除独占一行的页码形态（纯数字、`第X页`、`Page X`），不误删正文编号 |
| 目录识别 | 标题行 + 连续的列表/TOC 文本行，遇到下一个非目录标题时结束 |
| HTML 表格 | `<table>...</table>` 清洗时保护，恢复时做轻量规范化（去除换行和多余空格） |

## 上下游契约

- **上游**：`MinerUStandardReader` 输出的 `MinerUResult/*/full.md`
- **下游**：`MarkDownChunk` / `HybridMarkdownChunk` 读取 `DataCleaned.md`
- 输出为干净的 Markdown，保留所有有意义的结构（标题、列表、表格、代码块、图片）
