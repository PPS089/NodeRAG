from __future__ import annotations

import html as html_lib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from utils.FindProjectRoot import find_project_root as fr


TABLE_REF_TEMPLATE = "TABLE_REF:{chunk_id}"
IMAGE_REF_TEMPLATE = "IMAGE_REF:{chunk_id}"

MAX_TEXT_CHUNK_CHARS = 8000
MAX_TABLE_ROWS_PER_CHUNK = 30
MAX_CELL_TEXT_CHARS = 500
MAX_IMAGE_LINE_LENGTH = 200_000
MAX_TABLE_LINE_LENGTH = 200_000

HTML_TABLE_START = "<table"
HTML_TABLE_END = "</table>"


class OrderCounter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def truncate_text(text: Optional[str], max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[TRUNCATED {len(text) - max_chars} chars]"


def normalize_cell_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = html_lib.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return truncate_text(value.strip(), MAX_CELL_TEXT_CHARS)


def infer_level_from_numbered_title(title: str, markdown_level: int) -> int:
    """
    MinerU 有时会把所有标题都输出成 #，这里根据编号补偿层级：
    1 标题 / 1. 标题 -> level 1
    1.1 标题 -> level 2
    1.1.1 标题 -> level 3
    """
    s = title.strip()
    m = re.match(r"^(\d+(?:\.\d+)*)(?:\.|\s)+", s)
    if not m:
        return markdown_level
    return min(m.group(1).count(".") + 1, 6)


def parse_heading(line: str) -> Optional[Tuple[int, str]]:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None

    markdown_level = 0
    for ch in stripped:
        if ch == "#":
            markdown_level += 1
        else:
            break

    if markdown_level < 1 or markdown_level > 6:
        return None
    if len(stripped) <= markdown_level or stripped[markdown_level] != " ":
        return None

    title = stripped[markdown_level:].strip()
    if not title:
        return None

    return infer_level_from_numbered_title(title, markdown_level), title


def is_heading(line: str) -> bool:
    return parse_heading(line) is not None


def is_fence_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def is_toc_heading(title: str) -> bool:
    return title.strip() in {"目录", "Table of Contents", "Contents"}


def looks_like_toc_heading(title: str) -> bool:
    s = title.strip()
    return bool(re.match(r"^\d+(?:\.\d+)*\s+.+\s+\d+$", s))


def looks_like_toc_text_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\s+.+(?:\s+\.|\s+\d|\s*$)", s))


def current_title_path(section_stack: List[Dict[str, Any]]) -> List[str]:
    return [s["title"] for s in section_stack]


def current_parent_id(section_stack: List[Dict[str, Any]]) -> Optional[str]:
    return section_stack[-1]["id"] if section_stack else None


def make_chunk(
    chunk_type: str,
    content: str,
    source_path: str,
    line_start: int,
    line_end: int,
    title_path: Sequence[str],
    parent_id: Optional[str],
    order_counter: OrderCounter,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chunk: Dict[str, Any] = {
        "id": new_id(chunk_type),
        "type": chunk_type,
        "content": content,
        "title_path": list(title_path),
        "source_path": source_path,
        "line_range": [line_start, line_end],
        "parent_id": parent_id,
        "child_ids": [],
        "order_index": order_counter.next(),
    }
    if extra:
        chunk.update(extra)
    return chunk


def add_chunk(
    chunks: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    chunk: Dict[str, Any],
) -> None:
    chunks.append(chunk)
    chunks_by_id[chunk["id"]] = chunk

    parent_id = chunk.get("parent_id")
    if parent_id and parent_id in chunks_by_id:
        chunks_by_id[parent_id].setdefault("child_ids", []).append(chunk["id"])


def close_sections_until(
    section_stack: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    new_level: int,
    end_line: int,
) -> None:
    while section_stack and section_stack[-1]["level"] >= new_level:
        section = section_stack.pop()
        chunk = chunks_by_id[section["id"]]
        chunk["section_line_range"][1] = max(chunk["section_line_range"][1], end_line)


def close_all_sections(
    section_stack: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    end_line: int,
) -> None:
    while section_stack:
        section = section_stack.pop()
        chunk = chunks_by_id[section["id"]]
        chunk["section_line_range"][1] = max(chunk["section_line_range"][1], end_line)


def find_markdown_images(line: str) -> List[Dict[str, Any]]:
    """支持一行内多个 ![alt](path) 图片；不用大正则，避免超长行卡住。"""
    if len(line) > MAX_IMAGE_LINE_LENGTH:
        return []
    if "![" not in line or "](" not in line:
        return []

    result: List[Dict[str, Any]] = []
    pos = 0
    while True:
        start = line.find("![", pos)
        if start == -1:
            break

        alt_start = start + 2
        alt_end = line.find("]", alt_start)
        if alt_end == -1:
            break
        if alt_end + 1 >= len(line) or line[alt_end + 1] != "(":
            pos = alt_end + 1
            continue

        path_start = alt_end + 2
        path_end = line.find(")", path_start)
        if path_end == -1:
            break

        alt = line[alt_start:alt_end]
        path = line[path_start:path_end]
        original = line[start : path_end + 1]

        if len(alt) <= 500 and 0 < len(path) <= 2000:
            result.append(
                {
                    "start": start,
                    "end": path_end + 1,
                    "alt_text": alt.strip(),
                    "image_path": path.strip(),
                    "original": original,
                }
            )

        pos = path_end + 1

    return result


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) > MAX_TABLE_LINE_LENGTH:
        return False
    if "|" not in stripped or "-" not in stripped:
        return False

    cells = stripped.strip("|").split("|")
    if not cells:
        return False

    for cell in cells:
        cell = cell.strip()
        if len(cell) < 3:
            return False
        if cell.startswith(":"):
            cell = cell[1:]
        if cell.endswith(":"):
            cell = cell[:-1]
        if not cell or any(ch != "-" for ch in cell):
            return False
    return True


def is_markdown_table_start(lines: Sequence[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    line = lines[idx]
    next_line = lines[idx + 1]
    if len(line) > MAX_TABLE_LINE_LENGTH:
        return False
    if "|" not in line:
        return False
    return is_table_separator(next_line)


def collect_markdown_table(lines: Sequence[str], start_idx: int) -> Tuple[List[str], int]:
    table_lines: List[str] = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if is_heading(line):
            break
        if not stripped:
            break
        if "|" not in stripped:
            break
        if len(line) > MAX_TABLE_LINE_LENGTH:
            break
        table_lines.append(line)
        idx += 1
    return table_lines, idx


def split_table_rows(table_lines: Sequence[str]) -> Tuple[List[str], List[str]]:
    if len(table_lines) <= 2:
        return list(table_lines), []
    return list(table_lines[:2]), list(table_lines[2:])


def split_large_markdown_table_with_ranges(
    table_lines: Sequence[str],
    table_start_line: int,
    max_rows: int = MAX_TABLE_ROWS_PER_CHUNK,
) -> List[Dict[str, Any]]:
    """
    大 Markdown 表格按数据行切片，每片都重复原表头。
    line_range 对第一片包含表头和数据行；后续片的 line_range 指向数据行范围，
    同时额外给出 table_header_line_range / table_source_line_range 避免歧义。
    """
    header_lines, data_lines = split_table_rows(table_lines)
    table_end_line = table_start_line + len(table_lines) - 1

    if not data_lines:
        return [
            {
                "part_lines": list(table_lines),
                "line_range": [table_start_line, table_end_line],
                "table_header_line_range": [table_start_line, min(table_start_line + 1, table_end_line)],
                "table_data_line_range": None,
                "table_source_line_range": [table_start_line, table_end_line],
            }
        ]

    parts: List[Dict[str, Any]] = []
    total_parts = (len(data_lines) + max_rows - 1) // max_rows
    for part_index, start in enumerate(range(0, len(data_lines), max_rows), start=1):
        data_part = data_lines[start : start + max_rows]
        data_start_line = table_start_line + 2 + start
        data_end_line = data_start_line + len(data_part) - 1
        line_range = [table_start_line, data_end_line] if part_index == 1 else [data_start_line, data_end_line]
        parts.append(
            {
                "part_lines": header_lines + data_part,
                "line_range": line_range,
                "table_header_line_range": [table_start_line, table_start_line + 1],
                "table_data_line_range": [data_start_line, data_end_line],
                "table_source_line_range": [table_start_line, table_end_line],
                "table_part_index": part_index,
                "table_part_total": total_parts,
            }
        )
    return parts


def parse_markdown_table_cells(line: str) -> List[str]:
    return [truncate_text(cell.strip(), MAX_CELL_TEXT_CHARS) for cell in line.strip().strip("|").split("|")]


def markdown_table_to_text(table_lines: Sequence[str], row_offset: int = 0) -> str:
    if len(table_lines) < 2:
        return truncate_text("".join(table_lines), MAX_TEXT_CHUNK_CHARS)

    rows: List[List[str]] = []
    for line in table_lines:
        if is_table_separator(line):
            continue
        rows.append(parse_markdown_table_cells(line))

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]
    output: List[str] = []
    for row_idx, row in enumerate(data_rows, start=row_offset + 1):
        pairs = []
        for i, value in enumerate(row):
            header = headers[i] if i < len(headers) else f"column_{i + 1}"
            pairs.append(f"{header}: {value}")
        output.append(f"Row {row_idx}: " + "; ".join(pairs))

    return truncate_text("\n".join(output), MAX_TEXT_CHUNK_CHARS)


def describe_markdown_table(
    table_lines: Sequence[str],
    part_index: int = 1,
    part_total: int = 1,
    source_data_rows: Optional[int] = None,
) -> str:
    if not table_lines:
        return "Markdown table."

    headers = parse_markdown_table_cells(table_lines[0])
    part_data_rows = max(len(table_lines) - 2, 0)
    total_data_rows = source_data_rows if source_data_rows is not None else part_data_rows
    col_count = len(headers)
    shown_headers = headers[:20]
    more = "" if len(headers) <= 20 else f", ... and {len(headers) - 20} more columns"

    part_text = ""
    if part_total > 1:
        part_text = f" This is part {part_index} of {part_total}, with {part_data_rows} data rows in this part."

    return (
        f"Markdown table with {total_data_rows} data rows and {col_count} columns. "
        f"Columns: {', '.join(shown_headers)}{more}.{part_text}"
    )


def is_html_table_start(line: str) -> bool:
    return HTML_TABLE_START in line.lower()


def collect_html_table(lines: Sequence[str], start_idx: int) -> Tuple[List[str], int]:
    table_lines: List[str] = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        table_lines.append(line)
        if HTML_TABLE_END in line.lower():
            idx += 1
            break
        idx += 1
    return table_lines, idx


def parse_html_rows(html_text: str) -> List[List[str]]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S)
    parsed_rows: List[List[str]] = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)
        clean_cells = [normalize_cell_text(cell) for cell in cells]
        if clean_cells:
            parsed_rows.append(clean_cells)
    return parsed_rows


def is_key_value_table(rows: Sequence[Sequence[str]]) -> bool:
    if not rows:
        return False
    if not all(len(row) >= 2 and len(row) % 2 == 0 for row in rows):
        return False
    max_cols = max(len(row) for row in rows)
    return len(rows) <= 8 and max_cols <= 6


def html_table_to_text(html_text: str, row_offset: int = 0) -> str:
    rows = parse_html_rows(html_text)
    if not rows:
        return truncate_text(html_text, MAX_TEXT_CHUNK_CHARS)

    if is_key_value_table(rows):
        output = []
        for row_idx, row in enumerate(rows, start=row_offset + 1):
            pairs = []
            for i in range(0, len(row), 2):
                key = row[i]
                value = row[i + 1] if i + 1 < len(row) else ""
                pairs.append(f"{key}: {value}")
            output.append(f"Row {row_idx}: " + "; ".join(pairs))
        return truncate_text("\n".join(output), MAX_TEXT_CHUNK_CHARS)

    headers = rows[0]
    data_rows = rows[1:]
    output = []
    for row_idx, row in enumerate(data_rows, start=row_offset + 1):
        pairs = []
        for i, value in enumerate(row):
            header = headers[i] if i < len(headers) else f"column_{i + 1}"
            pairs.append(f"{header}: {value}")
        output.append(f"Row {row_idx}: " + "; ".join(pairs))
    return truncate_text("\n".join(output), MAX_TEXT_CHUNK_CHARS)


def describe_html_table(html_text: str) -> str:
    rows = parse_html_rows(html_text)
    if not rows:
        return "HTML table."

    row_count = len(rows)
    col_count = max(len(row) for row in rows)
    if is_key_value_table(rows):
        keys = []
        for row in rows:
            for i in range(0, len(row), 2):
                keys.append(row[i])
        shown_keys = keys[:20]
        more = "" if len(keys) <= 20 else f", ... and {len(keys) - 20} more fields"
        return f"HTML key-value table with {row_count} rows and {col_count} columns. Fields: {', '.join(shown_keys)}{more}."

    headers = rows[0]
    data_rows = max(len(rows) - 1, 0)
    shown_headers = headers[:20]
    more = "" if len(headers) <= 20 else f", ... and {len(headers) - 20} more columns"
    return f"HTML table with {data_rows} data rows and {col_count} columns. Columns: {', '.join(shown_headers)}{more}."


def flush_text_buffer(
    chunks: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    text_buffer: List[Tuple[int, str]],
    text_refs: List[Dict[str, Any]],
    source_path: str,
    section_stack: List[Dict[str, Any]],
    order_counter: OrderCounter,
) -> None:
    if not text_buffer:
        return

    parent_id = current_parent_id(section_stack)
    title_path = current_title_path(section_stack)
    pending_refs = list(text_refs)

    current_lines: List[str] = []
    current_start_line: Optional[int] = None
    current_end_line: Optional[int] = None
    current_size = 0

    def emit() -> None:
        nonlocal current_lines, current_start_line, current_end_line, current_size
        if not current_lines or current_start_line is None or current_end_line is None:
            return

        content = "".join(current_lines).strip("\n")
        if content.strip():
            refs_for_chunk = [ref for ref in pending_refs if ref["ref"] in content]
            chunk = make_chunk(
                chunk_type="text_chunk",
                content=content,
                source_path=source_path,
                line_start=current_start_line,
                line_end=current_end_line,
                title_path=title_path,
                parent_id=parent_id,
                order_counter=order_counter,
                extra={
                    "refs": refs_for_chunk,
                    "ref_target_ids": [ref["target_chunk_id"] for ref in refs_for_chunk],
                },
            )
            add_chunk(chunks, chunks_by_id, chunk)

            for ref in refs_for_chunk:
                target = chunks_by_id.get(ref["target_chunk_id"])
                if target is not None:
                    target.setdefault("referenced_by_text_chunk_ids", []).append(chunk["id"])

        current_lines = []
        current_start_line = None
        current_end_line = None
        current_size = 0

    for line_no, line in text_buffer:
        remaining = line
        if remaining == "":
            continue

        while remaining:
            if current_start_line is None:
                current_start_line = line_no

            available = MAX_TEXT_CHUNK_CHARS - current_size
            if available <= 0:
                emit()
                continue

            part = remaining[:available]
            remaining = remaining[available:]
            current_lines.append(part)
            current_end_line = line_no
            current_size += len(part)

            if current_size >= MAX_TEXT_CHUNK_CHARS:
                emit()

    emit()
    text_buffer.clear()
    text_refs.clear()


def add_text_line(
    text_buffer: List[Tuple[int, str]],
    line_no: int,
    line: str,
    current_text_buffer_size: int,
) -> int:
    text_buffer.append((line_no, line))
    return current_text_buffer_size + len(line)


def add_ref_text_chunk(
    chunks: List[Dict[str, Any]],
    chunks_by_id: Dict[str, Dict[str, Any]],
    text_buffer: List[Tuple[int, str]],
    text_refs: List[Dict[str, Any]],
    source_path: str,
    section_stack: List[Dict[str, Any]],
    order_counter: OrderCounter,
    refs: List[Dict[str, Any]],
    line_start: int,
    line_end: Optional[int] = None,
) -> None:
    """
    在正文位置写入 REF 占位符并立即 flush 成 text_chunk。
    对表格这类多行对象，text_chunk 的 line_range 覆盖原始表格范围；
    对图片这类单行对象，line_start == line_end。
    """
    if line_end is None:
        line_end = line_start

    text_refs.extend(refs)
    text_buffer.append((line_start, "\n".join(ref["ref"] for ref in refs) + "\n"))
    if line_end != line_start:
        # 只用于扩展 line_range，strip 后不会影响正文占位符内容。
        text_buffer.append((line_end, "\n"))

    flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)


def split_markdown(input_md: str | Path, skip_toc: bool = True) -> List[Dict[str, Any]]:
    source_path = str(input_md)
    text = Path(input_md).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)

    chunks: List[Dict[str, Any]] = []
    chunks_by_id: Dict[str, Dict[str, Any]] = {}
    section_stack: List[Dict[str, Any]] = []
    text_buffer: List[Tuple[int, str]] = []
    text_refs: List[Dict[str, Any]] = []
    text_buffer_size = 0
    order_counter = OrderCounter()

    idx = 0
    total_lines = len(lines)
    in_toc = False
    in_code_block = False

    while idx < total_lines:
        line = lines[idx]
        line_no = idx + 1

        if is_fence_line(line):
            text_buffer_size = add_text_line(text_buffer, line_no, line, text_buffer_size)
            in_code_block = not in_code_block
            idx += 1
            continue

        if in_code_block:
            text_buffer_size = add_text_line(text_buffer, line_no, line, text_buffer_size)
            if text_buffer_size >= MAX_TEXT_CHUNK_CHARS:
                flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
                text_buffer_size = 0
            idx += 1
            continue

        heading = parse_heading(line)
        if heading:
            level, title = heading

            if skip_toc and is_toc_heading(title):
                flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
                text_buffer_size = 0
                in_toc = True
                idx += 1
                continue

            if skip_toc and in_toc and looks_like_toc_heading(title):
                idx += 1
                continue

            if skip_toc and in_toc and not looks_like_toc_heading(title):
                in_toc = False

            flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
            text_buffer_size = 0
            close_sections_until(section_stack, chunks_by_id, new_level=level, end_line=line_no - 1)

            parent_id = current_parent_id(section_stack)
            section_chunk = make_chunk(
                chunk_type="section_chunk",
                content=line.strip(),
                source_path=source_path,
                line_start=line_no,
                line_end=line_no,
                title_path=current_title_path(section_stack) + [title],
                parent_id=parent_id,
                order_counter=order_counter,
                extra={
                    "level": level,
                    "title": title,
                    "heading_line_range": [line_no, line_no],
                    "section_line_range": [line_no, line_no],
                },
            )
            add_chunk(chunks, chunks_by_id, section_chunk)
            section_stack.append({"id": section_chunk["id"], "level": level, "title": title})

            idx += 1
            continue

        if skip_toc and in_toc:
            if looks_like_toc_text_line(line):
                idx += 1
                continue
            in_toc = False

        if is_html_table_start(line):
            flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
            text_buffer_size = 0

            table_lines, next_idx = collect_html_table(lines, idx)
            table_start_line = idx + 1
            table_end_line = next_idx
            raw_table = "".join(table_lines).strip()

            table_chunk = make_chunk(
                chunk_type="table_chunk",
                content=raw_table,
                source_path=source_path,
                line_start=table_start_line,
                line_end=table_end_line,
                title_path=current_title_path(section_stack),
                parent_id=current_parent_id(section_stack),
                order_counter=order_counter,
                extra={
                    "raw_table": raw_table,
                    "raw_markdown_table": None,
                    "raw_html_table": raw_table,
                    "table_text": html_table_to_text(raw_table),
                    "table_description": describe_html_table(raw_table),
                    "table_format": "html",
                    "table_part_index": 1,
                    "table_part_total": 1,
                    "table_source_line_range": [table_start_line, table_end_line],
                },
            )
            add_chunk(chunks, chunks_by_id, table_chunk)

            ref = TABLE_REF_TEMPLATE.format(chunk_id=table_chunk["id"])
            add_ref_text_chunk(
                chunks,
                chunks_by_id,
                text_buffer,
                text_refs,
                source_path,
                section_stack,
                order_counter,
                [
                    {
                        "ref": ref,
                        "target_chunk_id": table_chunk["id"],
                        "target_type": "table_chunk",
                        "source_line_range": [table_start_line, table_end_line],
                    }
                ],
                table_start_line,
                table_end_line,
            )

            idx = next_idx
            continue

        if is_markdown_table_start(lines, idx):
            flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
            text_buffer_size = 0

            table_lines, next_idx = collect_markdown_table(lines, idx)
            table_start_line = idx + 1
            table_end_line = next_idx
            source_data_rows = max(len(table_lines) - 2, 0)
            table_parts = split_large_markdown_table_with_ranges(table_lines, table_start_line)
            refs: List[Dict[str, Any]] = []

            for part in table_parts:
                part_lines = part["part_lines"]
                part_index = part.get("table_part_index", 1)
                part_total = part.get("table_part_total", len(table_parts))
                row_offset = 0
                if part.get("table_data_line_range"):
                    row_offset = max(part["table_data_line_range"][0] - table_start_line - 2, 0)

                raw_table = "".join(part_lines).strip("\n")
                line_start, line_end = part["line_range"]
                table_chunk = make_chunk(
                    chunk_type="table_chunk",
                    content=raw_table,
                    source_path=source_path,
                    line_start=line_start,
                    line_end=line_end,
                    title_path=current_title_path(section_stack),
                    parent_id=current_parent_id(section_stack),
                    order_counter=order_counter,
                    extra={
                        "raw_table": raw_table,
                        "raw_markdown_table": raw_table,
                        "raw_html_table": None,
                        "table_text": markdown_table_to_text(part_lines, row_offset=row_offset),
                        "table_description": describe_markdown_table(
                            part_lines,
                            part_index=part_index,
                            part_total=part_total,
                            source_data_rows=source_data_rows,
                        ),
                        "table_format": "markdown",
                        "table_part_index": part_index,
                        "table_part_total": part_total,
                        "table_header_line_range": part.get("table_header_line_range"),
                        "table_data_line_range": part.get("table_data_line_range"),
                        "table_source_line_range": part.get("table_source_line_range", [table_start_line, table_end_line]),
                    },
                )
                add_chunk(chunks, chunks_by_id, table_chunk)

                ref = TABLE_REF_TEMPLATE.format(chunk_id=table_chunk["id"])
                refs.append(
                    {
                        "ref": ref,
                        "target_chunk_id": table_chunk["id"],
                        "target_type": "table_chunk",
                        "source_line_range": [table_start_line, table_end_line],
                    }
                )

            add_ref_text_chunk(
                chunks,
                chunks_by_id,
                text_buffer,
                text_refs,
                source_path,
                section_stack,
                order_counter,
                refs,
                table_start_line,
                table_end_line,
            )

            idx = next_idx
            continue

        images = find_markdown_images(line)
        if images:
            flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
            text_buffer_size = 0

            replaced_parts: List[str] = []
            refs: List[Dict[str, Any]] = []
            last_pos = 0
            for image in images:
                replaced_parts.append(line[last_pos : image["start"]])

                image_chunk = make_chunk(
                    chunk_type="image_chunk",
                    content=image["original"],
                    source_path=source_path,
                    line_start=line_no,
                    line_end=line_no,
                    title_path=current_title_path(section_stack),
                    parent_id=current_parent_id(section_stack),
                    order_counter=order_counter,
                    extra={
                        "alt_text": image["alt_text"],
                        "image_path": image["image_path"],
                        "original_markdown": image["original"],
                    },
                )
                add_chunk(chunks, chunks_by_id, image_chunk)

                image_ref = IMAGE_REF_TEMPLATE.format(chunk_id=image_chunk["id"])
                replaced_parts.append(image_ref)
                refs.append(
                    {
                        "ref": image_ref,
                        "target_chunk_id": image_chunk["id"],
                        "target_type": "image_chunk",
                        "source_line_range": [line_no, line_no],
                    }
                )
                last_pos = image["end"]

            replaced_parts.append(line[last_pos:])
            replaced_line = "".join(replaced_parts)
            text_refs.extend(refs)
            text_buffer.append((line_no, replaced_line))
            flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
            text_buffer_size = 0

            idx += 1
            continue

        text_buffer_size = add_text_line(text_buffer, line_no, line, text_buffer_size)
        if text_buffer_size >= MAX_TEXT_CHUNK_CHARS:
            flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
            text_buffer_size = 0

        idx += 1

    flush_text_buffer(chunks, chunks_by_id, text_buffer, text_refs, source_path, section_stack, order_counter)
    close_all_sections(section_stack, chunks_by_id, end_line=total_lines)
    return chunks


def main(input_md: str | Path = "full.md") -> List[Dict[str, Any]]:
    """
    input_md 是 Markdown 输入文件。
    返回 chunks，并在同目录写出 <input_md>.chunks.json。
    """
    input_md = Path(input_md)
    chunks = split_markdown(input_md, skip_toc=True)

    output_path = input_md.with_suffix(".chunks.json")
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(chunks, ensure_ascii=False, indent=2))
    return chunks


if __name__ == "__main__":
    path = fr()

    input_md = (
            path
            / ".mineru_cache"
            / "5ac5a6c3b1da4f87b1045865a51cbcf4e73faad43f5873e6676946902f981f64"
            / "DataCleaned.md"
    )
    chunks = main(input_md)
