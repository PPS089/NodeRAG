import re
from pathlib import Path
from utils.FindProjectRoot import find_project_root as fr


CODE_BLOCK_RE = re.compile(r"(```.*?```|~~~.*?~~~)", flags=re.S)
HTML_TABLE_RE = re.compile(r"(<table\b.*?</table>)", flags=re.S | re.I)
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_markdown(text: str) -> str:
    """
    清洗 Markdown 文档内容

    主要用于 RAG 分片前的数据清洗：
    - 统一换行符
    - 删除目录
    - 删除 HTML 注释
    - 清理部分 HTML 标签
    - 规范标题、列表、表格
    - 压缩多余空行和空格
    - 保留代码块内容
    - 不处理 Markdown 链接
    """

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 清理 OCR/PDF 常见不可见字符
    text = normalize_invisible_chars(text)

    # 保护代码块，避免后续规则误处理代码内容
    text, code_blocks = protect_code_blocks(text)

    # 保护 HTML 表格，避免清洗标签时破坏下游表格分片
    text, html_tables = protect_html_tables(text)

    # 删除目录
    text = remove_toc(text)

    # 删除 YAML front matter
    text = remove_front_matter(text)

    # 删除 HTML 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

    # 删除常见 HTML 标签，但保留标签中的文字
    text = remove_common_html_tags(text)

    # 删除无意义分隔线
    text = remove_redundant_separators(text)

    # 删除保守可识别的页码/页脚噪声
    text = remove_page_noise(text)

    # 删除空表格行
    text = remove_empty_table_rows(text)

    # 按行清洗
    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines = []
    previous_line = None

    for line in lines:
        stripped = line.strip()

        # 跳过连续重复空行
        if stripped == "" and previous_line == "":
            continue

        # 跳过清洗后仍明显无意义的行
        if is_noise_line(stripped):
            continue

        # 规范标题：###标题 -> ### 标题
        stripped = normalize_heading(stripped)

        # 规范无序列表符号：* item / + item -> - item
        stripped = normalize_unordered_list(stripped)

        # 规范有序列表：1.内容 -> 1. 内容
        stripped = normalize_ordered_list(stripped)

        # 压缩普通文本中的多个空格
        if should_compress_spaces(stripped):
            stripped = re.sub(r"[ \t]{2,}", " ", stripped)

        # 删除完全重复的相邻行
        if stripped == previous_line and stripped != "":
            continue

        cleaned_lines.append(stripped)
        previous_line = stripped

    text = "\n".join(cleaned_lines)

    # 规范表格分隔行
    text = re.sub(
        r"^\|?[\s:.-]+\|[\s|:.-]+$",
        lambda m: normalize_table_separator(m.group()),
        text,
        flags=re.M
    )

    # 压缩过多空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 恢复 HTML 表格
    text = restore_html_tables(text, html_tables)

    # 恢复代码块
    text = restore_code_blocks(text, code_blocks)

    # 恢复后再做一次轻量清洗
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 文档结尾保留一个换行
    text = text.strip() + "\n"

    return text


def normalize_invisible_chars(text: str) -> str:
    """
    清理 MinerU/OCR 结果中常见的不可见字符和异常空白。
    """

    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    return text


def protect_code_blocks(text: str) -> tuple[str, list[str]]:
    """
    保护 Markdown 代码块

    避免代码块中的缩进、空格、列表符号、表格符号被误清洗。
    """

    code_blocks = []

    def replacer(match):
        placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
        code_blocks.append(match.group())
        return placeholder

    protected_text = CODE_BLOCK_RE.sub(replacer, text)

    return protected_text, code_blocks


def restore_code_blocks(text: str, code_blocks: list[str]) -> str:
    """
    恢复 Markdown 代码块
    """

    for index, code_block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{index}__", code_block)

    return text


def protect_html_tables(text: str) -> tuple[str, list[str]]:
    """
    保护 HTML 表格。

    MarkDownChunk 后续会识别 <table>，因此清洗阶段只保护结构，不转换格式。
    """

    html_tables = []

    def replacer(match):
        placeholder = f"__HTML_TABLE_{len(html_tables)}__"
        html_tables.append(match.group())
        return placeholder

    protected_text = HTML_TABLE_RE.sub(replacer, text)

    return protected_text, html_tables


def restore_html_tables(text: str, html_tables: list[str]) -> str:
    """
    恢复 HTML 表格。
    """

    for index, html_table in enumerate(html_tables):
        text = text.replace(f"__HTML_TABLE_{index}__", normalize_html_table(html_table))

    return text


def normalize_html_table(html_table: str) -> str:
    """
    对 HTML 表格做轻量规范化，保留标签结构。
    """

    html_table = html_table.replace("\n", "")
    html_table = re.sub(r">\s+<", "><", html_table)
    html_table = re.sub(r"[ \t]{2,}", " ", html_table)
    return html_table.strip()


def remove_front_matter(text: str) -> str:
    """
    删除 Markdown YAML front matter

    示例：
    ---
    title: xxx
    date: xxx
    ---
    """

    return re.sub(r"^\s*---\n.*?\n---\s*\n", "", text, flags=re.S)


def remove_common_html_tags(text: str) -> str:
    """
    删除常见 HTML 标签，但保留标签中的文字
    """

    return re.sub(
        r"</?(div|span|font|center|p|br|strong|em|b|i|u|section|article)[^>]*>",
        "",
        text,
        flags=re.I
    )


def remove_redundant_separators(text: str) -> str:
    """
    删除无意义 Markdown 分隔线

    例如：
    ---
    ***
    ___
    """

    return re.sub(r"(?m)^\s*([-*_])\s*(\1\s*){2,}$", "", text)


def remove_page_noise(text: str) -> str:
    """
    删除保守可识别的 PDF 页码/页脚噪声。

    只处理独占一行的页码形态，避免误删正文编号。
    """

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if is_page_number_line(stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_page_number_line(line: str) -> bool:
    """
    判断是否是独占一行的页码。
    """

    if re.match(r"^[-–—]?\s*\d{1,4}\s*[-–—]?$", line):
        return True

    if re.match(r"^(第\s*)?\d{1,4}\s*(页|/\s*\d{1,4}|of\s+\d{1,4})$", line, re.I):
        return True

    if re.match(r"^Page\s+\d{1,4}(\s*/\s*\d{1,4})?$", line, re.I):
        return True

    return False


def remove_empty_table_rows(text: str) -> str:
    """
    删除明显空表格行

    例如：
    | | | |
    """

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        if re.match(r"^\s*\|(\s*\|)+\s*$", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_noise_line(line: str) -> bool:
    """
    判断清洗后仍明显无意义的行。
    """

    if not line:
        return False

    if re.fullmatch(r"[·•\-\s]{3,}", line):
        return True

    if re.fullmatch(r"[.。·•]{4,}", line):
        return True

    return False


def normalize_heading(line: str) -> str:
    """
    规范 Markdown 标题

    ###标题 -> ### 标题
    """

    return re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", line)


def normalize_unordered_list(line: str) -> str:
    """
    规范无序列表

    * item -> - item
    + item -> - item
    """

    return re.sub(r"^\s*[\*\+]\s+", "- ", line)


def normalize_ordered_list(line: str) -> str:
    """
    规范有序列表

    1.内容 -> 1. 内容
    """

    return re.sub(r"^(\d+)\.([^\s])", r"\1. \2", line)


def should_compress_spaces(line: str) -> bool:
    """
    判断当前行是否适合压缩连续空格

    不压缩：
    - 表格行
    - 代码块占位符
    - 缩进行
    """

    if line.startswith("|"):
        return False

    if line.startswith("__CODE_BLOCK_"):
        return False

    if line.startswith("__HTML_TABLE_"):
        return False

    if line.startswith(" "):
        return False

    return True


def normalize_table_separator(line: str) -> str:
    """
    规范 Markdown 表格分隔线

    示例：
    | --- | :---: | ---: |
    """

    parts = [part.strip() for part in line.strip("|").split("|")]

    normalized = []

    for part in parts:
        if part.startswith(":") and part.endswith(":"):
            normalized.append(":---:")
        elif part.endswith(":"):
            normalized.append("---:")
        elif part.startswith(":"):
            normalized.append(":---")
        else:
            normalized.append("---")

    return "| " + " | ".join(normalized) + " |"


def remove_toc(text: str) -> str:
    """
    删除 Markdown 文档中的目录部分

    支持识别：
    - # 目录
    - ## 目录
    - Table of Contents
    - TOC
    """

    lines = text.split("\n")
    result = []

    in_toc = False

    for line in lines:
        stripped = line.strip()

        # 识别目录标题
        if re.match(r"^#{1,6}\s*(目录|Table of Contents|TOC)\s*$", stripped, re.I):
            in_toc = True
            continue

        if in_toc:
            # 目录通常是列表形式
            if re.match(r"^[-*+]\s+\[?.+?\]?\(?#.+\)?", stripped):
                continue

            # 有序目录
            if re.match(r"^\d+\.\s+\[?.+?\]?\(?#.+\)?", stripped):
                continue

            # 目录里的空行也跳过
            if stripped == "":
                continue

            # 遇到下一个标题，说明目录结束
            if re.match(r"^#{1,6}\s+", stripped):
                in_toc = False
                result.append(line)
                continue

            # 其他目录内容也跳过
            continue

        result.append(line)

    return "\n".join(result)


def clean_md_file(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    清洗 Markdown 文件

    参数:
        input_path: 输入 md 文件路径
        output_path: 输出 md 文件路径；如果不传，则自动生成 *_cleaned.md
    """

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"文件不存在: {input_file}")

    if input_file.suffix.lower() != ".md":
        raise ValueError("请输入 .md 格式的 Markdown 文件")

    if output_path is None:
        output_file = input_file.with_name(input_file.stem + "_cleaned.md")
    else:
        output_file = Path(output_path)

    raw_text = input_file.read_text(encoding="utf-8")
    cleaned_text = clean_markdown(raw_text)
    output_file.write_text(cleaned_text, encoding="utf-8")

    return output_file


def find_mineru_full_md_files(result_dir: str | Path | None = None) -> list[Path]:
    """
    查找 MinerUResult 下每个文档目录中的 full.md。
    """

    project_root = fr()
    mineru_result_dir = Path(result_dir) if result_dir else project_root / "MinerUResult"

    if not mineru_result_dir.exists():
        raise FileNotFoundError(f"MinerUResult 目录不存在: {mineru_result_dir}")

    if not mineru_result_dir.is_dir():
        raise ValueError(f"不是有效目录: {mineru_result_dir}")

    return sorted(
        path
        for path in mineru_result_dir.glob("*/full.md")
        if path.is_file()
    )


def clean_mineru_result_dir(
    result_dir: str | Path | None = None,
    output_name: str = "DataCleaned.md",
) -> list[Path]:
    """
    批量清洗 MinerUResult 下所有文档目录中的 full.md。

    输出默认写回每个文档目录的 DataCleaned.md。
    """

    full_md_files = find_mineru_full_md_files(result_dir)

    if not full_md_files:
        target_dir = Path(result_dir) if result_dir else fr() / "MinerUResult"
        raise FileNotFoundError(f"未找到 MinerU 解析结果 full.md: {target_dir}")

    output_files = []
    for input_md in full_md_files:
        output_md = input_md.with_name(output_name)
        output_files.append(clean_md_file(input_md, output_md))

    return output_files


def main() -> list[Path]:
    """
    允许直接运行当前文件，从 MinerUResult 批量读取并清洗。
    """

    output_files = clean_mineru_result_dir()

    print(f"清洗完成，共输出 {len(output_files)} 个文件:")
    for output_file in output_files:
        print(f"- {output_file}")

    return output_files


if __name__ == "__main__":
    main()
