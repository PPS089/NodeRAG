import re
from pathlib import Path
from utils.FindProjectRoot import find_project_root as fr


CODE_BLOCK_RE = re.compile(r"(```.*?```|~~~.*?~~~)", flags=re.S)


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

    # 保护代码块，避免后续规则误处理代码内容
    text, code_blocks = protect_code_blocks(text)

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

    # 恢复代码块
    text = restore_code_blocks(text, code_blocks)

    # 恢复后再做一次轻量清洗
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 文档结尾保留一个换行
    text = text.strip() + "\n"

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


if __name__ == "__main__":
    # 获取项目根目录
    path = fr()

    # 待处理文档
    input_md = (
        path
        / ".mineru_cache"
        / "5ac5a6c3b1da4f87b1045865a51cbcf4e73faad43f5873e6676946902f981f64"
        / "full.md"
    )

    # 已处理文档：必须放在 full.md 同目录
    output_md = input_md.with_name("DataCleaned.md")

    result = clean_md_file(input_md, output_md)

    print(f"清洗完成，输出文件: {result}")