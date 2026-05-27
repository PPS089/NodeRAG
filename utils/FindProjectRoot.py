from pathlib import Path

# 获取项目根目录
def find_project_root(start: Path | None = None) -> Path:
    """
    从当前文件所在目录开始，向上查找 pyproject.toml。
    找到 pyproject.toml 的目录，就认为是项目根目录。
    """
    current = start or Path(__file__).resolve().parent

    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError("未找到 pyproject.toml，无法确定项目根目录")