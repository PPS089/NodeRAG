from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.chunks.HybridMarkdownChunk import hybrid_chunk_mineru_result_dir  # noqa: E402
from nodes.documents.DataClean import clean_mineru_result_dir  # noqa: E402
from nodes.documents.MinerUStandardReader import MinerUStandardReader  # noqa: E402
from nodes.embeddings.ChromaBailianEmbedding import index_hybrid_chunks  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "MinerUResult"


def find_pdf_files(data_dir: str | Path) -> List[Path]:
    source_dir = Path(data_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"data 目录不存在: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"不是有效目录: {source_dir}")

    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def build_pdf_summary(pdf_paths: List[Path]) -> List[Dict[str, Any]]:
    return [
        {
            "file_name": path.name,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for path in pdf_paths
    ]


class IngestPipeline:
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        result_dir: str | Path = DEFAULT_RESULT_DIR,
        model_version: str = "vlm",
        language: str = "ch",
        is_ocr: bool = False,
        enable_table: bool = True,
        enable_formula: bool = True,
        page_ranges: Optional[str] = None,
        force_reparse: bool = False,
        confirm_reparse: bool = False,
        skip_mineru: bool = False,
        skip_clean: bool = False,
        skip_chunk: bool = False,
        skip_embedding: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.model_version = model_version
        self.language = language
        self.is_ocr = is_ocr
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.page_ranges = page_ranges
        self.force_reparse = force_reparse
        self.confirm_reparse = confirm_reparse
        self.skip_mineru = skip_mineru
        self.skip_clean = skip_clean
        self.skip_chunk = skip_chunk
        self.skip_embedding = skip_embedding

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        started_at = time.time()
        pdf_paths = find_pdf_files(self.data_dir)
        if not pdf_paths:
            raise FileNotFoundError(f"目录下未找到 PDF 文件: {self.data_dir}")

        result: Dict[str, Any] = {
            "data_dir": str(self.data_dir),
            "result_dir": str(self.result_dir),
            "pdf_count": len(pdf_paths),
            "pdfs": build_pdf_summary(pdf_paths),
            "steps": {},
        }

        if dry_run:
            result["dry_run"] = True
            result["elapsed_seconds"] = round(time.time() - started_at, 3)
            return result

        if self.skip_mineru:
            result["steps"]["mineru"] = {"skipped": True}
        else:
            reader = MinerUStandardReader(result_dir=str(self.result_dir))
            mineru_results = reader.read_data_dir(
                data_dir=str(self.data_dir),
                model_version=self.model_version,
                language=self.language,
                is_ocr=self.is_ocr,
                enable_table=self.enable_table,
                enable_formula=self.enable_formula,
                page_ranges=self.page_ranges,
                force_reparse=self.force_reparse,
                confirm_reparse=self.confirm_reparse,
            )
            result["steps"]["mineru"] = {
                "skipped": False,
                "parsed_pdf_count": len(mineru_results),
                "cache_note": "MinerUStandardReader 使用文件内容 hash + 解析参数缓存，相同内容默认不重复上传。",
            }

        if self.skip_clean:
            result["steps"]["clean"] = {"skipped": True}
        else:
            cleaned_files = clean_mineru_result_dir(result_dir=self.result_dir)
            result["steps"]["clean"] = {
                "skipped": False,
                "output_count": len(cleaned_files),
                "outputs": [str(path) for path in cleaned_files],
            }

        if self.skip_chunk:
            result["steps"]["chunk"] = {"skipped": True}
        else:
            chunk_outputs = hybrid_chunk_mineru_result_dir(result_dir=self.result_dir)
            result["steps"]["chunk"] = {
                "skipped": False,
                "document_count": len(chunk_outputs),
                "outputs": chunk_outputs,
            }

        if self.skip_embedding:
            result["steps"]["embedding"] = {"skipped": True}
        else:
            embedding_result = index_hybrid_chunks(result_dir=self.result_dir)
            result["steps"]["embedding"] = embedding_result
            result["steps"]["embedding"]["incremental_note"] = (
                "ChromaBailianEmbedding 按 chunk id 增量入库；已存在 chunk 会跳过，chunk 模式会删除同文档过期 chunk。"
            )

        result["elapsed_seconds"] = round(time.time() - started_at, 3)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键知识库入库：PDF -> MinerU -> 清洗 -> Hybrid 分片 -> Chroma 向量库。")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="PDF 输入目录。")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="MinerUResult 输出目录。")
    parser.add_argument("--model-version", default="vlm", help="MinerU 模型版本。")
    parser.add_argument("--language", default="ch", help="MinerU 文档语言。")
    parser.add_argument("--ocr", action="store_true", help="开启 MinerU OCR。")
    parser.add_argument("--disable-table", action="store_true", help="关闭 MinerU 表格识别。")
    parser.add_argument("--disable-formula", action="store_true", help="关闭 MinerU 公式识别。")
    parser.add_argument("--page-ranges", help="MinerU 解析页码范围。")
    parser.add_argument("--force-reparse", action="store_true", help="强制重新上传并解析 PDF。")
    parser.add_argument("--confirm-reparse", action="store_true", help="缓存命中时询问是否重新解析。")
    parser.add_argument("--skip-mineru", action="store_true", help="跳过 MinerU 解析，直接使用已有 MinerUResult。")
    parser.add_argument("--skip-clean", action="store_true", help="跳过 Markdown 清洗。")
    parser.add_argument("--skip-chunk", action="store_true", help="跳过 Hybrid 分片。")
    parser.add_argument("--skip-embedding", action="store_true", help="跳过 Chroma embedding 入库。")
    parser.add_argument("--dry-run", action="store_true", help="只列出将处理的 PDF，不执行入库。")
    parser.add_argument("--output", "-o", help="输出执行摘要 JSON 文件。")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    pipeline = IngestPipeline(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        model_version=args.model_version,
        language=args.language,
        is_ocr=args.ocr,
        enable_table=not args.disable_table,
        enable_formula=not args.disable_formula,
        page_ranges=args.page_ranges,
        force_reparse=args.force_reparse,
        confirm_reparse=args.confirm_reparse,
        skip_mineru=args.skip_mineru,
        skip_clean=args.skip_clean,
        skip_chunk=args.skip_chunk,
        skip_embedding=args.skip_embedding,
    )
    result = pipeline.run(dry_run=args.dry_run)
    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)
    return result


if __name__ == "__main__":
    main()
