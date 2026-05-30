"""
BM25 词表与文档频率持久化 — 用于稀疏向量检索的 IDF 统计。

核心功能：
- 在 ingestion 阶段记录词表（vocab）和文档频率（doc_freq）
- 持久化到 JSON 文件，检索时加载，保证 IDF 统计跨进程/重启一致
- 支持增量更新（新增文档 / 删除文档）
- 检索时使用持久化的统计信息，而非每次重新计算全量语料的 IDF
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.retrieval.retrieval_config import BM25_B, BM25_K1  # noqa: E402
from nodes.retrieval.retrieval_utils import tokenize_for_search  # noqa: E402


# ------------------
# 配置与常量
# ------------------

DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "bm25_state.json"

# 版本号，用于未来字段扩展兼容
STATE_VERSION = 1

# 并发保护
_state_lock = threading.Lock()


# ------------------
# Pydantic 数据模型
# ------------------


class BM25State(BaseModel):
    """BM25 持久化状态模型。"""

    version: int = Field(default=STATE_VERSION, description="状态文件版本号")
    total_docs: int = Field(default=0, description="语料总文档数")
    sum_token_len: int = Field(default=0, description="所有文档 token 总长度")
    vocab: Dict[str, int] = Field(default_factory=dict, description="词表：token -> index")
    doc_freq: Dict[str, int] = Field(default_factory=dict, description="文档频率：token -> 包含该词的文档数")


# ------------------
# 核心服务类
# ------------------


class BM25StateManager:
    """
    BM25 词表与文档频率管理器。

    功能：
    - 持久化：vocab + doc_freq + total_docs + sum_token_len → JSON 文件
    - 加载：从 JSON 文件恢复状态
    - 增量更新：新增文档 / 删除文档时同步更新统计信息
    - 计算 BM25 分数：基于持久化的 IDF 统计
    """

    def __init__(
        self,
        state_path: Optional[str | Path] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", DEFAULT_STATE_PATH))
        self._k1 = float(k1)
        self._b = float(b)

        # 词表：token → index（整数索引，供稀疏向量使用）
        self._vocab: Dict[str, int] = {}
        self._vocab_counter = 0

        # 文档频率：token → 包含该词的文档数
        self._doc_freq: Counter[str] = Counter()

        # 全局统计
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0

        # 加载持久化状态
        self._load()

    # --------
    # 持久化
    # --------

    def _load(self) -> None:
        """从 JSON 文件加载状态。"""
        path = self._state_path
        if not path.is_file():
            return

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        if raw.get("version") != STATE_VERSION:
            # 版本不匹配时跳过加载（可扩展兼容逻辑）
            return

        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))

        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            self._vocab_counter = 0

        self._recompute_avg_len()

    def _persist_unlocked(self) -> None:
        """持久化状态到 JSON 文件（不持有锁，由 _persist 调用）。"""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _persist(self) -> None:
        """线程安全的持久化。"""
        with _state_lock:
            self._persist_unlocked()

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    # --------
    # 增量更新
    # --------

    def increment_add_documents(self, texts: Sequence[str]) -> None:
        """
        新增文档时增量更新 N / df / 长度和。

        :param texts: 文档文本列表（每个元素视为一篇文档）
        """
        if not texts:
            return

        with _state_lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)

                self._sum_token_len += doc_len
                self._total_docs += 1

                for token in set(tokens):
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    self._doc_freq[token] += 1

            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_remove_documents(self, texts: Sequence[str]) -> None:
        """
        删除文档时从统计中移除对应信息。

        词表索引不回收（避免与已存在的稀疏向量维度冲突）。
        :param texts: 文档文本列表
        """
        if not texts:
            return

        with _state_lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)

                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                self._total_docs = max(0, self._total_docs - 1)

                for token in set(tokens):
                    if token not in self._doc_freq:
                        continue
                    self._doc_freq[token] -= 1
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]

            self._recompute_avg_len()
            self._persist_unlocked()

    def increment_add_documents_by_file(
        self,
        file_path: str | Path,
        file_pattern: Optional[str] = None,
    ) -> int:
        """
        增量添加指定文件的所有 chunk（从 MinerUResult 的 chunk JSON 中读取）。

        :param file_path: MinerUResult 下某个文档的目录路径
        :param file_pattern: 可选，glob 模式筛选 chunk 文件
        :return: 添加的文档数
        """
        from pathlib import Path as PP

        base_path = PP(file_path)
        if not base_path.is_dir():
            return 0

        import glob

        pattern = file_pattern or "*.chunk.json"
        chunk_files = sorted(base_path.glob(pattern))
        if not chunk_files:
            # 兼容：直接遍历目录下所有 JSON
            chunk_files = sorted(base_path.glob("*.json"))

        added_count = 0
        texts = []
        for chunk_file in chunk_files:
            try:
                raw = json.loads(chunk_file.read_text(encoding="utf-8"))
                # MinerU chunk JSON 格式：{"id": "...", "content": "...", ...}
                if isinstance(raw, dict):
                    content = raw.get("content", "") or raw.get("text", "") or raw.get("document", "")
                    if content:
                        texts.append(content)
                elif isinstance(raw, list):
                    for item in raw:
                        content = item.get("content", "") or item.get("text", "") or ""
                        if content:
                            texts.append(content)
            except Exception:
                continue

        if texts:
            self.increment_add_documents(texts)
            added_count = len(texts)

        return added_count

    # --------
    # Tokenizer
    # --------

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """与 retrieval_utils.tokenize_for_search 保持一致的分词器。"""
        return tokenize_for_search(text)

    # --------
    # BM25 计算
    # --------

    def _idf_for_token(self, token: str) -> float:
        """计算单个 token 的 IDF 值。"""
        n = max(self._total_docs, 0)
        df = self._doc_freq.get(token, 0)
        if df == 0:
            return math.log((n + 1) / 1)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        计算 query 对单篇文档的 BM25 分数。

        :param query_tokens: 查询 token 列表（允许重复，用于加权）
        :param doc_tokens: 文档 token 列表
        :return: BM25 分数
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        doc_tf = Counter(doc_tokens)
        query_tf = Counter(query_tokens)

        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        score = 0.0
        for token, query_weight in query_tf.items():
            freq = doc_tf.get(token, 0)
            if freq == 0:
                continue

            idf = self._idf_for_token(token)
            denominator = freq + self._k1 * (1 - self._b + self._b * doc_len / avg)
            score += query_weight * idf * (freq * (self._k1 + 1) / denominator)

        return score

    def score_documents(
        self,
        query_text: str,
        documents: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对文档列表计算 BM25 分数并排序。

        :param query_text: 查询文本
        :param documents: 文档列表，每个 dict 包含 document/text 字段
        :return: 按 BM25 分数降序排列的文档列表（增加 bm25_score 字段）
        """
        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return list(documents)

        scored: List[Dict[str, Any]] = []
        for doc in documents:
            doc_text = doc.get("document", "") or doc.get("text", "") or ""
            doc_tokens = self.tokenize(doc_text)
            score = self.bm25_score(query_tokens, doc_tokens)
            scored_doc = dict(doc)
            scored_doc["bm25_score"] = score
            scored.append(scored_doc)

        scored.sort(key=lambda item: item["bm25_score"], reverse=True)
        return scored

    # --------
    # 稀疏向量（供 Milvus 混合检索使用）
    # --------

    def get_sparse_vector(self, text: str) -> Tuple[Dict[int, float], bool]:
        """
        获取文本的稀疏向量表示 {index: score}。

        :param text: 输入文本
        :return: (sparse_vector, vocab_changed)
        """
        tokens = self.tokenize(text)
        doc_len = len(tokens)
        tf = Counter(tokens)
        sparse_vector: Dict[int, float] = {}
        vocab_changed = False

        n = max(self._total_docs, 0)
        avg = max(self._avg_doc_len, 1.0)

        with _state_lock:
            for token, freq in tf.items():
                if token not in self._vocab:
                    self._vocab[token] = self._vocab_counter
                    self._vocab_counter += 1
                    vocab_changed = True

                idx = self._vocab[token]
                idf = self._idf_for_token(token)

                numerator = freq * (self._k1 + 1)
                denominator = freq + self._k1 * (1 - self._b + self._b * doc_len / avg)
                score = idf * numerator / denominator
                if score > 0:
                    sparse_vector[idx] = float(score)

        return sparse_vector, vocab_changed

    def get_sparse_vector_batch(self, texts: Sequence[str]) -> Tuple[List[Dict[int, float]], bool]:
        """
        批量获取稀疏向量。

        :param texts: 文本列表
        :return: (sparse_vectors, any_vocab_changed)
        """
        if not texts:
            return [], False

        out: List[Dict[int, float]] = []
        any_vocab_changed = False

        with _state_lock:
            for text in texts:
                sparse_vector, vocab_changed = self.get_sparse_vector(text)
                out.append(sparse_vector)
                any_vocab_changed = any_vocab_changed or vocab_changed

            if any_vocab_changed:
                self._persist_unlocked()

        return out, any_vocab_changed

    # --------
    # 查询接口
    # --------

    def get_stats(self) -> Dict[str, Any]:
        """返回当前统计信息的快照。"""
        with _state_lock:
            return {
                "version": STATE_VERSION,
                "total_docs": self._total_docs,
                "sum_token_len": self._sum_token_len,
                "avg_doc_len": round(self._avg_doc_len, 2),
                "vocab_size": len(self._vocab),
                "doc_freq_size": len(self._doc_freq),
                "state_path": str(self._state_path),
            }

    def reset(self) -> None:
        """重置所有状态（慎用）。"""
        with _state_lock:
            self._vocab.clear()
            self._vocab_counter = 0
            self._doc_freq.clear()
            self._total_docs = 0
            self._sum_token_len = 0
            self._avg_doc_len = 1.0
            self._persist_unlocked()


# ------------------
# 全局单例
# ------------------

_bm25_state_manager: Optional[BM25StateManager] = None


def get_bm25_state_manager() -> BM25StateManager:
    """获取 BM25 状态管理器的全局单例。"""
    global _bm25_state_manager
    if _bm25_state_manager is None:
        _bm25_state_manager = BM25StateManager()
    return _bm25_state_manager


# ------------------
# CLI 入口
# ------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BM25 词表持久化管理工具")
    parser.add_argument("--reset", action="store_true", help="重置 BM25 状态")
    parser.add_argument("--stats", action="store_true", help="打印当前统计信息")
    parser.add_argument(
        "--add-dir",
        help="MinerUResult 下的文档目录路径，批量添加 chunks 到词表",
    )
    parser.add_argument(
        "--add-file",
        action="append",
        default=[],
        help="单个文档文件路径，添加到词表（可重复）",
    )
    parser.add_argument("--state-path", help="覆盖状态文件路径")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    manager = get_bm25_state_manager()

    if args.state_path:
        manager = BM25StateManager(state_path=args.state_path)

    if args.reset:
        manager.reset()
        print("BM25 状态已重置。")

    if args.stats:
        stats = manager.get_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.add_dir:
        added = manager.increment_add_documents_by_file(args.add_dir)
        print(f"已从 {args.add_dir} 添加 {added} 个 chunk。")

    for file_path in args.add_file:
        try:
            text = Path(file_path).read_text(encoding="utf-8")
            manager.increment_add_documents([text])
            print(f"已添加文件: {file_path}")
        except Exception as e:
            print(f"添加文件失败 {file_path}: {e}")