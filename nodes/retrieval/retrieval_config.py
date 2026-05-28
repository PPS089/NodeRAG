from __future__ import annotations


DEFAULT_PER_QUERY_K = 12
DEFAULT_TOP_K = 8
DEFAULT_CONTEXT_LIMIT = 30
DEFAULT_BM25_K = 20
DEFAULT_RERANK_POOL_SIZE = 40
DEFAULT_MMR_LAMBDA = 0.72

BM25_K1 = 1.5
BM25_B = 0.75

TABLE_INTENT_TERMS = {
    "表",
    "表格",
    "价格",
    "金额",
    "预算",
    "薪酬",
    "等级",
    "比例",
    "字段",
    "清单",
    "模板",
}
