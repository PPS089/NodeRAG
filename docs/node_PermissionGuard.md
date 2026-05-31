# PermissionGuard — 权限校验节点

## 概述

`PermissionGuard` 实现基于等级的知识库权限控制（L1~L5），在 RAG pipeline 中做**检索前过滤**（限制可查询文档范围）和**检索后兜底过滤**（防止未授权内容进入上下文）。

**路径**：`nodes/auth/PermissionGuard.py`

## 在链路中的位置

```
# 检索前过滤
IntentRouter → ... → PermissionGuard (stage_apply_permission) → ChromaRetriever

# 检索后过滤
ChromaRetriever → PermissionGuard (stage_filter_retrieval_permission) → Reranker
```

## 权限等级体系

| 等级 | 数值 | 可访问范围 | 典型角色 |
| --- | --- | --- | --- |
| L1 | 1 | 仅公开制度 | 访客/全员 |
| L2 | 2 | L1 + HR基础 + 产品资料 | 普通员工 |
| L3 | 3 | L1~L2 + 合同 + 项目 | 经理 |
| L4 | 4 | L1~L3 + 薪酬 + 价格 | 高级管理者 |
| L5 | 5 | 全部知识库（含财务预算） | 最高管理者 |

**继承规则**：高等级自动继承低等级权限。

## 权限配置

`config/knowledge_permissions.json`：

```json
{
  "default_access": "deny",
  "levels": { "L1": 1, "L2": 2, ... },
  "knowledge_bases": [
    {
      "knowledge_base": "HR基础制度库",
      "required_level": "L2",
      "permission_code": "KB.HR.BASIC.READ",
      "document_names": ["HR基础制度库", "HR基础制度库.pdf"],
      "document_patterns": ["*HR基础制度库*"]
    }
  ]
}
```

### 匹配规则

| 匹配方式 | 字段 | 使用阶段 |
| --- | --- | --- |
| 精确匹配 | `document_names` | 检索前过滤（Chroma `where` 仅支持精确匹配） |
| 通配符匹配 | `document_patterns`（fnmatch） | 检索后兜底过滤 |

## 两层过滤

### 1. 检索前过滤 (`apply_permission_to_retrieval_inputs`)

```
if retrieval_inputs 指定了 document_names:
    → 过滤出权限允许的文档名
else:
    → 使用当前等级允许的全部 document_names
```

输出 `retrieval_inputs.document_names`（精确匹配列表）和 `permission_denied` 标记。

### 2. 检索后过滤 (`filter_retrieval_result_by_permission`)

```
for each hit:
    metadata.document_name 能匹配任意 allowed_rule 的 document_names 或 document_patterns
    → 保留
    → 否则 → 移入 denied_hits
```

同时过滤 `hit.expanded_context`。

## 核心函数

```python
build_permission_context(permission_level, config_path) → dict
can_access_document(document_name, permission_context) → bool
apply_permission_to_retrieval_inputs(retrieval_inputs, permission_context) → dict
filter_retrieval_result_by_permission(retrieval_result, permission_context) → dict
```

## CLI 使用

```powershell
# 查看 L3 权限等级可访问的知识库
.\.venv\Scripts\python.exe nodes\auth\PermissionGuard.py --permission-level L3
```

## 配置扩展

新增知识库时只需修改 `config/knowledge_permissions.json`：

```json
{
  "knowledge_base": "新知识库",
  "required_level": "L3",
  "permission_code": "KB.NEW.READ",
  "document_names": ["新知识库", "新知识库.pdf"],
  "document_patterns": ["*新知识库*"]
}
```

**注意**：Chroma 检索前过滤只能使用精确 `document_name`，因此真实 PDF 名称最好写入 `document_names`；`document_patterns` 主要用于检索后兜底。

## 上下游契约

- **上游**：`PipelineContext.retrieval_inputs` / `retrieval_result`
- **下游**：`ChromaRetriever`（检索前）/ `Reranker`（检索后）
- `permission_denied=True` 时 Pipeline 会提前结束
