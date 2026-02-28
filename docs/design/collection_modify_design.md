# Collection 配置修改设计文档

## 1. 背景与动机

### 1.1 当前状态

seekdb SDK 目前创建 Collection 后，其配置信息（HNSW 索引参数、Embedding Function 等）不可修改。用户如果需要调整配置，只能删除并重建 Collection，导致数据丢失。

### 1.2 用户诉求

在实际使用中，用户有以下常见需求：

1. **调优索引查询参数**：例如调整 `ef_search` 以在查询精度和速度之间取得更好的平衡
2. **更换 API 密钥**：Embedding Function 的 API Key 轮换或迁移
3. **调整 API 端点**：切换到不同的 API Endpoint（如灰度、多区域部署）
4. **修改 Collection 名称**：重命名 Collection 用于业务归类

### 1.3 设计约束

并非所有配置都可以修改。以下操作会导致**已有数据与新配置不兼容**，因此必须禁止：

- 修改向量维度（dimension）
- 修改距离度量函数（distance）
- 修改索引构建参数（ef_construction、M）
- 更换 Embedding Function 类型（如从 OpenAI 换为 Qwen）
- 修改 Embedding Function 的模型名（不同模型产生不兼容的向量空间）

### 1.4 参考实现

本设计参考了 [Chroma](https://github.com/chroma-core/chroma) 的 Collection 修改方案，并针对 seekdb 的底层（OceanBase SQL）和多语言 SDK 的场景进行了适配。

## 2. 设计目标

1. **安全性**：通过多层校验机制，确保不可变参数无法被修改，防止数据不一致
2. **灵活性**：允许用户自定义 Embedding Function 的校验规则，适配不同厂商的约束
3. **向后兼容**：现有的 `create_collection` 接口不受影响

## 3. 参数可变性规则

当前相关的配置信息包括

### 3.1 HNSW 索引参数

| 参数 | 可变性 | 原因 |
|------|--------|------|
| `dimension` | **不可变** | 决定 `VECTOR(dim)` 列类型，修改需要重建表和所有数据 |
| `distance` | **不可变** | 决定索引距离度量方式，修改需要重建索引 |
| `ef_construction` | **不可变** | 索引构建时的搜索宽度，已有节点按此参数构建，不可追溯修改 |
| `M` (max_neighbors) | **不可变** | 每层最大邻居数，决定图结构，修改需要重建索引 |
| `ef_search` | **可变** | 查询时参数，仅影响搜索过程的候选集大小，不影响索引结构 |

> **设计原则**：构建时参数（Build-time）不可变，查询时参数（Query-time）可变。

### 3.2 Fulltext 索引参数

| 参数 | 可变性 | 原因 |
|------|--------|------|
| `analyzer` | **不可变** | 分词器类型决定已有文档的分词结果，修改需要重建索引 |
| `properties` | **不可变** | 分词器参数（如 ngram 大小等），修改需要重建索引 |

> 全文索引参数均为构建时参数，暂不支持修改。

### 3.3 Embedding Function 参数

Embedding Function 的可变性规则**由各实现者自行定义**（详见第 5 节），框架层只做以下强制约束：

| 规则 | 级别 | 说明 |
|------|------|------|
| 不允许更换 EF 类型 | **框架强制** | 不同类型的 EF 产生不兼容的向量空间 |
| 不支持修改未实现持久化的 EF | **框架强制** | 无法持久化的 EF 不支持配置管理 |
| 具体参数的可变性 | **实现者定义** | 每个 EF 通过 `validate_config_update` 方法自行校验 |

常见 Embedding Function 的推荐可变性规则：

| EF 实现 | 不可变字段 | 可变字段 |
|---------|-----------|---------|
| OpenAI | model_name, dimensions | api_key_env, api_base, client_kwargs |
| Qwen | model_name, dimensions | api_key_env, api_base |
| SentenceTransformer | model_name | device, kwargs |
| Default | （全部） | （无） |

### 3.4 Collection 名称

| 操作 | 可变性 | 说明 |
|------|--------|------|
| 修改名称 | **可变** | 仅更新元数据表 `sdk_collections` 中的名称，不影响底层数据表 |

## 4. API 设计

### 4.1 `collection.modify()` 方法

这是面向用户的主入口，所有 SDK 语言保持一致的语义。

**Python SDK (pyseekdb)**：

```python
collection.modify(
    name: str | None = None,
    configuration: UpdateConfiguration | UpdateHNSWConfiguration | None = None,
    embedding_function: EmbeddingFunction | None = None,
) -> None
```

**JavaScript SDK (seekdb-js)**：

```typescript
collection.modify(options: {
    name?: string;
    configuration?: UpdateConfiguration | UpdateHNSWConfiguration;
    embeddingFunction?: EmbeddingFunction;
}): Promise<void>
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| name | string \| None | 新的 Collection 名称。为空表示不修改 |
| configuration | UpdateConfiguration \| None | 索引参数更新。仅包含可变参数 |
| embedding_function | EmbeddingFunction \| None | 新的 Embedding Function 实例。必须与原 EF 同类型 |

**返回值**：无返回值。操作成功后，Collection 对象的本地缓存同步更新。

**异常**：

| 异常 | 触发条件 |
|------|----------|
| ValueError | 尝试修改不可变参数（如 distance、dimension） |
| ValueError | 新旧 Embedding Function 类型不匹配 |
| ValueError | Embedding Function 的 `validate_config_update` 校验失败 |
| ValueError | Collection 不存在 |

### 4.2 UpdateHNSWConfiguration

用于更新 HNSW 索引的运行时参数。与创建时使用的 `HNSWConfiguration` 不同，**此类型刻意排除了不可变参数**。

**Python SDK**：

```python
@dataclass
class UpdateHNSWConfiguration:
    """HNSW 索引运行时参数更新。仅包含可在运行时安全修改的参数。"""
    properties: dict[str, str | int | float | bool] | None = None
```

**JavaScript SDK**：

```typescript
interface UpdateHNSWConfiguration {
    properties?: Record<string, string | number | boolean>;
}
```

**对比 Create 和 Update 的参数范围**：

```
HNSWConfiguration (创建时)           UpdateHNSWConfiguration (更新时)
├── dimension          ──────────── ✗ 不可变，排除
├── distance           ──────────── ✗ 不可变，排除
├── properties                      ├── properties
│   ├── ef_construction ─────────── │   ├── ✗ 不可变，排除 (校验拦截)
│   ├── M               ─────────── │   ├── ✗ 不可变，排除 (校验拦截)
│   ├── ef_search        ─────────── │   ├── ef_search ✓ 可变
│   └── ...              ─────────── │   └── ...
```

> **设计决策**：`dimension` 和 `distance` 通过类型设计直接排除（不存在对应字段）。`ef_construction`、`M` 等通过 `properties` 字典传入的不可变参数，在校验层拦截（因为 properties 是开放的 dict，无法在类型层面完全约束）。

### 4.3 UpdateConfiguration

`UpdateConfiguration` 是更新配置的顶层包装，与创建时的 `Configuration` 对应。

**Python SDK**：

```python
class UpdateConfiguration:
    def __init__(self, hnsw: UpdateHNSWConfiguration | None = None):
        self.hnsw = hnsw
```

**JavaScript SDK**：

```typescript
interface UpdateConfiguration {
    hnsw?: UpdateHNSWConfiguration;
}
```

> 当前只支持更新 HNSW 参数。未来如果 Fulltext 索引或 Sparse Vector 索引有可修改的运行时参数，可以在此扩展。

### 4.4 使用示例

#### 4.4.1 修改 HNSW 查询参数

```python
# Python
from pyseekdb import UpdateHNSWConfiguration

collection.modify(
    configuration=UpdateHNSWConfiguration(properties={"ef_search": 128})
)
```

```javascript
// JavaScript
await collection.modify({
    configuration: { hnsw: { properties: { ef_search: 128 } } }
});
```

#### 4.4.2 更换 Embedding Function 的 API 密钥

```python
# Python
from pyseekdb.utils.embedding_functions import OpenAIEmbeddingFunction

new_ef = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",    # 必须与原来一致
    api_key_env="NEW_OPENAI_API_KEY",       # 更换密钥
)
collection.modify(embedding_function=new_ef)
```

```javascript
// JavaScript
const newEf = new OpenAIEmbeddingFunction({
    modelName: "text-embedding-3-small",    // 必须与原来一致
    apiKeyEnv: "NEW_OPENAI_API_KEY",        // 更换密钥
});
await collection.modify({ embeddingFunction: newEf });
```

#### 4.4.3 重命名 Collection

```python
# Python
collection.modify(name="new_collection_name")
```

```javascript
// JavaScript
await collection.modify({ name: "new_collection_name" });
```

#### 4.4.4 错误示例

```python
# ❌ 尝试修改 distance — 被类型系统排除
# UpdateHNSWConfiguration 没有 distance 字段，用户无法传入

# ❌ 尝试通过 properties 修改 ef_construction — 被校验层拦截
collection.modify(
    configuration=UpdateHNSWConfiguration(properties={"ef_construction": 200})
)
# → ValueError: Cannot modify 'ef_construction' after collection creation.
#   This parameter is determined at index build time and requires index rebuild.

# ❌ 尝试更换 EF 类型 — 被框架层拦截
from pyseekdb.utils.embedding_functions import QwenEmbeddingFunction
collection.modify(embedding_function=QwenEmbeddingFunction(model_name="..."))
# → ValueError: Cannot change embedding function type: 'openai' -> 'qwen'.
#   Changing embedding function type would make existing vectors incompatible.

# ❌ 尝试修改模型名 — 被 EF 自身的 validate_config_update 拦截
new_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-large")
collection.modify(embedding_function=new_ef)
# → ValueError: Cannot change model_name after collection creation.
#   Different models produce incompatible embeddings.
```

## 5. Embedding Function 校验协议

### 5.1 设计思路

与 HNSW 参数的"框架统一定义可变性"不同，Embedding Function 采用**多态回调**模式——由各 EF 实现者自行定义哪些参数可以修改。

原因：不同厂商的 EF 有不同的约束。例如：
- OpenAI EF：model_name 和 dimensions 不可变，api_key_env 可变
- SentenceTransformer EF：model_name 不可变，但本地路径可能需要变更（如模型文件迁移）
- 用户自定义 EF：框架无法预知用户的需求

### 5.2 接口定义

在 `EmbeddingFunction` 协议中新增 `validate_config_update` 方法：

**Python SDK**：

```python
class EmbeddingFunction(Protocol[D]):
    # ... 现有方法 ...

    def validate_config_update(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> None:
        """
        校验配置更新是否合法。

        当用户调用 collection.modify(embedding_function=new_ef) 时，框架会调用
        new_ef.validate_config_update(old_ef.get_config(), new_ef.get_config())。

        实现者应在此方法中检查不可变字段是否被修改，如果不合法则抛出 ValueError。
        默认实现为空（允许所有更新）。

        Args:
            old_config: 当前 EF 的配置（来自 old_ef.get_config()）
            new_config: 新 EF 的配置（来自 new_ef.get_config()）

        Raises:
            ValueError: 如果更新不合法
        """
        return  # 默认：允许所有更新
```

**JavaScript SDK**：

```typescript
interface EmbeddingFunction<D = Documents> {
    // ... 现有方法 ...

    /**
     * 校验配置更新是否合法。默认实现允许所有更新。
     * @throws Error 如果更新不合法
     */
    validateConfigUpdate?(
        oldConfig: Record<string, any>,
        newConfig: Record<string, any>,
    ): void;
}
```

### 5.3 实现模式参考

实现者可以根据自身需求选择不同的校验策略：

**策略一：黑名单模式**（列出不可变字段，其余默认可变）

适用于大多数云端 EF，不可变字段较少且明确。

```python
# OpenAI Embedding Function
def validate_config_update(self, old_config, new_config):
    if new_config.get("model_name") != old_config.get("model_name"):
        raise ValueError("Cannot change model_name after collection creation.")
    if new_config.get("dimensions") != old_config.get("dimensions"):
        raise ValueError("Cannot change dimensions after collection creation.")
```

**策略二：白名单模式**（列出可变字段，其余默认不可变）

适用于可变字段较少、需要严格控制的场景。

```python
# BM25 Embedding Function
def validate_config_update(self, old_config, new_config):
    mutable_keys = {"k", "b", "avg_doc_length", "stopwords"}
    for key in new_config:
        if key not in mutable_keys:
            raise ValueError(f"Cannot modify '{key}' for BM25 embedding function.")
```

**策略三：完全开放**（默认行为）

适用于本地模型等所有参数都可安全修改的场景。

```python
# SentenceTransformer Embedding Function
def validate_config_update(self, old_config, new_config):
    return  # 允许所有修改（如模型文件路径迁移）
```

### 5.4 方法是否必须实现

`validate_config_update` 为**可选方法**，提供默认实现（允许所有更新）。这样：
- 现有 EF 不受影响（向后兼容）
- 新开发的 EF 推荐实现此方法以保证安全性

## 6. 校验层次总结

修改请求经过**五层校验**，形成纵深防御：

```
用户调用 collection.modify(...)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  第 1 层：类型系统约束                               │
│  UpdateHNSWConfiguration 没有 dimension/distance    │
│  字段，编码阶段就不可能传入这些参数。                  │
│  对于强类型语言(TypeScript)尤其有效。                 │
└───────────────────────┬───────────────────────────┘
                        │ 通过
                        ▼
┌───────────────────────────────────────────────────┐
│  第 2 层：Update 类型的 properties 黑名单校验         │
│  UpdateHNSWConfiguration 构造时校验 properties       │
│  中的 ef_construction/M 等已知不可变键。              │
│  即使通过 dict 绕过类型约束，也会在此被拦截。          │
└───────────────────────┬───────────────────────────┘
                        │ 通过
                        ▼
┌───────────────────────────────────────────────────┐
│  第 3 层：框架级 EF 校验                              │
│  - name() 必须一致（不能换 EF 类型）                  │
│  - 必须支持持久化（实现 get_config 等方法）            │
└───────────────────────┬───────────────────────────┘
                        │ 通过
                        ▼
┌───────────────────────────────────────────────────┐
│  第 4 层：EF 实现者自定义校验                          │
│  调用 new_ef.validate_config_update(old, new)       │
│  由 EF 实现者根据具体约束校验字段可变性。               │
└───────────────────────┬───────────────────────────┘
                        │ 通过
                        ▼
┌───────────────────────────────────────────────────┐
│  第 5 层：数据库执行层兜底                             │
│  OceanBase ALTER INDEX 执行时，如果传入了数据库        │
│  不支持修改的参数，数据库本身会返回错误。                │
└───────────────────────┬───────────────────────────┘
                        │ 通过
                        ▼
                   修改成功 ✓
```

## 7. 存储与持久化

### 7.1 数据存储位置

Collection 的配置信息存储在两个地方：

| 存储位置 | 内容 | 修改方式 |
|----------|------|----------|
| `sdk_collections.settings` (JSON) | EF 配置、Collection 名称等 SDK 层面的元数据 | `UPDATE sdk_collections SET settings = '...'` |
| 数据表的索引定义 | HNSW 索引参数（distance, ef_search 等） | `ALTER INDEX` SQL（具体语法待确认） |

### 7.2 sdk_collections 表结构

```sql
CREATE TABLE sdk_collections (
    collection_id   CHAR(32) PRIMARY KEY,
    collection_name STRING,
    settings        JSON,         -- SDK 配置，包含 EF 信息
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 7.3 settings JSON 格式

修改 EF 后的 settings 示例：

```json
{
  "version": 2,
  "embedding_function": {
    "name": "openai",
    "properties": {
      "model_name": "text-embedding-3-small",
      "api_key_env": "NEW_OPENAI_API_KEY",
      "api_base": "https://api.openai.com/v1",
      "dimensions": null
    }
  }
}
```

### 7.4 HNSW 参数修改的 SQL 映射

修改 HNSW 运行时参数需要通过 OceanBase 的 ALTER INDEX 语句下发：

```sql
-- 预期语法（需确认 OceanBase 实际支持的语法）
ALTER INDEX idx_vec ON `c$v2$<collection_id>` REBUILD WITH (ef_search=128);
```

> **待确认项**：OceanBase 的 ALTER INDEX 对向量索引的具体语法和支持的参数范围，需与数据库团队确认。

### 7.5 Collection 重命名的 SQL 映射

V2 Collection 的底层表名为 `c$v2${collection_id}`，与 Collection 名称无关。重命名只需更新元数据：

```sql
UPDATE sdk_collections
SET collection_name = 'new_name'
WHERE collection_id = '<id>';
```

## 8. 执行流程

### 8.1 modify() 整体流程

```
collection.modify(name=..., configuration=..., embedding_function=...)
    │
    ├─ 1. 参数校验
    │     ├─ configuration: UpdateHNSWConfiguration 构造校验
    │     ├─ embedding_function: 框架校验 + EF 自身校验
    │     └─ name: 合法性校验（字符规则、长度、唯一性）
    │
    ├─ 2. 更新 HNSW 索引参数（如果 configuration 不为空）
    │     └─ 执行 ALTER INDEX SQL
    │
    ├─ 3. 更新 sdk_collections 元数据（如果 EF 或 name 变更）
    │     └─ 执行 UPDATE sdk_collections SQL
    │
    └─ 4. 更新 Collection 对象的本地缓存
          ├─ self._name = new_name
          ├─ self._embedding_function = new_ef
          └─ 返回成功
```

### 8.2 错误处理

| 步骤 | 失败处理 |
|------|----------|
| 参数校验失败 | 立即抛出 ValueError，不执行任何修改 |
| ALTER INDEX 失败 | 抛出异常，sdk_collections 不会被更新（保持一致性） |
| UPDATE sdk_collections 失败 | 抛出异常。注意此时 ALTER INDEX 可能已执行成功，存在不一致风险（见 8.3） |

### 8.3 一致性说明

由于 HNSW 参数和 EF 配置分别存储在索引定义和 sdk_collections 中，如果同时修改两者，不是原子操作。应对策略：

1. **优先执行 ALTER INDEX**：如果 ALTER INDEX 失败，不会更新 sdk_collections，保持一致
2. **ALTER INDEX 成功但 UPDATE 失败**：HNSW 运行时参数已变更（如 ef_search），但 EF 配置未变更。此时 HNSW 参数变更不会导致数据不一致（仅影响查询行为），可以安全重试
3. **建议用户分步操作**：如果同时修改 HNSW 参数和 EF 配置，建议分两次 modify 调用

> 未来如果需要更强的一致性保证，可以考虑在 SDK 层加入补偿机制或事务支持。

## 9. V1 Collection 兼容性

| 功能 | V1 Collection | V2 Collection |
|------|--------------|--------------|
| 修改 HNSW 参数 | 支持（通过 ALTER INDEX） | 支持 |
| 修改 EF 配置 | **不支持**（无 sdk_collections 元数据） | 支持 |
| 重命名 | **不支持**（表名包含 Collection 名称） | 支持 |

V1 Collection 调用不支持的修改操作时，应返回明确的错误信息，引导用户迁移到 V2。

## 10. 未来扩展

### 10.1 Schema 体系下的 modify

当 Schema 功能（参见 `schema_design.md`）实现后，`modify()` 方法需要扩展以支持：

- Sparse Vector Index 的运行时参数修改
- Sparse Embedding Function 的配置修改

接口可扩展为：

```python
collection.modify(
    name=...,
    configuration=...,              # HNSW 运行时参数
    embedding_function=...,         # 稠密向量 EF
    sparse_embedding_function=...,  # 稀疏向量 EF（未来）
)
```

### 10.2 Fulltext 索引参数

如果 OceanBase 未来支持全文索引的运行时参数修改（如动态切换分词器的某些参数），可以在 `UpdateConfiguration` 中扩展 `fulltext` 字段。

### 10.3 索引重建

当前设计明确排除了需要重建索引的修改操作。未来如果有「在线索引重建」的能力，可以考虑支持以下操作，但需要加上明确的用户确认机制（如 `force=True` 参数）：

- 修改 distance（触发索引重建）
- 修改 ef_construction / M（触发索引重建）

## 11. 待确认事项

| 编号 | 事项 | 责任方 | 状态 |
|------|------|--------|------|
| 1 | OceanBase ALTER INDEX 对向量索引的具体 SQL 语法和支持的参数范围 | 数据库团队 | 待确认 |
| 2 | OceanBase 是否支持在线修改 ef_search 而不需要 REBUILD | 数据库团队 | 待确认 |
| 3 | seekdb-js 的 EmbeddingFunction 接口现状，是否已有 `getConfig` / `name` 方法 | JS SDK 团队 | 待确认 |
| 4 | 是否需要在 modify 操作上做权限控制（如区分管理员和普通用户） | 产品 | 待确认 |
| 5 | Collection 重命名是否需要检查唯一性约束（同 database 下不能重名） | SDK 团队 | 待确认 |

## 12. 附录

### A. 与 Chroma 实现的对比

| 维度 | Chroma | SeekDB SDK |
|------|--------|------------|
| HNSW 可变性控制 | 通过 Create/Update 两套 TypedDict，字段级排除 | 类似：通过 Create/Update 两套类型 + properties 黑名单校验 |
| EF 可变性控制 | `validate_config_update` 回调方法 | 相同方案 |
| 存储后端 | 自有 Segment 存储 | OceanBase SQL（ALTER INDEX + sdk_collections 表） |
| 一致性 | 单事务 | 非原子（ALTER INDEX + UPDATE 分两步），需注意顺序 |
| 多语言支持 | Python only | Python + JavaScript（需统一接口语义） |

### B. 相关文档

- [Schema 设计文档](./schema_design.md)
- [OceanBase 向量索引文档](https://www.oceanbase.com/docs/oceanbase-database-cn)
- [Chroma Collection Configuration 源码](https://github.com/chroma-core/chroma/blob/main/chromadb/api/collection_configuration.py)
