# pyseekdb Schema 以及稀疏索引支持设计文档

## 1. 概述

### 1.1 背景

pyseekdb 目前支持基本的 Collection 配置，包括 HNSW 向量索引和全文索引配置。为了提供更灵活的索引控制能力，特别是支持稀疏向量索引（Sparse Vector Index）以实现更强大的混合搜索（Hybrid Search）能力，我们需要引入 Schema 功能。

### 1.2 参考资料

- [Chroma Schema 官方文档](https://docs.trychroma.com/cloud/schema/overview)
- [Chroma Schema Basics](https://docs.trychroma.com/cloud/schema/schema-basics)
- [Chroma Sparse Vector Search](https://docs.trychroma.com/cloud/schema/sparse-vector-search)
- [Chroma Index Configuration Reference](https://docs.trychroma.com/cloud/schema/index-reference)
- [OceanBase 向量功能文档](https://www.oceanbase.ai/docs/vector-function/)
- [OceanBase ob-vsag 项目](https://github.com/oceanbase/ob-vsag)

### 1.3 设计目标

1. **索引精细控制**：允许用户配置向量索引（HNSW 参数）、稀疏向量索引、全文索引的具体参数
2. **稀疏向量支持**：支持稀疏向量索引（Sparse Vector Index），实现基于关键词的检索（如 BM25、SPLADE）
3. **混合搜索增强**：结合稠密向量（Dense Vector）和稀疏向量（Sparse Vector）实现更强大的混合搜索。当前混搜接口还不支持，未来可以对接此功能。
4. **向后兼容**：现有的 `Configuration` 和 `HNSWConfiguration` 继续工作，Schema 作为增强功能

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         User API Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Schema                                                          │
│  ├── vector_index: VectorIndexConfig (稠密向量索引配置)         │
│  ├── sparse_vector_index: SparseVectorIndexConfig [NEW]         │
│  └── fulltext_index: FulltextIndexConfig (全文索引配置)         │
├─────────────────────────────────────────────────────────────────┤
│  Index Configuration Classes                                     │
│  ├── VectorIndexConfig (稠密向量索引 - HNSW)                    │
│  ├── SparseVectorIndexConfig (稀疏向量索引) [NEW]               │
│  └── FulltextIndexConfig (全文索引)                             │
├─────────────────────────────────────────────────────────────────┤
│  Embedding Functions                                             │
│  ├── EmbeddingFunction (稠密向量)                               │
│  └── SparseEmbeddingFunction (稀疏向量) [NEW]                   │
├─────────────────────────────────────────────────────────────────┤
│                    OceanBase / SeekDB Backend                    │
│  (metadata 存储为 JSON 类型，无需单独的倒排索引)                │
└─────────────────────────────────────────────────────────────────┘
```


### 2.2 核心类设计

#### 2.2.1 Schema 类

Schema 类用来描述新增索引或者相关索引的配置信息。此类将替换当前Configuration的使用。
> 当前不需要支持删除索引，未来如果有需要可以支持。

```python
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class Schema:
    """
    Schema 配置类，用于控制 Collection 的索引行为。

    PySeekDB Schema 设计简洁，直接配置三种索引：
    1. vector_index: 稠密向量索引（HNSW）
    2. sparse_vector_index: 稀疏向量索引（用于混合搜索）
    3. fulltext_index: 全文索引

    Example:
        >>> # 方式1: 使用构造函数
        >>> schema = Schema(
        ...     vector_index=VectorIndexConfig(
        ...         space="cosine",
        ...         embedding_function=OpenAIEmbeddingFunction(api_key="...")
        ...     ),
        ...     sparse_vector_index=SparseVectorIndexConfig(
        ...         embedding_function=BM25EmbeddingFunction(),
        ...         source_key=K.DOCUMENT
        ...     ),
        ...     fulltext_index=FulltextIndexConfig(analyzer="ik")
        ... )
        >>>
        >>> # 方式2: 使用链式调用
        >>> schema = (Schema()
        ...     .with_vector_index(VectorIndexConfig(space="cosine"))
        ...     .with_sparse_vector_index(SparseVectorIndexConfig(
        ...         embedding_function=BM25EmbeddingFunction()
        ...     ))
        ... )
        >>>
        >>> collection = client.create_collection("my_collection", schema=schema)
    """
    vector_index: Optional["VectorIndexConfig"] = None
    sparse_vector_index: Optional["SparseVectorIndexConfig"] = None
    fulltext_index: Optional["FulltextIndexConfig"] = None

    def with_vector_index(self, config: "VectorIndexConfig") -> "Schema":
        """
        配置稠密向量索引。

        Args:
            config: VectorIndexConfig 配置对象

        Returns:
            self，支持链式调用
        """
        self.vector_index = config
        return self

    def with_sparse_vector_index(self, config: "SparseVectorIndexConfig") -> "Schema":
        """
        配置稀疏向量索引（用于混合搜索）。

        Args:
            config: SparseVectorIndexConfig 配置对象

        Returns:
            self，支持链式调用
        """
        self.sparse_vector_index = config
        return self

    def with_fulltext_index(self, config: "FulltextIndexConfig") -> "Schema":
        """
        配置全文索引。

        Args:
            config: FulltextIndexConfig 配置对象

        Returns:
            self，支持链式调用
        """
        self.fulltext_index = config
        return self
```

#### 2.2.3 索引配置类

```python

@dataclass
class VectorIndexConfig:
    """
    稠密向量索引配置（HNSW 算法）

    Args:
        dimension: 向量维度（如果不指定，将从 embedding_function 推断）
        space: 距离度量方式，支持 'l2', 'cosine', 'inner_product'
        embedding_function: 嵌入函数，用于将文档转换为向量
        hnsw: HNSW 索引的高级配置参数

    Example:
        >>> config = VectorIndexConfig(
        ...     space="cosine",
        ...     embedding_function=OpenAIEmbeddingFunction(api_key="..."),
        ...     hnsw=HNSWConfiguration(ef_search=64)
        ... )
    """
    hnsw: HNSWConfiguration|None = None
    # ivf 未来可以用来支持IVF


@dataclass
class SparseVectorIndexConfig:
    """
    稀疏向量索引配置

    稀疏向量适用于基于关键词的检索，如 BM25、SPLADE 等。
    与稠密向量互补，可用于混合搜索（Hybrid Search）。

    Args:
        embedding_function: 稀疏嵌入函数（如 BM25EmbeddingFunction, SpladeEmbeddingFunction）
                           如果为 None，表示用户将直接提供稀疏向量
        source_key: 源字段键，指定从哪个字段生成稀疏向量。支持：
                   - K.DOCUMENT: 从 document 字段生成（默认）
                   - metadata 字段名: 如 "title", "summary" 等，必须是 str 类型。
                   - None: 用户直接提供稀疏向量，不自动生成

    Note:
        - 每个 Collection 只能有一个稀疏向量索引
        - 稀疏向量将存储在 sparse_embedding 列中
        - 如果 source_key 不为 None 且 embedding_function 为 None，将抛出错误

    Example:
        >>> # 从 document 字段自动生成
        >>> config = SparseVectorIndexConfig(
        ...     embedding_function=BM25EmbeddingFunction(),
        ...     source_key=K.DOCUMENT
        ... )
        >>>
        >>> # 从 metadata 的 title 字段生成
        >>> config = SparseVectorIndexConfig(
        ...     embedding_function=BM25EmbeddingFunction(),
        ...     source_key="title"
        ... )
        >>>
        >>> # 用户直接提供稀疏向量（不自动生成）
        >>> config = SparseVectorIndexConfig(
        ...     embedding_function=None,
        ...     source_key=None
        ... )
    """
    embedding_function: Optional["SparseEmbeddingFunction"] = None
    source_key: Optional[str] = "document"  # 默认从 document 字段生成，None 表示用户直接提供。或者从metadata指定的key中获取源数据


@dataclass
class FulltextIndexConfig:
    pass
```


### 2.3 稀疏嵌入函数接口

```python
# 稀疏向量类型定义
# seekdb/OceanBase 使用 dict[int, float] 格式
# key: 词汇/特征的索引位置
# value: 对应的权重值
@dataclass
class SparseVector:
    embeddings: dict[int, float] | None = None
    @staticmethod
    def from_dict(embeddings: dict[int, float]):
      pass
    @staticmethod
    def from_indices(indices: list[int], values: list[float]):
      pass

SparseVectors = list[SparseVector]

# 示例稀疏向量
# {100: 0.5, 200: 0.3, 500: 0.8} 表示在索引 100, 200, 500 位置分别有权重 0.5, 0.3, 0.8


@runtime_checkable
class SparseEmbeddingFunction(Protocol):
    """
    稀疏嵌入函数协议

    与 EmbeddingFunction 类似，但生成稀疏向量（dict[int, float]）而非稠密向量。
    稀疏向量适用于基于词汇的检索（如 BM25、SPLADE）。

    实现要求：
    - __call__(): 将文档转换为稀疏向量
    - name(): 静态方法，返回唯一名称标识符（用于注册和路由）
    - get_config(): 返回配置字典（用于持久化）
    - build_from_config(): 静态方法，从配置恢复实例

    Example:
        >>> class BM25EmbeddingFunction(SparseEmbeddingFunction):
        ...     def __call__(self, documents: Documents) -> SparseVectors:
        ...         # 生成 BM25 稀疏向量
        ...         # 返回 list[dict[int, float]]
        ...         ...
        ...
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "bm25"
        ...
        ...     def get_config(self) -> dict:
        ...         return {"k1": self.k1, "b": self.b}
        ...
        ...     @staticmethod
        ...     def build_from_config(config) -> "BM25EmbeddingFunction":
        ...         return BM25EmbeddingFunction(**config)
    """

    def __call__(self, documents: "Documents") -> SparseVectors:
        """
        将文档转换为稀疏向量

        Args:
            documents: 文档内容（str 或 list[str]）

        Returns:
            稀疏向量列表，每个稀疏向量是 dict[int, float]
            - key: 词汇/特征的索引位置（通常是 hash 值或词表索引）
            - value: 对应的权重值（如 BM25 分数、SPLADE 激活值）
        """
        ...

    @staticmethod
    def name() -> str:
        """返回唯一名称标识符（用于注册和路由）"""
        ...

    def get_config(self) -> dict[str, Any]:
        """
        获取配置字典（用于持久化）

        Returns:
            配置字典，不包含 'name' 字段（由上层处理）
        """
        ...

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "SparseEmbeddingFunction":
        """从配置恢复实例"""
        ...

    @staticmethod
    def support_persistence(sparse_embedding_function: Any) -> bool:
        """检查稀疏嵌入函数是否支持持久化"""
        pass
```

### 2.4 稀疏嵌入函数注册表
为了支持自定义EmbeddingFunction，与稠密emebddding function类似，增加一个registry。

```python
class SparseEmbeddingFunctionRegistry:
    """
    稀疏嵌入函数注册表

    与 EmbeddingFunctionRegistry 类似，用于管理稀疏嵌入函数的注册和恢复。
    支持动态注册自定义稀疏嵌入函数，实现配置持久化和自动恢复。

    内置稀疏嵌入函数：
    - bm25: BM25EmbeddingFunction
    - splade: SpladeEmbeddingFunction (需要 transformers 依赖)

    Example:
        >>> # 方式1: 使用装饰器注册
        >>> @register_sparse_embedding_function
        ... class MyCustomSparseEmbeddingFunction:
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_custom_sparse"
        ...     # ... 其他方法实现

        >>> # 方式2: 手动注册
        >>> SparseEmbeddingFunctionRegistry.register(MyCustomSparseEmbeddingFunction)

        >>> # 使用自定义稀疏嵌入函数
        >>> ef = MyCustomSparseEmbeddingFunction()
        >>> schema = Schema(
        ...     sparse_vector_index=SparseVectorIndexConfig(embedding_function=ef)
        ... )
        >>> collection = client.create_collection("my_collection", schema=schema)

        >>> # 重新获取 collection 时，稀疏嵌入函数会自动恢复
        >>> collection2 = client.get_collection("my_collection")
    """


    @classmethod
    def register(cls, sparse_embedding_function_class: type) -> None:
        """
        注册稀疏嵌入函数类

        Args:
            sparse_embedding_function_class: 要注册的类，必须实现：
                - name(): 静态方法，返回唯一名称标识符
                - get_config(): 返回配置字典
                - build_from_config(): 静态方法，从配置恢复实例

        Raises:
            ValueError: 如果类缺少必需的方法，或名称已被注册
        """
        pass

    @classmethod
    def get_class(cls, name: str) -> type | None:
        """根据名称获取稀疏嵌入函数类"""
        pass

    @classmethod
    def list_registered(cls) -> list[str]:
        """列出所有已注册的稀疏嵌入函数名称"""
        pass

    @classmethod
    def build_from_config(cls, name: str, config: dict[str, Any]) -> "SparseEmbeddingFunction":
        """
        从配置恢复稀疏嵌入函数实例

        Args:
            name: 稀疏嵌入函数名称
            config: 配置字典

        Returns:
            恢复的稀疏嵌入函数实例

        Raises:
            ValueError: 如果名称未注册
        """
        pass

T = TypeVar("T", bound=type)


def register_sparse_embedding_function(sparse_embedding_function_class: type[T]) -> type[T]:
    """
    装饰器：自动注册稀疏嵌入函数类

    Example:
        >>> @register_sparse_embedding_function
        ... class MyCustomSparseEmbeddingFunction:
        ...     def __init__(self, vocab_size: int = 30000):
        ...         self.vocab_size = vocab_size
        ...
        ...     def __call__(self, documents: Documents) -> SparseVectors:
        ...         # 自定义稀疏向量生成逻辑
        ...         ...
        ...
        ...     @staticmethod
        ...     def name() -> str:
        ...         return "my_custom_sparse"
        ...
        ...     def get_config(self) -> dict:
        ...         return {"vocab_size": self.vocab_size}
        ...
        ...     @staticmethod
        ...     def build_from_config(config: dict) -> "MyCustomSparseEmbeddingFunction":
        ...         return MyCustomSparseEmbeddingFunction(**config)
    """
    pass
```

## 3. API 设计

### 3.1 Collection 创建 API

```python
def create_collection(
    self,
    name: str,
    schema: Schema | None = None,
    embedding_function: EmbeddingFunction | None = DefaultEmbeddingFunction(),
    configuration: Configuration | HNSWConfiguration | None = None,  # 向后兼容
) -> Collection:
    """
    创建 Collection

    Args:
        name: Collection 名称
        schema: Schema 配置对象，用于精细控制索引
        embedding_function: 嵌入函数。如果schema中指定了embedding_function，将忽略此参数。
        configuration: 传统配置方式（向后兼容）。如果提供schema，将忽略此参数
    """
    ...
```

### 3.1.1 Collection.add() API 变更

```python
def add(
    self,
    ids: str | list[str],
    embeddings: list[float] | list[list[float]] | None = None,
    metadatas: dict | list[dict] | None = None,
    documents: str | list[str] | None = None,
    sparse_embeddings: dict[int, float] | list[dict[int, float]] | None = None,  # 新增
    **kwargs,
) -> None:
    """
    添加数据到 Collection

    Args:
        ids: ID 或 ID 列表
        embeddings: 稠密向量（可选，如果提供 documents 且有 embedding_function 则自动生成）
        metadatas: 元数据字典或列表
        documents: 文档内容
        sparse_embeddings: 稀疏向量（可选）
                          - 如果 Schema 配置了 source_key，将自动从对应字段生成
                          - 如果 source_key 为 None，需要用户直接提供

    Note:
        稀疏向量生成优先级：
        1. 用户直接提供 sparse_embeddings 参数
        2. 从 source_key 指定的字段自动生成。如果从source_key得到的数据不是str类型（包括没有指定的key），将抛出 ValueError异常。
        3. 如果都没有，则不存储稀疏向量
    """
    ...
```

`Collection.upsert` 有类似的变更。

### 3.1.2 Collection.sparse_query() - 稀疏向量检索接口（新增）

```python
def sparse_query(
    self,
    query_sparse_embeddings: dict[int, float] | list[dict[int, float]] | None = None,
    query_texts: str | list[str] | None = None,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    include: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    稀疏向量相似度查询

    使用稀疏向量（如 BM25、SPLADE 生成的向量）进行关键词级别的检索。
    稀疏向量搜索使用 inner_product 距离函数，分数越高越相似。

    Args:
        query_sparse_embeddings: 稀疏查询向量（单个或多个），格式为 dict[int, float]
                                 - key: 词汇/特征的索引位置
                                 - value: 对应的权重值
        query_texts: 查询文本（单个或多个），将使用 Schema 中配置的
                    sparse_embedding_function 自动转换为稀疏向量
        n_results: 返回结果数量（默认: 10）
        where: metadata 过滤条件，支持:
               - 比较操作: $eq, $lt, $gt, $lte, $gte, $ne, $in, $nin
               - 逻辑操作: $or, $and, $not
        where_document: document 过滤条件，支持:
               - $contains: 全文搜索
               - $regex: 正则表达式匹配
        include: 返回字段列表，如 ["documents", "metadatas", "embeddings", "sparse_embeddings"]

    Returns:
        Dict with keys (与 query() 格式一致):
        - ids: List[List[str]] - ID 列表
        - documents: Optional[List[List[str]]] - 文档列表（如果 included）
        - metadatas: Optional[List[List[Dict]]] - 元数据列表（如果 included）
        - sparse_embeddings: Optional[List[List[Dict[int, float]]]] - 稀疏向量（如果 included）
        - distances: List[List[float]] - 距离/分数列表（inner_product 分数，越高越相似）

    Raises:
        ValueError: 如果 Collection 没有配置稀疏向量索引
        ValueError: 如果同时未提供 query_sparse_embeddings 和 query_texts
        ValueError: 如果提供 query_texts 但 Collection 没有配置 sparse_embedding_function

    Example:
        >>> # 使用查询文本（自动转换为稀疏向量）
        >>> results = collection.sparse_query(
        ...     query_texts=["machine learning algorithms"],
        ...     n_results=5
        ... )
        >>> # results["ids"][0] 包含第一个查询的结果 ID
        >>> # results["distances"][0] 包含 inner_product 分数（越高越相似）

        >>> # 使用多个查询文本
        >>> results = collection.sparse_query(
        ...     query_texts=["machine learning", "deep neural networks"],
        ...     n_results=5
        ... )
        >>> # results["ids"][0] 包含第一个查询的结果
        >>> # results["ids"][1] 包含第二个查询的结果

        >>> # 直接提供稀疏向量
        >>> results = collection.sparse_query(
        ...     query_sparse_embeddings=[{100: 0.5, 200: 0.3, 500: 0.8}],
        ...     n_results=5
        ... )

        >>> # 带过滤条件的稀疏向量搜索
        >>> results = collection.sparse_query(
        ...     query_texts=["AI research"],
        ...     where={"category": {"$eq": "technology"}},
        ...     n_results=10,
        ...     include=["documents", "metadatas"]
        ... )
    """
    ...
```

### 3.1.3 query() 接口保持不变

现有的 `query()` 接口保持不变，专门用于稠密向量检索：

```python
def query(
    self,
    query_embeddings: list[float] | list[list[float]] | None = None,
    query_texts: str | list[str] | None = None,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    include: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    稠密向量相似度查询（现有接口，保持不变）

    使用稠密向量进行语义级别的检索。
    """
    ...
```

### 3.1.4 hybrid_search() 接口（保持不变）

`hybrid_search()` 支持全文搜索 + 稠密向量的混合搜索。

在接口中增加说明，当前混搜暂时不支持稀疏索引检索，比如：

> **注意**：`hybrid_search()` 目前**不支持稀疏向量索引**。稀疏向量检索请使用 `sparse_query()` 接口。
> 如需同时使用稠密向量和稀疏向量进行检索，可以分别调用 `query()` 和 `sparse_query()`，
> 然后在应用层进行结果融合（如 RRF）。


### 3.2 使用示例

#### 3.2.1 基本用法

```python
import pyseekdb
from pyseekdb import Schema, VectorIndexConfig
from pyseekdb.utils.embedding_functions import OpenAIEmbeddingFunction

client = pyseekdb.Client(path="./seekdb.db", database="test")

# 创建 Schema 并配置稠密向量索引
schema = Schema(
    vector_index=VectorIndexConfig(
        space="cosine",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        )
    )
)

# 创建 Collection
collection = client.create_collection(
    name="my_collection",
    schema=schema
)
```

#### 3.2.2 配置稀疏向量索引

```python
from pyseekdb import Schema, VectorIndexConfig, SparseVectorIndexConfig, K
from pyseekdb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    BM25EmbeddingFunction,  # 新增
)

# 创建 Schema，同时配置稠密和稀疏向量索引
schema = Schema(
    # 稠密向量索引（语义搜索）
    vector_index=VectorIndexConfig(
        space="cosine",
        embedding_function=OpenAIEmbeddingFunction(api_key="...")
    ),
    # 稀疏向量索引（关键词搜索）
    sparse_vector_index=SparseVectorIndexConfig(
        embedding_function=BM25EmbeddingFunction(),
        source_key=K.DOCUMENT  # 从 document 字段生成稀疏向量
    )
)

# 创建 Collection
collection = client.create_collection(
    name="hybrid_collection",
    schema=schema
)

# 添加数据（稀疏向量自动从 document 生成）
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps over a sleepy canine",
        "Machine learning is a subset of artificial intelligence"
    ],
    metadatas=[
        {"category": "animals"},
        {"category": "animals"},
        {"category": "technology"}
    ]
)
```

#### 3.2.3 稀疏向量检索（使用 sparse_query）

```python
# 方式1: 使用查询文本（自动转换为稀疏向量）
results = collection.sparse_query(
    query_texts=["fox animal"],
    n_results=5,
    include=["documents", "metadatas"]
)
print(f"Found {len(results['ids'][0])} results")
# results["distances"] 是 inner_product 分数，越高越相似

# 方式2: 直接提供稀疏向量
results = collection.sparse_query(
    query_sparse_embeddings=[{100: 0.5, 200: 0.3, 500: 0.8}],
    n_results=5
)

# 方式3: 带过滤条件的稀疏向量搜索
results = collection.sparse_query(
    query_texts=["machine learning"],
    where={"category": {"$eq": "technology"}},
    n_results=10
)

# 方式4: 多查询批量搜索
results = collection.sparse_query(
    query_texts=["fox animal", "machine learning", "deep neural networks"],
    n_results=5
)
# results["ids"][0] - 第一个查询的结果
# results["ids"][1] - 第二个查询的结果
# results["ids"][2] - 第三个查询的结果
```

#### 3.2.4 混合搜索（全文 + 稠密向量 RRF 融合）

```python
# 混合搜索：全文 + 稠密向量
# 注意：hybrid_search 不支持稀疏向量索引
results = collection.hybrid_search(
    query={
        "where_document": {"$contains": "fox"},
        "n_results": 20
    },
    knn={
        "query_texts": ["fox animal"],
        "n_results": 20
    },
    rank={
        "rrf": {
            "rank_window_size": 60,
            "rank_constant": 60
        }
    },
    n_results=10,
    include=["documents", "metadatas"]
)
```

#### 3.2.7 从 Metadata 字段生成稀疏向量

```python
from pyseekdb import Schema, SparseVectorIndexConfig
from pyseekdb.utils.embedding_functions import BM25EmbeddingFunction

# 从 metadata 的 title 字段生成稀疏向量（而不是 document）
schema = Schema(
    sparse_vector_index=SparseVectorIndexConfig(
        embedding_function=BM25EmbeddingFunction(),
        source_key="title"  # 从 metadata["title"] 生成
    )
)

collection = client.create_collection(
    name="title_search_collection",
    schema=schema
)

# 添加数据时，稀疏向量从 metadata["title"] 自动生成
collection.add(
    ids=["doc1", "doc2"],
    documents=["Full document content here...", "Another document..."],
    metadatas=[
        {"title": "Introduction to Machine Learning", "category": "tech"},
        {"title": "Deep Learning Fundamentals", "category": "tech"}
    ]
)
```

#### 3.2.8 用户直接提供稀疏向量

```python
from pyseekdb import Schema, SparseVectorIndexConfig

# 配置稀疏向量索引，但不自动生成（用户自己提供）
schema = Schema(
    sparse_vector_index=SparseVectorIndexConfig(
        embedding_function=None,  # 不使用 embedding_function
        source_key=None           # 不从任何字段自动生成
    )
)

collection = client.create_collection(
    name="custom_sparse_collection",
    schema=schema
)

# 用户自己计算稀疏向量后传入
# 稀疏向量格式: dict[int, float]，key 是索引位置，value 是权重
my_sparse_vectors = [
    {100: 0.5, 200: 0.3, 500: 0.8},  # 第一个文档的稀疏向量
    {150: 0.4, 300: 0.6, 600: 0.2}   # 第二个文档的稀疏向量
]

collection.add(
    ids=["doc1", "doc2"],
    documents=["Document 1", "Document 2"],
    sparse_embeddings=my_sparse_vectors  # 直接提供稀疏向量
)
```

#### 3.2.9 自定义稀疏嵌入函数

```python
from pyseekdb import register_sparse_embedding_function
from pyseekdb.client.sparse_embedding_function import SparseEmbeddingFunction, SparseVectors
from typing import Any
from collections import Counter
import hashlib

@register_sparse_embedding_function
class TFIDFSparseEmbeddingFunction:
    """
    基于 TF-IDF 的自定义稀疏嵌入函数示例
    """

    def __init__(self, max_features: int = 10000, normalize: bool = True):
        self.max_features = max_features
        self.normalize = normalize

    def __call__(self, documents: str | list[str]) -> SparseVectors:
        if isinstance(documents, str):
            documents = [documents]

        sparse_vectors: list[dict[int, float]] = []
        for doc in documents:
            tokens = doc.lower().split()
            term_freqs = Counter(tokens)

            sparse_vector: dict[int, float] = {}
            for term, freq in term_freqs.items():
                # 使用 hash 将词映射到固定范围的索引
                idx = int(hashlib.md5(term.encode()).hexdigest(), 16) % self.max_features
                # TF 权重（可以扩展为 TF-IDF）
                weight = 1 + math.log(freq) if freq > 0 else 0
                sparse_vector[idx] = weight

            # 可选：归一化
            if self.normalize and sparse_vector:
                norm = math.sqrt(sum(v ** 2 for v in sparse_vector.values()))
                sparse_vector = {k: v / norm for k, v in sparse_vector.items()}

            sparse_vectors.append(sparse_vector)

        return sparse_vectors

    @staticmethod
    def name() -> str:
        return "tfidf_sparse"

    def get_config(self) -> dict[str, Any]:
        return {
            "max_features": self.max_features,
            "normalize": self.normalize
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "TFIDFSparseEmbeddingFunction":
        return TFIDFSparseEmbeddingFunction(**config)


# 使用自定义稀疏嵌入函数
schema = Schema(
    sparse_vector_index=SparseVectorIndexConfig(
        embedding_function=TFIDFSparseEmbeddingFunction(max_features=50000),
        source_key=K.DOCUMENT
    )
)

collection = client.create_collection("tfidf_collection", schema=schema)

# 添加数据
collection.add(
    ids=["doc1"],
    documents=["Custom TF-IDF sparse embedding example"]
)

# 重新获取 collection 时，稀疏嵌入函数会自动恢复
collection2 = client.get_collection("tfidf_collection")
# collection2 的 sparse embedding function 会自动从 Registry 恢复
```

## 4. 实现细节

### 4.1 Schema 与 OceanBase SQL 映射

#### 4.1.1 当前表结构

```sql
CREATE TABLE collection_xxx (
    _id VARBINARY(512) PRIMARY KEY,
    document TEXT,
    embedding VECTOR(384),
    metadata JSON,
    FULLTEXT INDEX idx_fts(document) WITH PARSER ik,
    VECTOR INDEX idx_vec(embedding) WITH (DISTANCE=cosine, TYPE=hnsw, LIB=vsag)
);
```

#### 4.1.2 增加稀疏向量后的表结构

```sql
CREATE TABLE collection_xxx (
    _id VARBINARY(512) PRIMARY KEY,
    document TEXT,
    embedding VECTOR(384),
    sparse_embedding SPARSEVECTOR,  -- 新增稀疏向量列
    metadata JSON,
    FULLTEXT INDEX idx_fts(document) WITH PARSER ik,
    VECTOR INDEX idx_vec(embedding) WITH (DISTANCE=cosine, TYPE=hnsw, LIB=vsag),
    VECTOR INDEX idx_sparse(sparse_embedding) WITH (DISTANCE=inner_product, TYPE=hnsw, LIB=vsag)  -- 稀疏向量索引
);
```

> **OceanBase 稀疏向量说明**（参考 [seekdb 向量函数文档](https://www.oceanbase.ai/docs/vector-function/)）：
> - 稀疏向量类型：`SPARSEVECTOR`
> - 稀疏向量格式：`'{1:1.1, 2:2.2, 100:0.5}'`
> - 支持的距离函数：仅 `inner_product`（内积）
> - 内存稀疏向量索引支持 inner_product 作为距离算法

#### 4.1.3 稀疏向量数据插入

```sql
-- 插入稀疏向量数据
INSERT INTO collection_xxx (_id, document, embedding, sparse_embedding, metadata)
VALUES (
    'doc1',
    'Hello world',
    '[0.1, 0.2, 0.3, ...]',
    '{100:0.5, 200:0.3, 500:0.8}',  -- 稀疏向量格式
    '{"category": "test"}'
);
```

#### 4.1.4 稀疏向量相似度搜索 SQL

```sql
-- 稀疏向量近似搜索（使用 inner_product）
SELECT _id, document, metadata, inner_product(sparse_embedding, '{100:0.5, 200:0.3}') as score
FROM collection_xxx
ORDER BY inner_product(sparse_embedding, '{100:0.5, 200:0.3}') DESC
APPROXIMATE
LIMIT 10;

-- 稠密向量搜索
SELECT _id, document, metadata, l2_distance(embedding, '[0.1, 0.2, ...]') as distance
FROM collection_xxx
ORDER BY l2_distance(embedding, '[0.1, 0.2, ...]')
APPROXIMATE
LIMIT 10;
```

> **注意**：`APPROXIMATE` 关键字表示使用向量索引进行近似最近邻搜索（ANN），而不是全表扫描的精确搜索。

### 4.2 Schema 序列化与持久化

Schema 配置需要持久化到 `sdk_collections` 表的 `settings` 字段中：

```json
{
  "schema": {
    "vector_index": {
      "dimension": 1536,
      "space": "cosine",
      "embedding_function": {
        "name": "openai",
        "config": {
          "model_name": "text-embedding-3-small"
        }
      },
      "hnsw": {
        "ef_search": 64,
        "ef_construction": 200,
        "max_neighbors": 32
      }
    },
    "sparse_vector_index": {
      "source_key": "document",
      "embedding_function": {
        "name": "bm25",
        "config": {
          "k1": 1.2,
          "b": 0.75
        }
      }
    },
    "fulltext_index": {
      "analyzer": "ik",
      "properties": null
    }
  }
}
```

> **向后兼容**：如果 `settings` 中没有 `schema` 字段，则使用默认配置。现有的 `embedding_function` 字段将被迁移到 `schema.vector_index.embedding_function` 中。

### 4.3 稀疏向量生成逻辑

稀疏向量的生成有三种模式：

#### 模式 1: 从 document 字段自动生成（默认）

```python
# Schema 配置
sparse_vector_index=SparseVectorIndexConfig(
    embedding_function=BM25EmbeddingFunction(),
    source_key=K.DOCUMENT  # 或 "document"
)

# add() 时自动从 documents 生成稀疏向量
collection.add(ids=["1"], documents=["Hello world"])
```

#### 模式 2: 从 metadata 字段自动生成

```python
# Schema 配置
sparse_vector_index=SparseVectorIndexConfig(
    embedding_function=BM25EmbeddingFunction(),
    source_key="title"  # 从 metadata["title"] 生成
)

# add() 时自动从 metadata["title"] 生成稀疏向量
collection.add(
    ids=["1"],
    documents=["Full content..."],
    metadatas=[{"title": "My Title"}]
)
```

#### 模式 3: 用户直接提供

```python
# Schema 配置
sparse_vector_index=SparseVectorIndexConfig(
    embedding_function=None,
    source_key=None
)

# add() 时用户直接提供 sparse_embeddings 参数
# 格式: dict[int, float]，key 是索引，value 是权重
collection.add(
    ids=["1"],
    documents=["Hello world"],
    sparse_embeddings=[{100: 0.5, 200: 0.3}]
)
```

#### 实现逻辑

```python
def _collection_add(self, ..., documents, metadatas, sparse_embeddings=None, ...):
    # 获取 schema 中的稀疏向量配置
    sparse_config = self._get_sparse_vector_config(collection_id)

    if sparse_config is None:
        # Collection 没有配置稀疏向量索引，跳过
        final_sparse_embeddings = None
    elif sparse_embeddings is not None:
        # 用户直接提供了稀疏向量
        final_sparse_embeddings = sparse_embeddings
    elif sparse_config.source_key and sparse_config.embedding_function:
        # 从指定字段自动生成
        if sparse_config.source_key == K.DOCUMENT:
            source_texts = documents
        else:
            # 从 metadata 的指定字段获取
            source_texts = [m.get(sparse_config.source_key, "") for m in metadatas]
        final_sparse_embeddings = sparse_config.embedding_function(source_texts)
    else:
        # source_key 为 None 但用户没有提供 sparse_embeddings，报错或跳过
        final_sparse_embeddings = None

    # 将稀疏向量存储到 sparse_embedding 列
    ...
```

### 4.4 稀疏向量检索实现

#### 4.4.1 sparse_query() 方法实现

```python
def _collection_sparse_query(
    self,
    collection_id: str | None,
    collection_name: str,
    query_sparse_embeddings: list[dict[int, float]] | None = None,
    query_texts: list[str] | None = None,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    include: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    [Internal] 稀疏向量检索实现
    """
    # 获取 schema 配置
    schema = self._get_schema(collection_id)
    sparse_config = schema.sparse_vector_index if schema else None

    if sparse_config is None:
        raise ValueError(f"Collection '{collection_name}' does not have a sparse vector index configured")

    # 处理稀疏向量查询
    final_sparse_embeddings: list[dict[int, float]]
    if query_sparse_embeddings is not None:
        # 用户直接提供稀疏向量
        if isinstance(query_sparse_embeddings, dict):
            final_sparse_embeddings = [query_sparse_embeddings]
        else:
            final_sparse_embeddings = query_sparse_embeddings
    elif query_texts is not None:
        # 使用稀疏嵌入函数将文本转换为稀疏向量
        if sparse_config.embedding_function is None:
            raise ValueError(
                f"Collection '{collection_name}' has sparse vector index but no sparse_embedding_function configured. "
                "Please provide query_sparse_embeddings directly or configure a sparse_embedding_function in Schema."
            )
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        final_sparse_embeddings = sparse_config.embedding_function(query_texts)
    else:
        raise ValueError("Either query_sparse_embeddings or query_texts must be provided")

    # 执行稀疏向量搜索
    return self._execute_sparse_search(
        collection_id=collection_id,
        collection_name=collection_name,
        sparse_embeddings=final_sparse_embeddings,
        n_results=n_results,
        where=where,
        where_document=where_document,
        include=include,
        **kwargs
    )
```

#### 4.4.2 稀疏向量搜索 SQL 生成

```python
def _build_sparse_knn_sql(
    self,
    table_name: str,
    sparse_embedding: dict[int, float],
    n_results: int,
    where_clause: str | None = None,
) -> str:
    """
    构建稀疏向量搜索的 SQL 语句

    OceanBase 稀疏向量使用 inner_product 距离函数，分数越高越相似
    """
    # 将 Python dict 转换为 OceanBase 稀疏向量格式
    # {100: 0.5, 200: 0.3} -> '{100:0.5, 200:0.3}'
    sparse_str = "{" + ", ".join(f"{k}:{v}" for k, v in sparse_embedding.items()) + "}"

    sql = f"""
    SELECT _id, document, metadata,
           inner_product(sparse_embedding, '{sparse_str}') as score
    FROM {table_name}
    """

    if where_clause:
        sql += f" WHERE {where_clause}"

    # 稀疏向量使用 inner_product，分数越高越相似，所以 DESC 排序
    sql += f"""
    ORDER BY inner_product(sparse_embedding, '{sparse_str}') DESC
    APPROXIMATE
    LIMIT {n_results}
    """

    return sql
```

#### 4.4.3 _execute_sparse_search() 实现

```python
def _execute_sparse_search(
    self,
    collection_id: str | None,
    collection_name: str,
    sparse_embeddings: list[dict[int, float]],
    n_results: int,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    include: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    执行稀疏向量搜索
    """
    table_name = self._get_table_name(collection_id)

    # 构建 WHERE 子句
    where_clause = self._build_where_clause(where, where_document)

    # 多查询结果
    all_results: dict[str, Any] = {
        "ids": [],
        "distances": [],
    }
    if include:
        if "documents" in include:
            all_results["documents"] = []
        if "metadatas" in include:
            all_results["metadatas"] = []
        if "sparse_embeddings" in include:
            all_results["sparse_embeddings"] = []

    # 为每个查询稀疏向量执行搜索
    for sparse_emb in sparse_embeddings:
        sql = self._build_sparse_knn_sql(
            table_name=table_name,
            sparse_embedding=sparse_emb,
            n_results=n_results,
            where_clause=where_clause
        )

        rows = self._execute_sql(sql)

        # 处理结果
        ids = []
        distances = []
        documents = [] if include and "documents" in include else None
        metadatas = [] if include and "metadatas" in include else None
        sparse_embs = [] if include and "sparse_embeddings" in include else None

        for row in rows:
            ids.append(row["_id"])
            distances.append(row["score"])
            if documents is not None:
                documents.append(row.get("document"))
            if metadatas is not None:
                metadatas.append(row.get("metadata"))
            if sparse_embs is not None:
                sparse_embs.append(oceanbase_to_sparse_dict(row.get("sparse_embedding")))

        all_results["ids"].append(ids)
        all_results["distances"].append(distances)
        if documents is not None:
            all_results["documents"].append(documents)
        if metadatas is not None:
            all_results["metadatas"].append(metadatas)
        if sparse_embs is not None:
            all_results["sparse_embeddings"].append(sparse_embs)

    return all_results
```

#### 4.4.4 hybrid_search() 说明

> **注意**：`hybrid_search()` 接口**不支持稀疏向量索引**。
>
> OceanBase 的 `DBMS_HYBRID_SEARCH.GET_SQL` 目前仅支持全文搜索和稠密向量（VECTOR 类型）的 knn 搜索，
> 不支持稀疏向量（SPARSEVECTOR 类型）的 knn 搜索。
>
> 如需使用稀疏向量进行检索，请使用 `sparse_query()` 接口。
> 如需同时使用稠密向量和稀疏向量进行检索，可以分别调用 `query()` 和 `sparse_query()`，
> 然后在应用层进行结果融合（如 RRF 算法）。

`hybrid_search()` 的实现保持不变，仅支持全文搜索 + 稠密向量搜索的 RRF 融合。
详细实现请参考现有 `client_base.py` 中的 `_collection_hybrid_search` 方法。

### 4.5 稀疏向量格式转换

Python `dict[int, float]` 与 OceanBase `SPARSEVECTOR` 格式之间的转换：

```python
def sparse_dict_to_oceanbase(sparse: dict[int, float]) -> str:
    """
    Python dict -> OceanBase SPARSEVECTOR 字符串格式

    {100: 0.5, 200: 0.3} -> '{100:0.5, 200:0.3}'
    """
    if not sparse:
        return '{}'
    return "{" + ", ".join(f"{k}:{v}" for k, v in sparse.items()) + "}"


def oceanbase_to_sparse_dict(sparse_str: str) -> dict[int, float]:
    """
    OceanBase SPARSEVECTOR 字符串格式 -> Python dict

    '{100:0.5, 200:0.3}' -> {100: 0.5, 200: 0.3}
    """
    if not sparse_str or sparse_str == '{}':
        return {}
    # 移除花括号
    content = sparse_str.strip('{}')
    if not content:
        return {}
    # 解析键值对
    result = {}
    for pair in content.split(','):
        pair = pair.strip()
        if ':' in pair:
            key, value = pair.split(':', 1)
            result[int(key.strip())] = float(value.strip())
    return result
```

## 5. 稀疏嵌入函数实现示例

### 5.1 BM25 嵌入函数

```python
from collections import Counter
from typing import Callable

class BM25EmbeddingFunction(SparseEmbeddingFunction):
    """
    BM25 稀疏嵌入函数

    使用 BM25 算法将文档转换为稀疏向量，适用于关键词检索。

    Args:
        k1: 词频饱和参数 (default: 1.2)
        b: 文档长度归一化参数 (default: 0.75)
        tokenizer: 分词器，默认使用空格分词
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or str.split

    def __call__(self, documents: Documents) -> SparseVectors:
        """
        将文档转换为稀疏向量

        Returns:
            list[dict[int, float]]: 每个文档对应一个稀疏向量
        """
        if isinstance(documents, str):
            documents = [documents]

        sparse_vectors: list[dict[int, float]] = []
        for doc in documents:
            tokens = self.tokenizer(doc)
            # 计算词频
            term_freqs = Counter(tokens)
            # 转换为稀疏向量 (dict[int, float])
            sparse_vector: dict[int, float] = {}
            for term, freq in term_freqs.items():
                # 使用 hash 将词映射到索引（取正数）
                idx = hash(term) % (2**31)
                # BM25 词权重计算
                score = (freq * (self.k1 + 1)) / (freq + self.k1)
                sparse_vector[idx] = score
            sparse_vectors.append(sparse_vector)

        return sparse_vectors

    @staticmethod
    def name() -> str:
        return "bm25"

    def get_config(self) -> dict:
        return {"k1": self.k1, "b": self.b}

    @staticmethod
    def build_from_config(config: dict) -> "BM25EmbeddingFunction":
        return BM25EmbeddingFunction(**config)
```

### 5.2 SPLADE 嵌入函数

```python
import torch

class SpladeEmbeddingFunction(SparseEmbeddingFunction):
    """
    SPLADE 稀疏嵌入函数

    使用 SPLADE 模型生成稀疏向量，比 BM25 具有更好的语义理解能力。

    Args:
        model_name: SPLADE 模型名称
        device: 运行设备 ('cpu', 'cuda')
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def __call__(self, documents: Documents) -> SparseVectors:
        """
        将文档转换为稀疏向量

        Returns:
            list[dict[int, float]]: 每个文档对应一个稀疏向量
        """
        self._load_model()

        if isinstance(documents, str):
            documents = [documents]

        sparse_vectors: list[dict[int, float]] = []
        for doc in documents:
            # SPLADE 编码
            inputs = self._tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # SPLADE 激活: log(1 + ReLU(x))
                logits = outputs.logits
                weights = torch.max(torch.log1p(torch.relu(logits)), dim=1)[0].squeeze()

            # 提取非零元素，转换为 dict[int, float]
            sparse_vector: dict[int, float] = {}
            non_zero_indices = (weights > 0).nonzero().squeeze(-1)
            for idx in non_zero_indices.tolist():
                sparse_vector[idx] = float(weights[idx])

            sparse_vectors.append(sparse_vector)

        return sparse_vectors

    @staticmethod
    def name() -> str:
        return "splade"

    def get_config(self) -> dict:
        return {"model_name": self.model_name, "device": self.device}

    @staticmethod
    def build_from_config(config: dict) -> "SpladeEmbeddingFunction":
        return SpladeEmbeddingFunction(**config)
```
