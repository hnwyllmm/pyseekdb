# 多模态支持设计文档

## 1. 背景与动机

### 1.1 背景
随着多模态大模型的发展，向量数据库不仅需要支持文本数据，还需要支持图片、音频等多模态数据。对于多媒体数据和文本数据混合的信息，使用同一个模型生成的向量，在检索时，效果会更好。比如对于一个带图片的文档，可以根据图片搜索到关联的文档，也可以根据文本信息搜到相关的图片。ChromaDB 已经率先支持了多模态数据（主要是图片），允许用户直接存储和查询图片。
对于开发者来说，用户可以自己通过多模态embedding function自己生成向量数据，然后插入到数据库中。那么对于客户端(pyseekdb)来说，我们可以为用户提供更加方便的embedding接口和向量数据更新接口。

### 1.2 目标
在 pyseekdb 中实现类似 ChromaDB 的多模态支持，当前只有支持**图片**模态。使得用户可以：
1.  将图片（图片内容或URI）转换为向量数据存储到数据库中。
2.  使用文本或图片进行跨模态检索（以文搜图、以图搜图）。

> 对于其它的多媒体类型，当前暂时不支持。首先当前已知的模型支持像音频、视频的能力比较弱（支持的大小特别小）。其次音视频消耗的token比较大，普通用户很难承受这么高昂的费用。最后真正对有这些需求的用户，可以通过自己生成向量数据的方式，实现自己的需求。

### 1.3 参考实现
本设计参考了 [Chroma](https://docs.trychroma.com/docs/embeddings/multimodal) 的多模态实现，保持 API 接口的高度兼容性。

## 2. 核心概念与组件

### 2.1 新增数据类型
在 `src/pyseekdb/client/types.py` 中定义以下类型：

*   **ImageData**: 图片数据，通常表示为 `numpy.ndarray`，数组的元素可以是 uint8, int64 或者 float64。
*   **ImageURI**: 一个继承自 `str` 的包装类，用于明确标识该字符串是图片路径而非文本内容。
    ```python
    class ImageURI(str):
        pass
    ```
*   **Image**: `Union[ImageData, ImageURI]`。
*   **DataLoader**: 数据加载器协议，负责从 URI 加载原始数据。

### 2.3 EmbeddingFunction
*   **职责**: 提供多模态 Embedding 能力。
*   **接口变更**: `__call__` 方法接受 `inputs` 参数，类型为 `list[str | Image]`。
    *   支持混合列表：`["text", np.array(...), ImageURI("/path/to/img")]`。
*   **实现策略**:
    *   对于支持直接传入 URI 的远程模型（如 Jina AI），EF 可以直接将 URI 发送给服务端，避免本地下载和传输带来的带宽消耗。
    *   对于只支持像素输入的本地模型（如 OpenCLIP），EF 需要先加载 URI 指向的图片数据（可使用 `data_loader` 或内部实现），再进行推理。
*   **OpenCLIP实现**: `src/pyseekdb/utils/embedding_functions/open_clip_embedding_function.py` 将作为参考实现。

## 3. API 设计与变更

### 3.1 Collection 接口变更
`src/pyseekdb/client/collection.py` 中的 `Collection` 类将进行以下增强：

#### 3.1.1 初始化 (`__init__`)
新增 `data_loader` 参数，允许用户自定义数据加载逻辑（默认为 `ImageLoader` 或 None）。

```python
def __init__(self, ..., data_loader: DataLoader | None = None, ...): ...
```

为了增加这个参数，在 `create_collection` 和 `get_collection` 中必要时都要传入 `data_loader` 参数。当然，绝大部分情况下都不需要传入此参数，因为我们默认的 DataLoader 可以支持 HTTP和本地文件的图片加载。

#### 3.1.2 数据添加 (`add`, `upsert`, `update`)
引入统一的 `inputs` 参数，同时保留 `documents` 以保持向后兼容（仅支持文本）。

```python
def add(
    self,
    ids: str | list[str],
    embeddings: list[float] | list[list[float]] | None = None,
    metadatas: dict | list[dict] | None = None,
    # 统一输入：支持文本、图片数据、图片URI的混合列表
    inputs: list[str | Image | ImageURI] | None = None,
    # 兼容参数。如果用户同时传入inputs和documents，会抛出异常。
    documents: str | list[str] | None = None,
    **kwargs,
) -> None: ...
```

**参数处理逻辑**:
1.  **参数互斥检查**: `inputs`, `documents` 二者在逻辑上应尽量只使用其一。如果同时提供 `inputs` 和 `documents`，将抛出异常以避免歧义。
2.  **`inputs` 解析**:
    遍历 `inputs` 列表，根据元素类型分发：
    *   `str`: 视为文本 -> 存入 `document` 列。
    *   `ImageData` (numpy): 视为图片 -> `uri` 列存 NULL。
    *   `ImageURI` (str包装类): 视为图片 -> `uri` 列存该路径。
3.  **向量生成**:
    *   如果有 `embeddings`，直接使用。
    *   否则，将 `inputs` 列表整体传给 `embedding_function(inputs)`。EF 需要能够处理这种混合列表。

**场景示例 (混合文档)**:
```python
# 解析结果：段落1(文本)，图片1(数据)，段落2(文本)
data = [
    "Introduction to cats...",
    numpy_image_array,
    "Cats represent..."
]
ids = ["id1", "id2", "id3"]
collection.add(ids=ids, inputs=data)
```
数据库中：
*   Row 1: document="Introduction...", uri=NULL, embedding=Vec(Text)
*   Row 2: document=NULL, uri=NULL, embedding=Vec(Image)
*   Row 3: document="Cats...", uri=NULL, embedding=Vec(Text)



#### 3.1.3 数据查询 (`query`)
新增 `query_inputs` 参数用于检索，支持文本、`ImageData` 或 `ImageURI`。
新增 `include` 字段支持 `uris`。

```python
def query(
    self,
    query_embeddings: list[float] | list[list[float]] | None = None,
    query_texts: str | list[str] | None = None,
    query_inputs: list[str | Image | ImageURI] | None = None,  # 支持文本、ImageData 或 ImageURI
    n_results: int = 10,
    include: list[str] | None = None,
    ...
) -> dict[str, Any]: ...
```

**逻辑流**:
如果 `include` 包含 `data` 或 `images`，且结果中包含 `uris`，则在返回前调用 `data_loader(uris)` 加载图片数据填充到 `data` 字段。

#### 3.1.4 数据获取 (`get`)
`include` 参数支持 `uris`。

### 3.2 数据库 Schema 变更
底层存储表结构需要新增 `uri` 列用于存储资源路径。

**方案一**
```sql
CREATE TABLE `{table_name}` (
    _id varbinary(512) PRIMARY KEY NOT NULL,
    document string,
    uri string,          -- 新增列
    embedding vector({dimension}),
    metadata json,
    ...
)
```
此方案扩展性稍差，之前创建的表无法直接新增字段，仅能支持新创建的表。
**新旧 Collection**:
    *   V2 Collection (新创建) 将包含 `uri` 列。
    *   V1 Collection (旧) 或 现有 V2 表没有 `uri` 列。
    *   **处理策略**: 在 `add` 或 `query` 中涉及 `uri` 操作时，如果底层表不存在 `uri` 列，数据库会报错。建议新功能仅对新 Collection 或执行过 Schema 迁移的 Collection 生效。
    *   SDK 层面暂不自动执行 `ALTER TABLE ADD COLUMN uri`，依赖新建 Collection 生效。

**方案二**
在metadata中，增加键值 `seekdb:uri` 存储image的URI。
这个方案扩展性强，以后支持音频视频或PDF，都可以使用，比如支持视频 `seekdb:vedio_uri`。

参考 chroma 的做法，在metadata中使用 `chroma:document` 和 `chroma:uri` 来记录特殊信息。

## 4. 限制
*   当前仅支持图片模态。
*   `uri` 列主要用于引用外部存储，数据库本身不直接存储图片二进制（BLOB），保持轻量级。
*   数据一致性依赖用户保证 `uri` 指向的资源有效。

## 5. SDK 中提供的附带模型支持

参考 Chroma 的实现，我们会支持以下几个平台：
OpenCLIPEmbeddingFunction
RoboflowEmbeddingFunction
CohereEmbeddingFunction
JinaEmbeddingFunction

另外支持国内平台[通义千问平台](https://bailian.console.aliyun.com/cn-beijing/?spm=5176.29619931.J_4NWEMkQ5nDwOgLi8EJmHs.8.74cd10d7kx7kvU&tab=doc#/doc/?type=model&url=2842587)。
