# SeekDBClient

SeekDBClient is a unified Python client that wraps three database connection modes—embedded SeekDB, remote SeekDB servers, and OceanBase—behind a single, concise API. 

## Installation
```bash
cd .../pyobvector
poetry install
```

## Quick Start
### Embedded SeekDB
```python
import seekdbclient

client = seekdbclient.Client(path="./seekdb", database="demo")
rows = client.execute("SELECT 1")
print(rows)
```

### Remote SeekDB Server
```python
import seekdbclient

with seekdbclient.Client(
    host="127.0.0.1",
    port=2882,
    database="demo",
    user="root",
    password=""
) as client:
    print(client.execute("SHOW TABLES"))
```

### OceanBase
```python
import seekdbclient

with seekdbclient.OBClient(
    host="127.0.0.1",
    port=11402,
    tenant="mysql",
    database="test",
    user="root",
    password=""
) as client:
    version = client.execute("SELECT version() AS v")
    print(version[0]["v"])
```

## API Overview
### Factory Functions
```python
seekdbclient.Client(path="/data/seekdb", database="demo")        # SeekdbEmbeddedClient
seekdbclient.Client(host="localhost", port=2882, database="demo") # SeekdbServerClient
seekdbclient.OBClient(host="localhost", tenant="mysql")           # OceanBaseServerClient
```

### Client Methods
| Method / Property     | Description                                                    |
|-----------------------|----------------------------------------------------------------|
| `execute(sql)`        | Run SQL and return cursor results (commits automatically when needed). |
| `is_connected()`      | Check whether an underlying connection is active.             |
| `get_raw_connection()`| Access the underlying seekdb / pymysql connection.            |
| `mode`                | Returns the concrete client class name (`SeekdbEmbeddedClient`, `SeekdbServerClient`, or `OceanBaseServerClient`). |
| `_ensure_connection()`| Internal lazy connector (not part of public API).             |
| `_cleanup()`          | Internal cleanup hook; called by `__exit__` / `__del__`.       |

### AdminClient for Database Management
```python
import seekdbclient

# Server mode - Database management
admin = seekdbclient.AdminClient(host="127.0.0.1", port=2881, user="root")
admin.create_database("my_database")
databases = admin.list_databases()
admin.delete_database("my_database")

# OceanBase mode - Database management (multi-tenant)
admin = seekdbclient.OBAdminClient(host="127.0.0.1", port=11402, tenant="mysql")
admin.create_database("analytics")
databases = admin.list_databases()  # Scoped to tenant
```

**AdminClient Methods:**
| Method                    | Description                                        |
|---------------------------|----------------------------------------------------|
| `create_database(name)`   | Create a new database                              |
| `get_database(name)`      | Get database object with metadata                  |
| `delete_database(name)`   | Delete a database                                  |
| `list_databases(limit, offset)` | List all databases                          |

**Note:** 
- Embedded/Server mode: No tenant concept (tenant=None in Database objects)
- OceanBase mode: Multi-tenant architecture (tenant is set in Database objects)

### Collection Operations

The Collection class provides two main methods for retrieving data: `get()` and `query()`.

#### Creating a Collection

```python
import seekdbclient

# Create a client
client = seekdbclient.Client(host="127.0.0.1", port=2881, database="test")

# Create a collection with vector dimension
collection = client.create_collection(name="my_collection", dimension=3)
```

#### `get()` - Retrieve Data by IDs or Filters

The `get()` method retrieves documents from a collection without vector similarity search. It supports filtering by IDs, metadata, and document content.

**Parameters:**
- `ids` (optional): Single ID (string) or list of IDs to retrieve
- `where` (optional): Metadata filter conditions using operators like `$eq`, `$gte`, `$in`, `$or`, etc.
- `where_document` (optional): Document content filter using `$contains` for full-text search
- `limit` (optional): Maximum number of results to return
- `offset` (optional): Number of results to skip for pagination
- `include` (optional): List of fields to include: `["documents", "metadatas", "embeddings"]`

**Examples:**

```python
# Get by single ID
results = collection.get(ids="123")

# Get by multiple IDs
results = collection.get(ids=["1", "2", "3"])

# Get by metadata filter
results = collection.get(
    where={"category": {"$eq": "AI"}},
    limit=10
)

# Get by comparison operator
results = collection.get(
    where={"score": {"$gte": 90}},
    limit=10
)

# Get by $in operator
results = collection.get(
    where={"tag": {"$in": ["ml", "python"]}},
    limit=10
)

# Get by logical operators ($or)
results = collection.get(
    where={
        "$or": [
            {"category": {"$eq": "AI"}},
            {"tag": {"$eq": "python"}}
        ]
    },
    limit=10
)

# Get by document content filter
results = collection.get(
    where_document={"$contains": "machine learning"},
    limit=10
)

# Get with combined filters
results = collection.get(
    where={"category": {"$eq": "AI"}},
    where_document={"$contains": "machine"},
    limit=10
)

# Get with pagination
results = collection.get(limit=2, offset=1)

# Get with specific fields
results = collection.get(
    ids=["1", "2"],
    include=["documents", "metadatas", "embeddings"]
)

# Get all data (up to limit)
results = collection.get(limit=100)
```

#### `query()` - Vector Similarity Search

The `query()` method performs vector similarity search to find the most similar documents to the query vector(s).

**Parameters:**
- `query_embeddings` (required): Single vector (list of floats) or list of vectors for batch queries
- `n_results` (required): Number of similar results to return
- `where` (optional): Metadata filter conditions (same operators as `get()`)
- `where_document` (optional): Document content filter
- `include` (optional): List of fields to include: `["documents", "metadatas", "embeddings"]`

**Examples:**

```python
# Basic vector similarity query
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    n_results=3
)

# Query with metadata filter
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"category": {"$eq": "AI"}},
    n_results=5
)

# Query with comparison operator
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"score": {"$gte": 90}},
    n_results=5
)

# Query with document filter
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where_document={"$contains": "machine learning"},
    n_results=5
)

# Query with combined filters
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
    where_document={"$contains": "machine"},
    n_results=5
)

# Query with $in operator
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={"tag": {"$in": ["ml", "python"]}},
    n_results=5
)

# Query with logical operators ($or)
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    where={
        "$or": [
            {"category": {"$eq": "AI"}},
            {"tag": {"$eq": "python"}}
        ]
    },
    n_results=5
)

# Query with multiple vectors (batch query)
results = collection.query(
    query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
    n_results=2
)

# Query with specific fields
results = collection.query(
    query_embeddings=[1.0, 2.0, 3.0],
    include=["documents", "metadatas", "embeddings"],
    n_results=3
)
```

#### Filter Operators

**Metadata Filters (`where` parameter):**
- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal to
- `$lt`: Less than
- `$lte`: Less than or equal to
- `$in`: Value in array
- `$nin`: Value not in array
- `$or`: Logical OR
- `$and`: Logical AND

**Document Filters (`where_document` parameter):**
- `$contains`: Full-text search (contains substring)

#### Return Values

Both `get()` and `query()` return a `QueryResult` object, which is iterable and contains result items. Each item has:
- `_id`: Record ID (always included)
- `document`: Document text (if included)
- `embedding`: Vector embedding (if included)
- `metadata`: Metadata dictionary (if included)
- `distance`: Similarity distance (only for `query()` results)

```python
# Access results
results = collection.get(ids=["1", "2"])
for item in results:
    print(item._id)
    print(item.document)
    print(item.metadata)
    
# Or convert to dictionary
for item in results:
    result_dict = item.to_dict()
    print(result_dict)
```


## Testing
```bash
python3 -m pytest seekdbclient/tests/ -v

python3 -m pytest seekdbclient/tests/ -v -s # print log

python3 -m pytest seekdbclient/tests/test_client_creation.py::TestClientCreation::test_create_server_client -v

python3 -m pytest seekdbclient/tests/test_client_creation.py -v

```

### Environment Variables (Optional)
`test_client_creation.py` honors the following overrides:
```bash
export SEEKDB_PATH=/data/seekdb
export SEEKDB_DATABASE=demo
export SERVER_HOST=127.0.0.1
export SERVER_PORT=2881           # SeekDB Server port
export SERVER_USER=root
export SERVER_PASSWORD=secret
export OB_HOST=127.0.0.1
export OB_PORT=11402               # OceanBase port (from 'ob do mysql -n ob1')
export OB_TENANT=mysql             # OceanBase tenant
export OB_USER=root
export OB_PASSWORD=
```

## Architecture
- **ClientAPI**: Collection operations interface
- **AdminAPI**: Database operations interface
- **ServerAPI (BaseClient)**: Implements both interfaces
- **_ClientProxy**: Exposes only collection operations
- **_AdminClientProxy**: Exposes only database operations

```
Client() → _ClientProxy → BaseClient (ServerAPI)
AdminClient() → _AdminClientProxy → BaseClient (ServerAPI)
```

## License
MIT License
