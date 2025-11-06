"""
Base client interface definition
"""
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Dict, Any, Union, TYPE_CHECKING, Tuple, Callable

from .base_connection import BaseConnection
from .admin_client import AdminAPI, DEFAULT_TENANT
from .sql_based_collection_operator import SqlBasedCollectionOperator
from .meta_info import CollectionNames
from .query_result import QueryResult
from .filters import FilterBuilder

from .collection import Collection

from .database import Database

logger = logging.getLogger(__name__)

class ClientAPI(ABC):
    """
    Client API interface for collection operations only.
    This is what end users interact with through the Client proxy.
    """
    
    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        **kwargs
    ) -> "Collection":
        """Create collection"""
        pass
    
    @abstractmethod
    def get_collection(self, name: str) -> "Collection":
        """Get collection object"""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete collection"""
        pass
    
    @abstractmethod
    def list_collections(self) -> List["Collection"]:
        """List all collections"""
        pass
    
    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """Check if collection exists"""
        pass


class BaseClient(BaseConnection, AdminAPI):
    """
    Abstract base class for all clients.
    
    Design Pattern:
    1. Provides public collection management methods (create_collection, get_collection, etc.)
    2. Defines internal operation interfaces (_collection_* methods) called by Collection objects
    3. Subclasses implement all abstract methods to provide specific business logic
    
    Benefits of this design:
    - Collection object interface is unified regardless of which client created it
    - Different clients can have completely different underlying implementations (SQL/gRPC/REST)
    - Easy to extend with new client types
    
    Inherits connection management from BaseConnection and database operations from AdminAPI.
    """
    
    # ==================== Collection Management (User-facing) ====================
    
    def create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        **kwargs
    ) -> "Collection":
        """
        Create a collection (user-facing API)
        
        Args:
            name: Collection name
            dimension: Vector dimension
            **kwargs: Additional parameters
            
        Returns:
            Collection object
        """
        if dimension is None:
            raise ValueError("dimension parameter is required for creating a collection")
        
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Construct CREATE TABLE SQL statement
        sql = f"""CREATE TABLE `{table_name}` (
            _id bigint PRIMARY KEY NOT NULL AUTO_INCREMENT,
            document string,
            embedding vector({dimension}),
            metadata json,
            FULLTEXT INDEX idx1(document),
            VECTOR INDEX idx2 (embedding) with(distance=l2, type=hnsw, lib=vsag)
        );"""
        
        # Execute SQL to create table
        self.execute(sql)
        
        # Create and return Collection object
        return Collection(client=self, name=name, dimension=dimension, **kwargs)
    
    def get_collection(self, name: str) -> "Collection":
        """
        Get a collection object (user-facing API)
        
        Args:
            name: Collection name
            
        Returns:
            Collection object
            
        Raises:
            ValueError: If collection does not exist
        """
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Check if table exists by describing it
        try:
            table_info = self.execute(f"DESCRIBE `{table_name}`")
            if not table_info or len(table_info) == 0:
                raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)")
        except Exception as e:
            # If DESCRIBE fails, check if it's because table doesn't exist
            error_msg = str(e).lower()
            if "doesn't exist" in error_msg or "not found" in error_msg or "table" in error_msg:
                raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)") from e
            raise
        
        # Extract dimension from embedding column
        dimension = None
        for row in table_info:
            # Handle both dict and tuple formats
            if isinstance(row, dict):
                field_name = row.get('Field', row.get('field', ''))
                field_type = row.get('Type', row.get('type', ''))
            elif isinstance(row, (tuple, list)):
                field_name = row[0] if len(row) > 0 else ''
                field_type = row[1] if len(row) > 1 else ''
            else:
                continue
            
            if field_name == 'embedding' and 'vector' in str(field_type).lower():
                # Extract dimension from vector(dimension) format
                match = re.search(r'vector\s*\(\s*(\d+)\s*\)', str(field_type), re.IGNORECASE)
                if match:
                    dimension = int(match.group(1))
                break
        
        # Create and return Collection object
        return Collection(client=self, name=name, dimension=dimension)
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection (user-facing API)
        
        Args:
            name: Collection name
            
        Raises:
            ValueError: If collection does not exist
        """
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Check if table exists first
        if not self.has_collection(name):
            raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)")
        
        # Execute DROP TABLE SQL
        self.execute(f"DROP TABLE IF EXISTS `{table_name}`")
    
    def list_collections(self) -> List["Collection"]:
        """
        List all collections (user-facing API)
        
        Returns:
            List of Collection objects
        """
        # List all tables that start with 'c$v1'
        # Use SHOW TABLES LIKE 'c$v1%' to filter collection tables
        try:
            tables = self.execute("SHOW TABLES LIKE 'c$v1$%'")
        except Exception:
            # Fallback: try to query information_schema
            try:
                # Get current database name
                db_result = self.execute("SELECT DATABASE()")
                if db_result and len(db_result) > 0:
                    db_name = db_result[0][0] if isinstance(db_result[0], (tuple, list)) else db_result[0].get('DATABASE()', '')
                    tables = self.execute(
                        f"SELECT TABLE_NAME FROM information_schema.TABLES "
                        f"WHERE TABLE_SCHEMA = '{db_name}' AND TABLE_NAME LIKE 'c$v1$%'"
                    )
                else:
                    return []
            except Exception:
                return []
        
        collections = []
        for row in tables:
            # Extract table name
            if isinstance(row, dict):
                # Server client returns dict, get the first value
                table_name = list(row.values())[0] if row else ''
            elif isinstance(row, (tuple, list)):
                # Embedded client returns tuple, first element is table name
                table_name = row[0] if len(row) > 0 else ''
            else:
                table_name = str(row)
            
            # Extract collection name from table name (remove 'c$v1$' prefix)
            if table_name.startswith('c$v1$'):
                collection_name = table_name[5:]  # Remove 'c$v1$' prefix
                
                # Get collection with dimension
                try:
                    collection = self.get_collection(collection_name)
                    collections.append(collection)
                except Exception:
                    # Skip if we can't get collection info
                    continue
        
        return collections
    
    def has_collection(self, name: str) -> bool:
        """
        Check if a collection exists (user-facing API)
        
        Args:
            name: Collection name
            
        Returns:
            True if exists, False otherwise
        """
        # Construct table name: c$v1${name}
        table_name = CollectionNames.table_name(name)
        
        # Check if table exists
        try:
            # Try to describe the table
            table_info = self.execute(f"DESCRIBE `{table_name}`")
            return table_info is not None and len(table_info) > 0
        except Exception:
            # If DESCRIBE fails, table doesn't exist
            return False
    
    def get_or_create_collection(
        self,
        name: str,
        dimension: Optional[int] = None,
        **kwargs
    ) -> "Collection":
        """
        Get an existing collection or create it if it doesn't exist (user-facing API)
        
        Args:
            name: Collection name
            dimension: Vector dimension (required if creating new collection)
            **kwargs: Additional parameters for create_collection
            
        Returns:
            Collection object
            
        Raises:
            ValueError: If collection doesn't exist and dimension is not provided
        """
        # First, try to get the collection
        if self.has_collection(name):
            return self.get_collection(name)
        
        # Collection doesn't exist, create it
        if dimension is None:
            raise ValueError(
                f"Collection '{name}' does not exist and dimension parameter is required "
                f"for creating a new collection"
            )
        
        return self.create_collection(name=name, dimension=dimension, **kwargs)
    
    # ==================== Collection Internal Operations (Called by Collection) ====================
    # These methods are called by Collection objects, different clients implement different logic
    
    # -------------------- DML Operations --------------------
    
    def _collection_add(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Add data to collection
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            vectors: Single vector or list of vectors (optional)
            metadatas: Single metadata dict or list of metadata dicts (optional)
            documents: Single document or list of documents (optional)
            **kwargs: Additional parameters
        """
        SqlBasedCollectionOperator.add(self, collection_name, ids, vectors, metadatas, documents)
    
    def _collection_update(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Update data in collection
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to update
            vectors: New vectors (optional)
            metadatas: New metadata (optional)
            documents: New documents (optional)
            **kwargs: Additional parameters
        """
        SqlBasedCollectionOperator.update(self, collection_name, ids, vectors, metadatas, documents)
    
    def _collection_upsert(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Insert or update data in collection
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs
            vectors: Vectors (optional)
            metadatas: Metadata (optional)
            documents: Documents (optional)
            **kwargs: Additional parameters
        """
        SqlBasedCollectionOperator.upsert(self, collection_name, ids, vectors, metadatas, documents)
    
    def _collection_delete(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        [Internal] Delete data from collection
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs to delete (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            **kwargs: Additional parameters
        """
        SqlBasedCollectionOperator.delete(self, collection_name, ids, where, where_document, **kwargs)
    
    # -------------------- DQL Operations --------------------
    # Note: _collection_query() and _collection_get() are implemented below with common SQL-based logic
    
    def _normalize_query_vectors(
        self,
        query_embeddings: Optional[Union[List[float], List[List[float]]]]
    ) -> List[List[float]]:
        """
        Normalize query vectors to list of lists format
        
        Args:
            query_embeddings: Single vector or list of vectors
            
        Returns:
            List of vectors (each vector is a list of floats)
        """
        if query_embeddings is None:
            return []
        
        # Check if it's a single vector (list of numbers)
        if query_embeddings and isinstance(query_embeddings[0], (int, float)):
            return [query_embeddings]
        
        return query_embeddings
    
    def _normalize_include_fields(
        self,
        include: Optional[List[str]]
    ) -> Dict[str, bool]:
        """
        Normalize include parameter to a dictionary
        
        Args:
            include: List of fields to include (e.g., ["documents", "metadatas", "embeddings"])
            
        Returns:
            Dictionary with field names as keys and True as values
            Default includes: documents, metadatas (but not embeddings)
        """
        # Default includes documents and metadatas
        default_fields = {"documents": True, "metadatas": True}
        
        if include is None:
            return default_fields
        
        # Build include dict from list
        include_dict = {}
        for field in include:
            include_dict[field] = True
        
        return include_dict
    
    def _embed_texts(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """
        Embed text(s) to vector(s)
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional parameters for embedding
            
        Returns:
            List of vectors
            
        Note:
            This is a placeholder method. Subclasses should override this
            to provide actual embedding functionality, or users should
            provide query_embeddings directly.
        """
        raise NotImplementedError(
            "Text embedding is not implemented yet. "
            "Please provide query_embeddings directly instead of query_texts."
        )
    
    def _normalize_row(self, row: Any, cursor_description: Optional[Any] = None) -> Dict[str, Any]:
        """
        Normalize database row to dictionary format
        
        Args:
            row: Database row (can be dict or tuple)
            cursor_description: Cursor description for tuple rows
            
        Returns:
            Dictionary with column names as keys
        """
        if isinstance(row, dict):
            return row
        
        # Convert tuple to dict using cursor description
        if cursor_description is not None:
            row_dict = {}
            for idx, col_desc in enumerate(cursor_description):
                row_dict[col_desc[0]] = row[idx]
            return row_dict
        
        # Fallback: assume it's already a dict or try to convert
        return dict(row) if hasattr(row, '_asdict') else row
    
    def _execute_query_with_cursor(
        self,
        conn: Any,
        sql: str,
        params: List[Any],
        use_context_manager: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return normalized rows
        
        Args:
            conn: Database connection
            sql: SQL query string
            params: Query parameters
            use_context_manager: Whether to use context manager for cursor (default: True)
            
        Returns:
            List of normalized row dictionaries
        """
        if use_context_manager:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                # Normalize rows
                normalized_rows = []
                for row in rows:
                    normalized_rows.append(self._normalize_row(row, cursor.description))
                return normalized_rows
        else:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                # Normalize rows
                normalized_rows = []
                for row in rows:
                    normalized_rows.append(self._normalize_row(row, cursor.description))
                return normalized_rows
            finally:
                cursor.close()
    
    def _build_select_clause(self, include_fields: Dict[str, bool]) -> str:
        """
        Build SELECT clause based on include fields
        
        Args:
            include_fields: Dictionary of fields to include
            
        Returns:
            SELECT clause string
        """
        select_fields = ["_id"]
        if include_fields.get("embeddings") or include_fields.get("embedding"):
            select_fields.append("embedding")
        if include_fields.get("documents") or include_fields.get("document"):
            select_fields.append("document")
        if include_fields.get("metadatas") or include_fields.get("metadata"):
            select_fields.append("metadata")
        
        return ", ".join(select_fields)
    
    def _build_where_clause(
        self,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        id_list: Optional[List[str]] = None
    ) -> Tuple[str, List[Any]]:
        """
        Build WHERE clause from filters
        
        Args:
            where: Metadata filter
            where_document: Document filter
            id_list: List of IDs to filter
            
        Returns:
            Tuple of (where_clause, params)
        """
        where_clauses = []
        params = []
        
        # Add ids filter if provided
        if id_list:
            placeholders = ",".join(["%s"] * len(id_list))
            where_clauses.append(f"_id IN ({placeholders})")
            params.extend(id_list)
        
        # Add metadata filter
        if where:
            meta_clause, meta_params = FilterBuilder.build_metadata_filter(where, "metadata")
            if meta_clause:
                where_clauses.append(meta_clause)
                params.extend(meta_params)
        
        # Add document filter
        if where_document:
            doc_clause, doc_params = FilterBuilder.build_document_filter(where_document, "document")
            if doc_clause:
                where_clauses.append(doc_clause)
                params.extend(doc_params)
        
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return where_clause, params
    
    def _parse_row_value(self, value: Any) -> Any:
        """
        Parse row value (handle JSON strings)
        
        Args:
            value: Raw value from database
            
        Returns:
            Parsed value
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
        
        return value
    
    def _process_query_row(
        self,
        row: Dict[str, Any],
        include_fields: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Process a row from query results
        
        Args:
            row: Normalized row dictionary
            include_fields: Fields to include
            
        Returns:
            Result item dictionary
        """
        result_item = {"_id": row["_id"]}
        
        if "document" in row and row["document"] is not None:
            result_item["document"] = row["document"]
        
        if "embedding" in row and row["embedding"] is not None:
            result_item["embedding"] = self._parse_row_value(row["embedding"])
        
        if "metadata" in row and row["metadata"] is not None:
            result_item["metadata"] = self._parse_row_value(row["metadata"])
        
        if "distance" in row:
            result_item["distance"] = float(row["distance"])
        
        return result_item
    
    def _process_get_row(
        self,
        row: Dict[str, Any],
        include_fields: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Process a row from get results
        
        Args:
            row: Normalized row dictionary
            include_fields: Fields to include
            
        Returns:
            Result item dictionary with id, document, embedding, metadata
        """
        record_id = row["_id"]
        document = None
        embedding = None
        metadata = None
        
        # Include document if requested
        if include_fields.get("documents") or include_fields.get("document"):
            if "document" in row:
                document = row["document"]
        
        # Include metadata if requested
        if include_fields.get("metadatas") or include_fields.get("metadata"):
            if "metadata" in row and row["metadata"] is not None:
                metadata = self._parse_row_value(row["metadata"])
        
        # Include embedding if requested
        if include_fields.get("embeddings") or include_fields.get("embedding"):
            if "embedding" in row and row["embedding"] is not None:
                embedding = self._parse_row_value(row["embedding"])
        
        return {
            "id": record_id,
            "document": document,
            "embedding": embedding,
            "metadata": metadata
        }
    
    def _use_context_manager_for_cursor(self) -> bool:
        """
        Whether to use context manager for cursor
        
        Returns:
            True if context manager should be used, False otherwise
        """
        # Default implementation: use context manager
        # Subclasses can override this if they need different behavior
        return True
    
    # -------------------- DQL Operations (Common Implementation) --------------------
    
    def _collection_query(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        query_texts: Optional[Union[str, List[str]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> QueryResult:
        """
        [Internal] Query collection by vector similarity - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query_embeddings: Query vector(s) (preferred)
            query_texts: Query text(s) - will be embedded if provided (preferred)
            n_results: Number of results (default: 10)
            where: Metadata filter
            where_document: Document filter
            include: Fields to include
            **kwargs: Additional parameters
            
        Returns:
            QueryResult object containing query results
        """
        logger.info(f"Querying collection '{collection_name}' with n_results={n_results}")
        conn = self._ensure_connection()
        
        # Convert collection name to table name
        table_name = f"c$v1${collection_name}"
        
        # Handle text embedding if query_texts provided
        if query_texts is not None and query_embeddings is None:
            logger.info("Embedding query texts...")
            query_embeddings = self._embed_texts(query_texts, **kwargs)
        
        # Normalize query vectors to list of lists
        query_vectors = self._normalize_query_vectors(query_embeddings)
        
        if not query_vectors:
            raise ValueError("Either query_embeddings or query_texts must be provided")
        
        # Normalize include fields
        include_fields = self._normalize_include_fields(include)
        
        # Build SELECT clause
        select_clause = self._build_select_clause(include_fields)
        
        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document)
        
        # Collect all results from all query vectors
        all_results = []
        use_context_manager = self._use_context_manager_for_cursor()
        
        for query_vector in query_vectors:
            # Convert vector to string format for SQL
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
            # Build SQL query with vector distance calculation
            # Reference: SELECT id, vec FROM t2 ORDER BY l2_distance(vec, '[0.1, 0.2, 0.3]') APPROXIMATE LIMIT 5;
            # Need to include distance in SELECT for result processing
            sql = f"""
                SELECT {select_clause}, 
                       l2_distance(embedding, '{vector_str}') AS distance
                FROM `{table_name}`
                {where_clause}
                ORDER BY l2_distance(embedding, '{vector_str}')
                APPROXIMATE
                LIMIT %s
            """
            
            # Execute query
            query_params = params + [n_results]
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"Parameters: {query_params}")
            
            rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)
            
            # Process rows
            for row in rows:
                result_item = self._process_query_row(row, include_fields)
                all_results.append(result_item)
        
        # Sort by distance and limit to n_results (in case of multiple query vectors)
        all_results.sort(key=lambda x: x.get("distance", float("inf")))
        all_results = all_results[:n_results]
        
        # Convert to QueryResult
        query_result = QueryResult()
        for result_dict in all_results:
            query_result.add_item(
                id=result_dict.get("_id"),
                document=result_dict.get("document"),
                embedding=result_dict.get("embedding"),
                metadata=result_dict.get("metadata"),
                distance=result_dict.get("distance")
            )
        
        logger.info(f"✅ Query completed for '{collection_name}', found {len(query_result)} results")
        return query_result
    
    def _collection_get(
        self,
        collection_id: Optional[str],
        collection_name: str,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> QueryResult:
        """
        [Internal] Get data from collection by IDs or filters - Common SQL-based implementation
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            ids: Single ID or list of IDs (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            limit: Maximum number of results (optional)
            offset: Number of results to skip (optional)
            include: Fields to include in results (optional)
            **kwargs: Additional parameters
            
        Returns:
            QueryResult object containing get results
        """
        logger.info(f"Getting data from collection '{collection_name}'")
        conn = self._ensure_connection()
        
        # Convert collection name to table name
        table_name = f"c$v1${collection_name}"
        
        # Set defaults
        if limit is None:
            limit = 100
        if offset is None:
            offset = 0
        
        # Normalize ids to list
        id_list = None
        if ids is not None:
            if isinstance(ids, str):
                id_list = [ids]
            else:
                id_list = ids
        
        # Normalize include fields (default includes documents and metadatas)
        include_fields = self._normalize_include_fields(include)
        
        # Build SELECT clause - always include _id
        select_clause = self._build_select_clause(include_fields)
        
        # Build WHERE clause from filters
        where_clause, params = self._build_where_clause(where, where_document, id_list)
        
        # Build SQL query
        sql = f"""
            SELECT {select_clause}
            FROM `{table_name}`
            {where_clause}
            LIMIT %s OFFSET %s
        """
        
        # Execute query
        query_params = params + [limit, offset]
        logger.debug(f"Executing SQL: {sql}")
        logger.debug(f"Parameters: {query_params}")
        
        use_context_manager = self._use_context_manager_for_cursor()
        rows = self._execute_query_with_cursor(conn, sql, query_params, use_context_manager)
        
        # Build QueryResult
        query_result = QueryResult()
        
        for row in rows:
            # Process row
            processed_row = self._process_get_row(row, include_fields)
            
            query_result.add_item(
                id=processed_row["id"],
                document=processed_row["document"],
                embedding=processed_row["embedding"],
                metadata=processed_row["metadata"]
            )
        
        logger.info(f"✅ Get completed for '{collection_name}', found {len(query_result)} results")
        return query_result
    
    @abstractmethod
    def _collection_hybrid_search(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query_vector: Optional[Union[List[float], List[List[float]]]] = None,
        query_text: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        [Internal] Hybrid search combining vector similarity and filters
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query_vector: Query vector(s) (optional)
            query_text: Query text(s) (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            n_results: Number of results to return
            include: Fields to include in results (optional)
            **kwargs: Additional parameters
            
        Returns:
            Search results dictionary
        """
        pass
    
    # -------------------- Collection Info --------------------
    
    @abstractmethod
    def _collection_count(
        self,
        collection_id: Optional[str],
        collection_name: str
    ) -> int:
        """
        [Internal] Get the number of items in collection
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            
        Returns:
            Item count
        """
        pass
    
    @abstractmethod
    def _collection_describe(
        self,
        collection_id: Optional[str],
        collection_name: str
    ) -> Dict[str, Any]:
        """
        [Internal] Get detailed collection information
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            
        Returns:
            Collection information dictionary
        """
        pass
