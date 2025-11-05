"""
Base client interface definition
"""
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Dict, Any, Union, TYPE_CHECKING

from .base_connection import BaseConnection
from .admin_client import AdminAPI, DEFAULT_TENANT
from .sql_based_collection_operator import SqlBasedCollectionOperator

if TYPE_CHECKING:
    from .collection import Collection
    from .database import Database
else:
    from .collection import Collection  # Import for runtime use

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
    ) -> Collection:
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
        table_name = f"c$v1{name}"
        
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
    
    def get_collection(self, name: str) -> Collection:
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
        table_name = f"c$v1{name}"
        
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
        table_name = f"c$v1{name}"
        
        # Check if table exists first
        if not self.has_collection(name):
            raise ValueError(f"Collection '{name}' does not exist (table '{table_name}' not found)")
        
        # Execute DROP TABLE SQL
        self.execute(f"DROP TABLE IF EXISTS `{table_name}`")
    
    def list_collections(self) -> List[Collection]:
        """
        List all collections (user-facing API)
        
        Returns:
            List of Collection objects
        """
        # List all tables that start with 'c$v1'
        # Use SHOW TABLES LIKE 'c$v1%' to filter collection tables
        try:
            tables = self.execute("SHOW TABLES LIKE 'c$v1%'")
        except Exception:
            # Fallback: try to query information_schema
            try:
                # Get current database name
                db_result = self.execute("SELECT DATABASE()")
                if db_result and len(db_result) > 0:
                    db_name = db_result[0][0] if isinstance(db_result[0], (tuple, list)) else db_result[0].get('DATABASE()', '')
                    tables = self.execute(
                        f"SELECT TABLE_NAME FROM information_schema.TABLES "
                        f"WHERE TABLE_SCHEMA = '{db_name}' AND TABLE_NAME LIKE 'c$v1%'"
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
            
            # Extract collection name from table name (remove 'c$v1' prefix)
            if table_name.startswith('c$v1'):
                collection_name = table_name[4:]  # Remove 'c$v1' prefix
                
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
        table_name = f"c$v1{name}"
        
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
    ) -> Collection:
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
    
    @abstractmethod
    def _collection_query(
        self,
        collection_id: Optional[str],
        collection_name: str,
        query_vector: Optional[Union[List[float], List[List[float]]]] = None,
        query_text: Optional[Union[str, List[str]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        [Internal] Query collection by vector similarity
        
        Args:
            collection_id: Collection ID
            collection_name: Collection name
            query_vector: Query vector(s) (optional)
            query_text: Query text(s) (optional)
            n_results: Number of results to return
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            include: Fields to include in results (optional)
            **kwargs: Additional parameters
            
        Returns:
            Query results dictionary
        """
        pass
    
    @abstractmethod
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
    ) -> Dict[str, Any]:
        """
        [Internal] Get data from collection by IDs or filters
        
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
            Results dictionary
        """
        pass
    
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
