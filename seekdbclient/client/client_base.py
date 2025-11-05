"""
Base client interface definition
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Dict, Any, Union, TYPE_CHECKING

from .base_connection import BaseConnection
from .admin_client import AdminAPI, DEFAULT_TENANT

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
    
    @abstractmethod
    def get_collection(self, name: str) -> "Collection":
        """
        Get a collection object (user-facing API)
        
        Args:
            name: Collection name
            
        Returns:
            Collection object
        """
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection (user-facing API)
        
        Args:
            name: Collection name
        """
        pass
    
    @abstractmethod
    def list_collections(self) -> List["Collection"]:
        """
        List all collections (user-facing API)
        
        Returns:
            List of Collection objects
        """
        pass
    
    @abstractmethod
    def has_collection(self, name: str) -> bool:
        """
        Check if a collection exists (user-facing API)
        
        Args:
            name: Collection name
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    # ==================== Collection Internal Operations (Called by Collection) ====================
    # These methods are called by Collection objects, different clients implement different logic
    
    # -------------------- DML Operations --------------------
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
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
