"""
Collection query tests - testing collection.query() interface for all three modes
Supports configuring connection parameters via environment variables
"""
import pytest
import sys
import os
import time
import json
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import seekdbclient


# ==================== Environment Variable Configuration ====================
# Embedded mode
SEEKDB_PATH = os.environ.get('SEEKDB_PATH', os.path.join(project_root, "seekdb_store"))
SEEKDB_DATABASE = os.environ.get('SEEKDB_DATABASE', 'test')

# Server mode
SERVER_HOST = os.environ.get('SERVER_HOST', 'localhost')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))
SERVER_DATABASE = os.environ.get('SERVER_DATABASE', 'test')
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SERVER_PASSWORD', '')

# OceanBase mode
OB_HOST = os.environ.get('OB_HOST', 'localhost')
OB_PORT = int(os.environ.get('OB_PORT', '11202'))
OB_TENANT = os.environ.get('OB_TENANT', 'mysql')
OB_DATABASE = os.environ.get('OB_DATABASE', 'test')
OB_USER = os.environ.get('OB_USER', 'root')
OB_PASSWORD = os.environ.get('OB_PASSWORD', '')


class TestCollectionQuery:
    """Test collection.query() interface for all three modes"""
    
    def _create_test_collection(self, client, collection_name: str, dimension: int = 3):
        """Helper method to create a test collection"""
        table_name = f"c$v1${collection_name}"
        sql = f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
            _id bigint PRIMARY KEY NOT NULL AUTO_INCREMENT,
            document string,
            embedding vector({dimension}),
            metadata json,
            FULLTEXT INDEX idx1(document),
            VECTOR INDEX idx2 (embedding) with(distance=l2, type=hnsw, lib=vsag)
        );"""
        client._server.execute(sql)
        return seekdbclient.Collection(client=client._server, name=collection_name, dimension=dimension)
    
    def _insert_test_data(self, client, collection_name: str):
        """Helper method to insert test data"""
        table_name = f"c$v1${collection_name}"
        
        # Insert test data with vectors, documents, and metadata
        test_data = [
            {
                "document": "This is a test document about machine learning",
                "embedding": [1.0, 2.0, 3.0],
                "metadata": {"category": "AI", "score": 95, "tag": "ml"}
            },
            {
                "document": "Python programming tutorial for beginners",
                "embedding": [2.0, 3.0, 4.0],
                "metadata": {"category": "Programming", "score": 88, "tag": "python"}
            },
            {
                "document": "Advanced machine learning algorithms",
                "embedding": [1.1, 2.1, 3.1],
                "metadata": {"category": "AI", "score": 92, "tag": "ml"}
            },
            {
                "document": "Data science with Python",
                "embedding": [2.1, 3.1, 4.1],
                "metadata": {"category": "Data Science", "score": 90, "tag": "python"}
            },
            {
                "document": "Introduction to neural networks",
                "embedding": [1.2, 2.2, 3.2],
                "metadata": {"category": "AI", "score": 85, "tag": "neural"}
            }
        ]
        
        for data in test_data:
            # Convert vector to string format: [1.0,2.0,3.0]
            vector_str = "[" + ",".join(map(str, data["embedding"])) + "]"
            # Convert metadata to JSON string
            metadata_str = json.dumps(data["metadata"]).replace("'", "\\'")
            # Escape single quotes in document
            document_str = data["document"].replace("'", "\\'")
            
            sql = f"""INSERT INTO `{table_name}` (document, embedding, metadata) 
                     VALUES ('{document_str}', '{vector_str}', '{metadata_str}')"""
            client._server.execute(sql)
    
    def _cleanup_collection(self, client, collection_name: str):
        """Helper method to cleanup test collection"""
        table_name = f"c$v1${collection_name}"
        try:
            client._server.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"   Cleaned up test table: {table_name}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup test table: {cleanup_error}")
    
    def test_embedded_collection_query(self):
        """Test collection.query() with embedded client"""
        if not os.path.exists(SEEKDB_PATH):
            pytest.skip(
                f"SeekDB data directory does not exist: {SEEKDB_PATH}\n"
                f"Set SEEKDB_PATH environment variable to run this test"
            )
        
        # Check if seekdb package is available
        try:
            import seekdb
        except ImportError:
            pytest.skip("SeekDB embedded package is not installed")
        
        # Create embedded client
        client = seekdbclient.Client(
            path=SEEKDB_PATH,
            database=SEEKDB_DATABASE
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, seekdbclient.SeekdbEmbeddedClient)
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        collection = self._create_test_collection(client, collection_name, dimension=3)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for embedded client")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results")
            
            # Test 2: Query with metadata filter
            print(f"✅ Testing query with metadata filter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"category": {"$eq": "AI"}},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with category='AI'")
            
            # Test 3: Query with document filter
            print(f"✅ Testing query with document filter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where_document={"$contains": "machine learning"},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results containing 'machine learning'")
            
            # Test 4: Query with include parameter
            print(f"✅ Testing query with include parameter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                include=["documents", "metadatas"],
                n_results=3
            )
            assert results is not None
            if len(results) > 0:
                # Check that results have the expected fields
                for item in results:
                    assert hasattr(item, '_id') or '_id' in item.to_dict()
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_server_collection_query(self):
        """Test collection.query() with server client"""
        # Create server client
        client = seekdbclient.Client(
            host=SERVER_HOST,
            port=SERVER_PORT,
            database=SERVER_DATABASE,
            user=SERVER_USER,
            password=SERVER_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, seekdbclient.SeekdbServerClient)
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.skip(f"Server connection failed ({SERVER_HOST}:{SERVER_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        collection = self._create_test_collection(client, collection_name, dimension=3)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for server client")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results")
            
            # Test 2: Query with metadata filter using comparison operators
            print(f"✅ Testing query with metadata filter ($gte)")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"score": {"$gte": 90}},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with score >= 90")
            
            # Test 3: Query with combined filters
            print(f"✅ Testing query with combined filters")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
                where_document={"$contains": "machine"},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results matching all filters")
            
            # Test 4: Query with $in operator
            print(f"✅ Testing query with $in operator")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                where={"tag": {"$in": ["ml", "python"]}},
                n_results=5
            )
            assert results is not None
            print(f"   Found {len(results)} results with tag in ['ml', 'python']")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_collection_query(self):
        """Test collection.query() with OceanBase client"""
        # Create OceanBase client
        client = seekdbclient.OBClient(
            host=OB_HOST,
            port=OB_PORT,
            tenant=OB_TENANT,
            database=OB_DATABASE,
            user=OB_USER,
            password=OB_PASSWORD
        )
        
        assert client is not None
        assert hasattr(client, '_server')
        assert isinstance(client._server, seekdbclient.OceanBaseServerClient)
        
        # Test connection
        try:
            result = client._server.execute("SELECT 1 as test")
            assert result is not None
        except Exception as e:
            pytest.skip(f"OceanBase connection failed ({OB_HOST}:{OB_PORT}): {e}")
        
        # Create test collection
        collection_name = f"test_query_{int(time.time())}"
        collection = self._create_test_collection(client, collection_name, dimension=3)
        
        try:
            # Insert test data
            self._insert_test_data(client, collection_name)
            
            # Test 1: Basic vector similarity query
            print(f"\n✅ Testing basic query for OceanBase client")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                n_results=3
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results")
            
            # Test 2: Query with multiple vectors
            print(f"✅ Testing query with multiple vectors")
            results = collection.query(
                query_embeddings=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                n_results=2
            )
            assert results is not None
            print(f"   Found {len(results)} results for multiple query vectors")
            
            # Test 3: Query with logical operators
            print(f"✅ Testing query with logical operators ($or)")
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
            assert results is not None
            print(f"   Found {len(results)} results with $or condition")
            
            # Test 4: Query with include parameter to get specific fields
            print(f"✅ Testing query with include parameter")
            results = collection.query(
                query_embeddings=[1.0, 2.0, 3.0],
                include=["documents", "metadatas", "embeddings"],
                n_results=3
            )
            assert results is not None
            if len(results) > 0:
                # Verify result structure
                for item in results:
                    result_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                    assert '_id' in result_dict
                    print(f"   Result keys: {list(result_dict.keys())}")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SeekDBClient - Collection Query Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

