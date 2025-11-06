"""
Collection get tests - testing collection.get() interface for all three modes
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


class TestCollectionGet:
    """Test collection.get() interface for all three modes"""
    
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
        """Helper method to insert test data and return inserted IDs"""
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
        
        # Get the maximum ID before inserting (to identify newly inserted records)
        max_id_before = None
        try:
            result = client._server.execute(f"SELECT MAX(_id) as max_id FROM `{table_name}`")
            if result and len(result) > 0:
                # Handle both dict and tuple return formats
                row = result[0]
                if isinstance(row, dict):
                    max_id_value = row.get('max_id')
                else:
                    # Tuple format - assume first column is max_id
                    max_id_value = row[0] if len(row) > 0 else None
                if max_id_value is not None:
                    max_id_before = int(max_id_value)
        except Exception:
            pass
        
        # Insert all data
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
        
        # Get all inserted IDs by querying records with ID > max_id_before
        inserted_ids = []
        try:
            if max_id_before is not None and max_id_before > 0:
                sql = f"SELECT _id FROM `{table_name}` WHERE _id > {max_id_before} ORDER BY _id ASC"
            else:
                # If table was empty, get all records (should be safe since we just created the table)
                sql = f"SELECT _id FROM `{table_name}` ORDER BY _id ASC"
            
            result = client._server.execute(sql)
            if result:
                for row in result:
                    # Handle both dict and tuple return formats
                    if isinstance(row, dict):
                        id_value = row.get('_id')
                    else:
                        # Tuple format - assume first column is _id
                        id_value = row[0] if len(row) > 0 else None
                    if id_value is not None:
                        inserted_ids.append(str(id_value))
        except Exception as e:
            # Fallback: try to get the last inserted ID
            print(f"   Warning: Failed to get IDs from query, trying fallback: {e}")
            try:
                result = client._server.execute("SELECT LAST_INSERT_ID() as id")
                if result and len(result) > 0:
                    row = result[0]
                    if isinstance(row, dict):
                        last_id = row.get('id') or row.get('LAST_INSERT_ID()')
                    else:
                        last_id = row[0] if len(row) > 0 else None
                    if last_id is not None:
                        inserted_ids = [str(last_id)]
            except Exception:
                pass
        
        return inserted_ids
    
    def _cleanup_collection(self, client, collection_name: str):
        """Helper method to cleanup test collection"""
        table_name = f"c$v1${collection_name}"
        try:
            client._server.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            print(f"   Cleaned up test table: {table_name}")
        except Exception as cleanup_error:
            print(f"   Warning: Failed to cleanup test table: {cleanup_error}")
    
    def test_embedded_collection_get(self):
        """Test collection.get() with embedded client"""
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
        collection_name = f"test_get_{int(time.time())}"
        collection = self._create_test_collection(client, collection_name, dimension=3)
        
        try:
            # Insert test data and get IDs
            inserted_ids = self._insert_test_data(client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")
            
            # Test 1: Get by single ID
            print(f"\n✅ Testing get by single ID for embedded client")
            results = collection.get(ids=inserted_ids[0])
            assert results is not None
            assert len(results) == 1
            print(f"   Found {len(results)} result for ID={inserted_ids[0]}")
            
            # Test 2: Get by multiple IDs
            print(f"✅ Testing get by multiple IDs")
            if len(inserted_ids) >= 2:
                results = collection.get(ids=inserted_ids[:2])
                assert results is not None
                assert len(results) <= 2
                print(f"   Found {len(results)} results for IDs={inserted_ids[:2]}")
            
            # Test 3: Get by metadata filter
            print(f"✅ Testing get with metadata filter")
            results = collection.get(
                where={"category": {"$eq": "AI"}},
                limit=10
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results with category='AI'")
            
            # Test 4: Get by document filter
            print(f"✅ Testing get with document filter")
            results = collection.get(
                where_document={"$contains": "machine learning"},
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results containing 'machine learning'")
            
            # Test 5: Get with include parameter
            print(f"✅ Testing get with include parameter")
            results = collection.get(
                ids=inserted_ids[:2],
                include=["documents", "metadatas"]
            )
            assert results is not None
            if len(results) > 0:
                for item in results:
                    result_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                    assert '_id' in result_dict
                    print(f"   Result keys: {list(result_dict.keys())}")
            
            # Test 6: Get all data with limit
            print(f"✅ Testing get all data with limit")
            results = collection.get(limit=3)
            assert results is not None
            assert len(results) <= 3
            print(f"   Found {len(results)} results (limit=3)")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_server_collection_get(self):
        """Test collection.get() with server client"""
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
        collection_name = f"test_get_{int(time.time())}"
        collection = self._create_test_collection(client, collection_name, dimension=3)
        
        try:
            # Insert test data and get IDs
            inserted_ids = self._insert_test_data(client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")
            
            # Test 1: Get by single ID
            print(f"\n✅ Testing get by single ID for server client")
            results = collection.get(ids=inserted_ids[0])
            assert results is not None
            assert len(results) == 1
            print(f"   Found {len(results)} result for ID={inserted_ids[0]}")
            
            # Test 2: Get by multiple IDs
            print(f"✅ Testing get by multiple IDs")
            if len(inserted_ids) >= 3:
                results = collection.get(ids=inserted_ids[:3])
                assert results is not None
                assert len(results) <= 3
                print(f"   Found {len(results)} results for IDs={inserted_ids[:3]}")
            
            # Test 3: Get by metadata filter with comparison operator
            print(f"✅ Testing get with metadata filter ($gte)")
            results = collection.get(
                where={"score": {"$gte": 90}},
                limit=10
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results with score >= 90")
            
            # Test 4: Get by combined metadata filters
            print(f"✅ Testing get with combined metadata filters")
            results = collection.get(
                where={"category": {"$eq": "AI"}, "score": {"$gte": 90}},
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results with category='AI' and score >= 90")
            
            # Test 5: Get by document filter
            print(f"✅ Testing get with document filter")
            results = collection.get(
                where_document={"$contains": "Python"},
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results containing 'Python'")
            
            # Test 6: Get with $in operator
            print(f"✅ Testing get with $in operator")
            results = collection.get(
                where={"tag": {"$in": ["ml", "python"]}},
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results with tag in ['ml', 'python']")
            
            # Test 7: Get with limit and offset
            print(f"✅ Testing get with limit and offset")
            results = collection.get(limit=2, offset=1)
            assert results is not None
            assert len(results) <= 2
            print(f"   Found {len(results)} results (limit=2, offset=1)")
            
            # Test 8: Get with include parameter
            print(f"✅ Testing get with include parameter")
            results = collection.get(
                ids=inserted_ids[:2],
                include=["documents", "metadatas", "embeddings"]
            )
            assert results is not None
            if len(results) > 0:
                for item in results:
                    result_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                    assert '_id' in result_dict
                    print(f"   Result keys: {list(result_dict.keys())}")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)
    
    def test_oceanbase_collection_get(self):
        """Test collection.get() with OceanBase client"""
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
        collection_name = f"test_get_{int(time.time())}"
        collection = self._create_test_collection(client, collection_name, dimension=3)
        
        try:
            # Insert test data and get IDs
            inserted_ids = self._insert_test_data(client, collection_name)
            assert len(inserted_ids) > 0, f"Failed to get inserted IDs. Expected at least 1, got {len(inserted_ids)}"
            if len(inserted_ids) < 5:
                print(f"   Warning: Expected 5 inserted IDs, but got {len(inserted_ids)}")
            
            # Test 1: Get by single ID
            print(f"\n✅ Testing get by single ID for OceanBase client")
            results = collection.get(ids=inserted_ids[0])
            assert results is not None
            assert len(results) == 1
            print(f"   Found {len(results)} result for ID={inserted_ids[0]}")
            
            # Test 2: Get by multiple IDs
            print(f"✅ Testing get by multiple IDs")
            if len(inserted_ids) >= 3:
                results = collection.get(ids=inserted_ids[:3])
                assert results is not None
                assert len(results) <= 3
                print(f"   Found {len(results)} results for IDs={inserted_ids[:3]}")
            
            # Test 3: Get by metadata filter
            print(f"✅ Testing get with metadata filter")
            results = collection.get(
                where={"category": {"$eq": "AI"}},
                limit=10
            )
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} results with category='AI'")
            
            # Test 4: Get by logical operators ($or)
            print(f"✅ Testing get with logical operators ($or)")
            results = collection.get(
                where={
                    "$or": [
                        {"category": {"$eq": "AI"}},
                        {"tag": {"$eq": "python"}}
                    ]
                },
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results with $or condition")
            
            # Test 5: Get by document filter
            print(f"✅ Testing get with document filter")
            results = collection.get(
                where_document={"$contains": "machine"},
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results containing 'machine'")
            
            # Test 6: Get with combined filters (where + where_document)
            print(f"✅ Testing get with combined filters")
            results = collection.get(
                where={"category": {"$eq": "AI"}},
                where_document={"$contains": "machine"},
                limit=10
            )
            assert results is not None
            print(f"   Found {len(results)} results matching all filters")
            
            # Test 7: Get with limit and offset
            print(f"✅ Testing get with limit and offset")
            results = collection.get(limit=2, offset=2)
            assert results is not None
            assert len(results) <= 2
            print(f"   Found {len(results)} results (limit=2, offset=2)")
            
            # Test 8: Get all data without filters
            print(f"✅ Testing get all data without filters")
            results = collection.get(limit=100)
            assert results is not None
            assert len(results) > 0
            print(f"   Found {len(results)} total results")
            
            # Test 9: Get with include parameter
            print(f"✅ Testing get with include parameter")
            results = collection.get(
                ids=inserted_ids[:2],
                include=["documents", "metadatas", "embeddings"]
            )
            assert results is not None
            if len(results) > 0:
                for item in results:
                    result_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                    assert '_id' in result_dict
                    print(f"   Result keys: {list(result_dict.keys())}")
            
        finally:
            # Cleanup
            self._cleanup_collection(client, collection_name)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SeekDBClient - Collection Get Tests")
    print("="*60)
    print(f"\nEnvironment Variable Configuration:")
    print(f"  Embedded mode: path={SEEKDB_PATH}, database={SEEKDB_DATABASE}")
    print(f"  Server mode: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}/{SERVER_DATABASE}")
    print(f"  OceanBase mode: {OB_USER}@{OB_TENANT} -> {OB_HOST}:{OB_PORT}/{OB_DATABASE}")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "-s"])

