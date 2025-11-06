"""
Example usage of collection query functionality

This example demonstrates:
1. Vector similarity query with default n_results (10)
2. Vector similarity query with custom n_results
3. Query with metadata filters (comparison operators)
4. Query with document filters (full-text search and regex)
5. Query with combined filters
6. Specifying return fields with include parameter
"""
import json
from seekdbclient import Client


def example_basic_query():
    """Example 1: Basic vector similarity query"""
    print("=" * 60)
    print("Example 1: Basic vector similarity query")
    print("=" * 60)
    
    # Create client (example using SeekDB server mode)
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    
    # Get collection
    collection = client.get_collection("my_collection")
    
    # Query with single vector, default n_results=10
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1]
    )
    
    print(f"Found {len(results)} results")
    print(json.dumps(results, indent=2, ensure_ascii=False))


def example_multiple_vectors_query():
    """Example 2: Query with multiple vectors"""
    print("\n" + "=" * 60)
    print("Example 2: Query with multiple vectors")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Query with multiple vectors, custom n_results
    results = collection.query(
        query_embeddings=[
            [11.1, 12.1, 13.1],
            [1.1, 2.3, 3.2]
        ],
        n_results=5
    )
    
    print(f"Found {len(results)} results")
    for result in results:
        print(f"ID: {result['_id']}, Distance: {result.get('distance', 'N/A')}")


def example_metadata_filters():
    """Example 3: Query with metadata filters"""
    print("\n" + "=" * 60)
    print("Example 3: Query with metadata filters")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Filter with comparison operators
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        where={
            "chapter": {"$gte": 3},  # chapter >= 3
            "verse": {"$lt": 20}     # verse < 20
        },
        n_results=10
    )
    
    print(f"Found {len(results)} results with chapter >= 3 and verse < 20")
    for result in results:
        metadata = result.get('metadata', {})
        print(f"ID: {result['_id']}, Chapter: {metadata.get('chapter')}, Verse: {metadata.get('verse')}")


def example_logical_operators():
    """Example 4: Query with logical operators"""
    print("\n" + "=" * 60)
    print("Example 4: Query with logical operators ($and, $or)")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Using $or operator
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        where={
            "$or": [
                {"color": "pink_8682"},
                {"chapter": {"$eq": 3}}
            ]
        },
        n_results=10
    )
    
    print(f"Found {len(results)} results with color='pink_8682' OR chapter=3")


def example_in_operator():
    """Example 5: Query with $in and $nin operators"""
    print("\n" + "=" * 60)
    print("Example 5: Query with $in and $nin operators")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Using $in operator
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        where={
            "chapter": {"$in": [1, 2, 3, 4, 5]}
        },
        n_results=10
    )
    
    print(f"Found {len(results)} results with chapter in [1,2,3,4,5]")


def example_document_filters():
    """Example 6: Query with document filters"""
    print("\n" + "=" * 60)
    print("Example 6: Query with document filters")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Full-text search using $contains
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        where_document={"$contains": "machine learning"},
        n_results=10
    )
    
    print(f"Found {len(results)} results containing 'machine learning'")
    
    # Regex search
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        where_document={"$regex": "^hello.*world$"},
        n_results=10
    )
    
    print(f"Found {len(results)} results matching regex '^hello.*world$'")


def example_combined_filters():
    """Example 7: Query with combined metadata and document filters"""
    print("\n" + "=" * 60)
    print("Example 7: Combined metadata and document filters")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Combine metadata and document filters
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        where={
            "chapter": {"$gte": 3},
            "color": {"$ne": "blue"}
        },
        where_document={"$contains": "python"},
        n_results=10
    )
    
    print(f"Found {len(results)} results matching all filters")
    for result in results:
        print(f"ID: {result['_id']}, Document: {result.get('document', 'N/A')[:50]}...")


def example_include_fields():
    """Example 8: Specify return fields with include parameter"""
    print("\n" + "=" * 60)
    print("Example 8: Specify return fields")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    # Include only specific fields
    results = collection.query(
        query_embeddings=[11.1, 12.1, 13.1],
        include=["documents", "metadatas", "embeddings"],
        n_results=5
    )
    
    print(f"Found {len(results)} results")
    print("Result structure:")
    if results:
        print(f"Keys in result: {list(results[0].keys())}")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))


def example_query_texts():
    """Example 9: Query with texts (will be embedded automatically)"""
    print("\n" + "=" * 60)
    print("Example 9: Query with texts (requires embedding implementation)")
    print("=" * 60)
    
    client = Client(mode="seekdb_server", host="localhost", port=2882)
    collection = client.get_collection("my_collection")
    
    try:
        # Note: This will raise NotImplementedError until embedding is implemented
        results = collection.query(
            query_texts=["my query text"],
            n_results=10
        )
        
        print(f"Found {len(results)} results")
    except NotImplementedError as e:
        print(f"Text embedding not yet implemented: {e}")
        print("Please use query_embeddings directly for now.")


def main():
    """Run all examples"""
    print("Collection Query Examples")
    print("=" * 60)
    print()
    
    # Note: These examples assume you have a running database and collection
    # You may need to adjust connection parameters and collection name
    
    try:
        example_basic_query()
        example_multiple_vectors_query()
        example_metadata_filters()
        example_logical_operators()
        example_in_operator()
        example_document_filters()
        example_combined_filters()
        example_include_fields()
        example_query_texts()
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure your database is running and collection exists.")


if __name__ == "__main__":
    main()

