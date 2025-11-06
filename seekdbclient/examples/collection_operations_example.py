"""
Collection Operations Example
Demonstrates a complete workflow: create, add, update, upsert, query, and delete operations
"""
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import seekdbclient


def collection_operations_example():
    """Demonstrate collection operations: create, add, update, upsert, query, delete"""
    print("\n" + "=" * 60)
    print("Collection Operations Demonstration")
    print("=" * 60)

    client = seekdbclient.Client(
        path="./seekdb_store",
        database="test"
    )

    collection_name = "workflow_example"
    dimension = 64

    try:
        # Step 1: Create collection
        print("\n--- Step 1: Create Collection ---")
        if client.has_collection(collection_name):
            client.delete_collection(collection_name)
            print(f"✅ Deleted existing collection: {collection_name}")

        collection = client.create_collection(
            name=collection_name,
            dimension=dimension
        )
        print(f"✅ Created collection: {collection_name}")

        # Step 2: Add initial data
        print("\n--- Step 2: Add Initial Data ---")
        collection.add(
            ids=["item1", "item2", "item3"],
            vectors=[
                [0.1] * dimension,
                [0.2] * dimension,
                [0.3] * dimension
            ],
            documents=[
                "The latest smartphone features a high-resolution display and advanced camera system with AI-powered photography.",
                "This laptop offers exceptional performance with the newest processor and long-lasting battery life for productivity.",
                "Our cloud computing service provides scalable infrastructure with 99.9% uptime guarantee and 24/7 support."
            ],
            metadatas=[
                {"type": "product", "price": 10.0},
                {"type": "product", "price": 20.0},
                {"type": "service", "price": 30.0}
            ]
        )
        print("✅ Added 3 items")

        # Step 3: Update some items
        print("\n--- Step 3: Update Items ---")
        collection.update(
            ids="item1",
            metadatas={"type": "product", "price": 15.0, "discount": 0.1}
        )
        print("✅ Updated item1 with new price and discount")

        # Step 4: Upsert new item
        print("\n--- Step 4: Upsert New Item ---")
        collection.upsert(
            ids="item4",
            vectors=[[0.4] * dimension],
            documents="Premium wireless headphones with noise cancellation technology for immersive music experience.",
            metadatas={"type": "product", "price": 40.0}
        )
        print("✅ Upserted item4 (new item)")

        # Step 5: Query items
        print("\n--- Step 5: Query Items ---")
        results = collection.get(
            ids=["item1", "item2", "item3", "item4"],
            include=["documents", "metadatas"]
        )
        print(f"✅ Retrieved {len(results.get('ids', []))} items")
        for i, doc_id in enumerate(results.get('ids', [])):
            doc = results['documents'][i] if results.get('documents') else None
            meta = results['metadatas'][i] if results.get('metadatas') else None
            print(f"   ID: {doc_id}, Document: {doc}, Metadata: {meta}")

        # Step 6: Delete items
        print("\n--- Step 6: Delete Items ---")
        collection.delete(ids=["item3", "item4"])
        print("✅ Deleted item3 and item4")

        # Step 7: Clean up
        print("\n--- Step 7: Clean Up ---")
        client.delete_collection(collection_name)
        print(f"✅ Deleted collection: {collection_name}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run complete workflow example"""
    print("\n" + "=" * 60)
    print("Collection Operations Example")
    print("=" * 60)
    print("\nThis example demonstrates a complete workflow:")
    print("  - Create collection")
    print("  - ADD data to collection")
    print("  - UPDATE existing data")
    print("  - UPSERT data (insert or update)")
    print("  - Query data")
    print("  - DELETE data")
    print("  - Clean up")

    # Run collection operations demonstration
    collection_operations_example()

    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
