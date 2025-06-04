from setup_local_db import embedding_func, create_lancedb_table

# Test embedding generation
print("Testing embeddings...")
test_vector = embedding_func("test query")
print(f"Vector dimension: {len(test_vector)}")

# Test table creation
print("\nTesting table creation...")
table = create_lancedb_table("./test_db", "test_table")
print("Table created successfully!")
table.add([{"id": "1", "text": "test document"}])
print("Document added successfully!")

# Test search
results = table.search("test").limit(1).to_list()
print("\nSearch results:", results)