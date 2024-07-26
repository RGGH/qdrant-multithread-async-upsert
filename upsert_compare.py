import asyncio
import numpy as np
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Range
from time import time

async def main():
    # Initialize async client
    client = AsyncQdrantClient(url="http://localhost:6333")

    collection_name = "test"

    # Recreate collection
    await client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=100, distance=Distance.COSINE),
    )

    # Create a complex payload
    payload = {
        "foo": "Bar",
        "bar": 12,
        "baz": {
            "qux": "quux",
            "nested": {
                "level1": {
                    "level2": "data",
                    "array": [1, 2, 3, 4, 5]
                }
            }
        }
    }

    # Generate multiple points
    num_points = 10000
    vector_size = 100
    vectors = np.random.rand(num_points, vector_size)
    points = [
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload=payload
        )
        for idx, vector in enumerate(vectors)
    ]

    # Parallel upload
    start_parallel = time()
    
    chunk_size = 100
    tasks = [
        client.upsert(
            collection_name=collection_name,
            points=points[i:i + chunk_size]
        )
        for i in range(0, len(points), chunk_size)
    ]
    await asyncio.gather(*tasks)

    duration_parallel = time() - start_parallel

    # Sequential upload
    start_sequential = time()
    
    for i in range(0, len(points), chunk_size):
        await client.upsert(
            collection_name=collection_name,
            points=points[i:i + chunk_size]
        )
    
    duration_sequential = time() - start_sequential

    # Perform a search query
    query_vector = np.random.rand(vector_size).tolist()
    search_result = await client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=Filter(
            must=[FieldCondition(
                key='bar',
                range=Range(gte=12)
            )]
        ),
        limit=10,
    )
    print(search_result)

    # Print durations
    print(f"Parallel upload took: {duration_parallel:.2f} seconds")
    print(f"Sequential upload took: {duration_sequential:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())

