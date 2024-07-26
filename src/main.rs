use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Distance, Filter, PointStruct, ScalarQuantizationBuilder,
    SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use futures::future::join_all;
use std::sync::Arc;
use thiserror::Error;
use tokio::task::JoinError;
use tokio::time::Instant;

#[derive(Error, Debug)]
enum CustomError {
    #[error("Qdrant error: {0}")]
    Qdrant(#[from] QdrantError),
    #[error("Join error: {0}")]
    Join(#[from] JoinError),
}

#[tokio::main]
async fn main() -> Result<(), CustomError> {
    // Example of top level client
    let client = Arc::new(Qdrant::from_url("http://localhost:6334").build()?);

    let collections_list = client.list_collections().await?;
    dbg!(collections_list);

    let collection_name = "test";
    client.delete_collection(collection_name).await?;

    client
        .create_collection(
            CreateCollectionBuilder::new(collection_name)
                .vectors_config(VectorParamsBuilder::new(100, Distance::Cosine))
                .quantization_config(ScalarQuantizationBuilder::default()),
        )
        .await?;

    let collection_info = client.collection_info(collection_name).await?;
    dbg!(collection_info);

    let payload: Payload = serde_json::json!(
        {
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
    )
    .try_into()
    .unwrap();

    // Create multiple points
    let mut points = Vec::new();
    for i in 0..10_000 {  // Example with 10,000 points
        points.push(PointStruct::new(i, vec![12.; 100], payload.clone()));
    }

    // Measure time for parallel upload
    let start_parallel = Instant::now();

    // Split points into chunks and upsert them in parallel
    let chunk_size = 100;
    let mut tasks = Vec::new();
    for chunk in points.chunks(chunk_size) {
        let client_clone = Arc::clone(&client);
        let points_chunk = chunk.to_vec();
        let collection_name = collection_name.to_string();
        // Spawn a new task for each chunk to run concurrently
        tasks.push(tokio::spawn(async move {
            client_clone
                .upsert_points(UpsertPointsBuilder::new(&collection_name, points_chunk))
                .await
        }));
    }

    // Wait for all tasks to complete
    let results = join_all(tasks).await;
    for result in results {
        result??; // Convert JoinError to CustomError and propagate QdrantError if any
    }

    let duration_parallel = start_parallel.elapsed();

    // Sequential upload for comparison
    let start_sequential = Instant::now();
    for chunk in points.chunks(chunk_size) {
        client
            .upsert_points(UpsertPointsBuilder::new(collection_name, chunk.to_vec()))
            .await?;
    }

    let duration_sequential = start_sequential.elapsed();

    let search_result = client
        .search_points(
            SearchPointsBuilder::new(collection_name, [11.; 100], 10)
                .filter(Filter::all([Condition::matches("bar", 12)]))
                .with_payload(true)
                .params(SearchParamsBuilder::default().exact(true)),
        )
        .await?;
    dbg!(&search_result);

    let found_point = search_result.result.into_iter().next().unwrap();
    let mut payload = found_point.payload;
    let baz_payload = payload.remove("baz").unwrap().into_json();
    println!("search - baz: {}", baz_payload);

    // Print durations at the end
    println!("-----------------------------------");
    println!("Parallel upload took: {:?}", duration_parallel);
    println!("Sequential upload took: {:?}", duration_sequential);

    Ok(())
}

