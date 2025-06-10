# Image Search Ordering Analysis & Solutions

## Problem Identified

Your image search is not returning the exact same image as the top result due to **deduplication logic that breaks ChromaDB's distance-based ordering**.

### Root Causes:

1. **Deduplication Breaking Order**: When you deduplicate by `product_id`, you take the first occurrence of each product, not necessarily the most similar image.

2. **Multiple Images Per Product**: You have multiple images per product (e.g., `img_957_10233`, `img_957_10239` for product `957`), and ChromaDB orders by embedding similarity, but deduplication picks the first occurrence.

3. **Distance Information Loss**: The crucial similarity distances from ChromaDB are discarded.

## Solutions Implemented

### Current Fix (Recommended)

The updated `search_by_image_embedding` method now:

-   **Logs top 5 results** with distances for debugging
-   **Preserves ChromaDB ordering** while deduplicating
-   **Tracks which image was selected** for each product

### Alternative Solutions

#### Option 1: No Deduplication (Exact ChromaDB Order)

```python
# In search_by_image_embedding method, replace the deduplication logic with:
return [metadata.get("product_id") for metadata in metadatas[0] if metadata.get("product_id")]
```

#### Option 2: Distance-Based Deduplication

```python
# Group by product_id and keep the best (lowest distance) match per product
from collections import defaultdict

product_scores = defaultdict(lambda: {"distance": float('inf'), "product_id": None})
for i, (metadata, distance) in enumerate(zip(metadatas[0], distances[0])):
    pid = metadata.get("product_id")
    if pid and distance < product_scores[pid]["distance"]:
        product_scores[pid] = {"distance": distance, "product_id": pid}

# Sort by best distance per product
sorted_products = sorted(product_scores.values(), key=lambda x: x["distance"])
return [item["product_id"] for item in sorted_products]
```

## Testing Recommendations

1. **Run a test search** with an image you know exists in the database
2. **Check the logs** to see the distance values and which images are being selected
3. **Compare results** with and without deduplication
4. **Verify** that the exact same image gets distance `0.0` or very close to it

## Expected Behavior After Fix

-   When you search with an exact image from your database, it should appear as position 1 with distance ≈ 0.0
-   The ordering should now reflect actual embedding similarity rather than arbitrary first-occurrence during deduplication
-   You'll have better visibility into what's happening through the debug logs

## Configuration Notes

Your `.env` file shows `IMAGE_N_RESULTS=200`, which is good for getting comprehensive results. The logging will help you understand if the issue is in the embedding generation, ChromaDB querying, or the deduplication logic.
