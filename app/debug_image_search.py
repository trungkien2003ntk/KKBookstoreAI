#!/usr/bin/env python3
"""
Debug script to test image search ordering and help identify issues.
Run this script to test your image search with detailed logging.
"""

import asyncio
import base64
import logging
from src.services.service import ServiceManager
from src.dependencies.service_dependency import get_service

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_image_search_ordering():
    """Test image search with detailed analysis"""
    
    # Get service manager
    service = next(get_service())
    
    print("=" * 80)
    print("IMAGE SEARCH ORDERING DEBUG TEST")
    print("=" * 80)
    
    # Test 1: Get a sample image from the database to test with
    print("\n1. Getting sample images from database...")
    sample_results = service.image_collection.get(limit=5, include=["metadatas", "embeddings"])
    
    if not sample_results["metadatas"]:
        print("❌ No images found in database!")
        return
    
    print(f"✅ Found {len(sample_results['metadatas'])} sample images")
    for i, metadata in enumerate(sample_results["metadatas"][:3]):
        print(f"   Sample {i+1}: Product {metadata.get('product_id')}, Image {metadata.get('image_id')}")
    
    # Test 2: Use the first sample's embedding to search (should return itself as #1)
    print("\n2. Testing with known embedding...")
    test_embedding = sample_results["embeddings"][0]
    test_metadata = sample_results["metadatas"][0]
    
    print(f"🔍 Searching with embedding from Product {test_metadata.get('product_id')}, Image {test_metadata.get('image_id')}")
    
    # Direct ChromaDB query
    direct_results = service.image_collection.query(
        query_embeddings=[test_embedding],
        n_results=10,
        include=["metadatas", "distances", "documents"]
    )
    
    print("\n📊 Direct ChromaDB Results (Top 10):")
    print("Rank | Product ID | Image ID | Distance")
    print("-" * 50)
    
    for i in range(min(10, len(direct_results["metadatas"][0]))):
        metadata = direct_results["metadatas"][0][i]
        distance = direct_results["distances"][0][i] if direct_results["distances"] else "N/A"
        product_id = metadata.get("product_id", "N/A")
        image_id = metadata.get("image_id", "N/A")
        
        # Highlight if this is the original image
        highlight = "🎯" if image_id == test_metadata.get("image_id") else "  "
        print(f"{i+1:4d} | {product_id:10s} | {image_id:20s} | {distance:8.6f} {highlight}")
    
    # Test 3: Test with your service's search method (if you have a base64 image)
    print("\n3. Testing service search method...")
    
    # Create a dummy base64 image for testing the full pipeline
    # Note: In real testing, you'd use an actual image from your dataset
    try:
        # Try to find an actual image file or create a minimal test
        print("   (Skipping base64 test - would need actual image file)")
        
    except Exception as e:
        print(f"   ⚠️  Could not test full pipeline: {e}")
    
    # Test 4: Analyze deduplication impact
    print("\n4. Analyzing deduplication impact...")
    
    # Group results by product_id to see duplicates
    from collections import defaultdict
    product_groups = defaultdict(list)
    
    for i, metadata in enumerate(direct_results["metadatas"][0]):
        product_id = metadata.get("product_id")
        distance = direct_results["distances"][0][i] if direct_results["distances"] else float('inf')
        image_id = metadata.get("image_id")
        
        product_groups[product_id].append({
            "rank": i + 1,
            "image_id": image_id,
            "distance": distance
        })
    
    print("\n📈 Products with multiple images in top 10:")
    for product_id, images in product_groups.items():
        if len(images) > 1:
            print(f"   Product {product_id}: {len(images)} images")
            for img in images:
                print(f"      Rank {img['rank']:2d}: {img['image_id']} (distance: {img['distance']:.6f})")
            
            # Show which would be selected by current deduplication
            first_occurrence = min(images, key=lambda x: x['rank'])
            best_match = min(images, key=lambda x: x['distance'])
            
            if first_occurrence != best_match:
                print(f"      ⚠️  First occurrence (rank {first_occurrence['rank']}) != Best match (rank {best_match['rank']})")
            else:
                print(f"      ✅ First occurrence IS the best match")
    
    # Test 5: Recommendations
    print("\n5. Recommendations:")
    
    original_rank = None
    for i, metadata in enumerate(direct_results["metadatas"][0]):
        if metadata.get("image_id") == test_metadata.get("image_id"):
            original_rank = i + 1
            break
    
    if original_rank == 1:
        print("   ✅ Original image ranks #1 - ChromaDB ordering is correct")
        print("   🔍 Issue is likely in your deduplication logic")
    elif original_rank and original_rank <= 10:
        print(f"   ⚠️  Original image ranks #{original_rank} - Check embedding quality")
        print("   🔍 Possible issues: model weights, preprocessing, or database corruption")
    else:
        print("   ❌ Original image not in top 10 - Serious embedding issue")
        print("   🔍 Check: model loading, preprocessing pipeline, database integrity")
    
    print("\n" + "=" * 80)
    return direct_results

async def test_with_actual_base64_image(base64_image_string: str):
    """Test with an actual base64 image string"""
    service = next(get_service())
    
    print("\n" + "=" * 80)
    print("TESTING WITH ACTUAL BASE64 IMAGE")
    print("=" * 80)
    
    try:
        # Test the full pipeline
        results = await service.search_service.search_by_image_embedding(
            base64_image=base64_image_string,
            n_results=20
        )
        
        print(f"🔍 Service returned {len(results)} product IDs:")
        for i, product_id in enumerate(results[:10]):
            print(f"   {i+1:2d}. Product {product_id}")
            
    except Exception as e:
        print(f"❌ Error in service search: {e}")
        logger.exception("Full error details:")

if __name__ == "__main__":
    print("🚀 Starting image search debug test...")
    
    try:
        asyncio.run(test_image_search_ordering())
        
        # If you have a specific base64 image to test, uncomment and provide it:
        # test_base64 = "your_base64_image_string_here"
        # asyncio.run(test_with_actual_base64_image(test_base64))
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Full error details:")
    
    print("\n✅ Debug test completed!")
