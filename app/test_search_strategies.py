"""
Quick test script to compare different deduplication strategies.
Run this to see how the different approaches affect your search results.
"""

import requests
import json
import base64
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust to your FastAPI server
TEST_IMAGE_PATH = "path/to/your/test/image.jpg"  # Replace with actual image path

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def test_search_strategy(base64_image: str, strategy: str, n_results: int = 50):
    """Test a specific search strategy."""
    print(f"\n🔍 Testing strategy: {strategy}")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/product/related-by-image-enhanced",
            json={
                "base64_image": base64_image,
                "deduplication_strategy": strategy,
                "n_results": n_results
            },
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Got {len(results)} results")
            
            # Show top 10 results
            for i, product_id in enumerate(results[:10]):
                print(f"   {i+1:2d}. Product {product_id}")
            
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more")
                
            return results
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None

def compare_strategies(base64_image: str):
    """Compare all three deduplication strategies."""
    print("=" * 80)
    print("IMAGE SEARCH STRATEGY COMPARISON")
    print("=" * 80)
    
    strategies = ["first_occurrence", "best_match", "no_deduplication"]
    results = {}
    
    for strategy in strategies:
        results[strategy] = test_search_strategy(base64_image, strategy)
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    for strategy, result_list in results.items():
        if result_list:
            print(f"{strategy:20s}: {len(result_list):3d} results")
            if result_list:
                print(f"{'':20s}  Top 3: {result_list[:3]}")
        else:
            print(f"{strategy:20s}: FAILED")
    
    # Check for differences in top results
    if all(results.values()):
        print(f"\n🔍 TOP 5 COMPARISON:")
        print(f"{'Rank':4s} | {'First Occurrence':15s} | {'Best Match':15s} | {'No Dedup':15s}")
        print("-" * 70)
        
        max_len = max(len(r) for r in results.values() if r)
        for i in range(min(5, max_len)):
            row = f"{i+1:4d} |"
            for strategy in strategies:
                if results[strategy] and i < len(results[strategy]):
                    product_id = results[strategy][i]
                    row += f" {product_id:13s} |"
                else:
                    row += f" {'N/A':13s} |"
            print(row)
    
    return results

def main():
    print("🚀 Image Search Strategy Tester")
    
    # Option 1: Use an actual image file
    if Path(TEST_IMAGE_PATH).exists():
        print(f"📷 Using image: {TEST_IMAGE_PATH}")
        base64_image = encode_image_to_base64(TEST_IMAGE_PATH)
        if base64_image:
            compare_strategies(base64_image)
    else:
        print(f"⚠️  Test image not found: {TEST_IMAGE_PATH}")
        print("\n📝 To use this script:")
        print("1. Update TEST_IMAGE_PATH to point to an actual image file")
        print("2. Make sure your FastAPI server is running")
        print("3. Update API_BASE_URL if needed")
        print("\n💡 Alternatively, you can test manually:")
        print("   curl -X POST 'http://localhost:8000/product/related-by-image-enhanced' \\")
        print("        -H 'Content-Type: application/json' \\")
        print("        -d '{\"base64_image\": \"<your_base64_here>\", \"deduplication_strategy\": \"best_match\"}'")

if __name__ == "__main__":
    main()
