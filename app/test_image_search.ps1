# PowerShell script to test image search ordering
# Run this script to debug your image search issues

Write-Host "🚀 Image Search Debug Helper" -ForegroundColor Green
Write-Host "=" * 50

# Check if the FastAPI server is running
Write-Host "`n📡 Checking if FastAPI server is running..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    Write-Host "✅ Server is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Server is not running. Please start your FastAPI server first." -ForegroundColor Red
    Write-Host "   Run: python -m uvicorn main:app --reload" -ForegroundColor Cyan
    exit 1
}

# Run the debug script
Write-Host "`n🔍 Running image search debug test..." -ForegroundColor Yellow
try {
    python debug_image_search.py
} catch {
    Write-Host "❌ Failed to run debug script: $_" -ForegroundColor Red
    Write-Host "   Make sure you're in the correct directory and Python dependencies are installed" -ForegroundColor Cyan
}

# Test the enhanced API endpoints
Write-Host "`n🧪 Testing enhanced search API..." -ForegroundColor Yellow

# Create a sample test payload (you'll need to replace with actual base64 image)
$testPayload = @{
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # 1x1 pixel PNG
    deduplication_strategy = "best_match"
    n_results = 20
} | ConvertTo-Json

Write-Host "📝 Sample API test (using dummy 1x1 pixel image):" -ForegroundColor Cyan
Write-Host "POST /product/related-by-image-enhanced" -ForegroundColor Gray

try {
    $headers = @{ "Content-Type" = "application/json" }
    $response = Invoke-RestMethod -Uri "http://localhost:8000/product/related-by-image-enhanced" -Method POST -Body $testPayload -Headers $headers -TimeoutSec 30
    Write-Host "✅ API is working. Got $($response.Length) results" -ForegroundColor Green
    
    if ($response.Length -gt 0) {
        Write-Host "   Top 3 results: $($response[0..2] -join ', ')" -ForegroundColor Gray
    }
} catch {
    $statusCode = $_.Exception.Response.StatusCode
    $errorBody = $_.Exception.Response | ConvertFrom-Json -ErrorAction SilentlyContinue
    
    if ($statusCode -eq 400) {
        Write-Host "⚠️  API returned 400 (expected with dummy image)" -ForegroundColor Yellow
    } else {
        Write-Host "❌ API Error: $statusCode" -ForegroundColor Red
        if ($errorBody) {
            Write-Host "   Error: $($errorBody.detail.message)" -ForegroundColor Red
        }
    }
}

Write-Host "`n📋 Next Steps:" -ForegroundColor Green
Write-Host "1. Test with a real image from your database:" -ForegroundColor White
Write-Host "   - Update test_search_strategies.py with actual image path" -ForegroundColor Gray
Write-Host "   - Run: python test_search_strategies.py" -ForegroundColor Gray

Write-Host "`n2. Compare deduplication strategies:" -ForegroundColor White
Write-Host "   - first_occurrence: Current behavior" -ForegroundColor Gray
Write-Host "   - best_match: Use lowest distance per product" -ForegroundColor Gray
Write-Host "   - no_deduplication: Pure ChromaDB ordering" -ForegroundColor Gray

Write-Host "`n3. Check the logs for detailed similarity scores" -ForegroundColor White

Write-Host "`n4. If exact image still doesn't rank #1:" -ForegroundColor White
Write-Host "   - Check embedding model consistency" -ForegroundColor Gray
Write-Host "   - Verify image preprocessing" -ForegroundColor Gray
Write-Host "   - Validate database integrity" -ForegroundColor Gray

Write-Host "`n✅ Debug complete! Check the output above for issues." -ForegroundColor Green
