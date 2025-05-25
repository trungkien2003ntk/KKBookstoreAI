"""
Run the code in this file
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

from src.router.search import product_router
from src.health import add_health_routes, health_checker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal AI Search API",
    description="FastAPI application for multimodal product search using text and image embeddings",
    version="1.0.0"
)

@app.middleware("http")
async def update_last_request_time(request: Request, call_next):
    """Middleware to track last request time for health monitoring"""
    health_checker.update_last_request_time()
    response = await call_next(request)
    return response

@app.on_event("startup")
async def startup_event():
    """Startup event to initialize the application"""
    logger.info("Application startup initiated...")
    
    # Initialize health checker
    health_checker.update_database_status(True)  # Assume database is available
    
    # Mark models as loaded after first successful model loading
    # This will be updated when models are actually loaded
    await asyncio.sleep(10)  # Give more time for model initialization
    health_checker.update_models_status(True)
    
    logger.info("Application startup completed")

# Add health check routes
add_health_routes(app)

# app.include_router(root_router)
app.include_router(product_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860
    )
