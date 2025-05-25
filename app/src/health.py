"""
Advanced health check module for the multimodal AI application
"""
import asyncio
import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class HealthStatus(BaseModel):
    status: str
    timestamp: float
    uptime_seconds: float
    models_loaded: bool
    database_connected: bool
    memory_usage_mb: float
    last_request_time: float

class HealthChecker:
    def __init__(self):
        self.start_time = time.time()
        self.models_loaded = False
        self.database_connected = False
        self.last_request_time = time.time()
        
    def update_models_status(self, loaded: bool):
        """Update the models loaded status"""
        self.models_loaded = loaded
        logger.info(f"Models status updated: {loaded}")
    
    def update_database_status(self, connected: bool):
        """Update the database connection status"""
        self.database_connected = connected
        logger.info(f"Database status updated: {connected}")
    
    def update_last_request_time(self):
        """Update the last request time"""
        self.last_request_time = time.time()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_health_status(self) -> HealthStatus:
        """Get comprehensive health status"""
        current_time = time.time()
        return HealthStatus(
            status="healthy" if self.models_loaded and self.database_connected else "degraded",
            timestamp=current_time,
            uptime_seconds=current_time - self.start_time,
            models_loaded=self.models_loaded,
            database_connected=self.database_connected,
            memory_usage_mb=self.get_memory_usage(),
            last_request_time=self.last_request_time
        )
    
    def is_ready(self) -> bool:
        """Check if the application is ready to serve requests"""
        return self.models_loaded and self.database_connected

# Global health checker instance
health_checker = HealthChecker()

def add_health_routes(app: FastAPI):
    """Add health check routes to the FastAPI app"""
    
    @app.get("/health")
    async def health_check():
        """Basic health check - always returns if the service is running"""
        return {"status": "healthy", "service": "multimodal-ai-app"}
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with system information"""
        return health_checker.get_health_status()
    
    @app.get("/ready")
    async def readiness_check():
        """Readiness check - returns 503 if not ready to serve traffic"""
        if health_checker.is_ready():
            return {"status": "ready", "models_loaded": True, "database_connected": True}
        else:
            raise HTTPException(
                status_code=503, 
                detail={
                    "status": "not_ready", 
                    "models_loaded": health_checker.models_loaded,
                    "database_connected": health_checker.database_connected
                }
            )
    
    @app.get("/startup")
    async def startup_check():
        """Startup check - used by startup probe"""
        uptime = time.time() - health_checker.start_time
        if uptime > 30:  # After 30 seconds, consider startup complete
            return {"status": "started", "uptime_seconds": uptime}
        else:
            raise HTTPException(
                status_code=503,
                detail={"status": "starting", "uptime_seconds": uptime}
            )
