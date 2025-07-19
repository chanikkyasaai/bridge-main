"""
Main entry point for the Behavioral Authentication ML Engine
This serves as the primary entry point for production deployments
"""

import uvicorn
from ml_engine_api_service import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,  # Production mode
        log_level="info"
    )
