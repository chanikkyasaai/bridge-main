"""
BRIDGE ML-Engine Startup Script
Initializes and starts the complete ML-Engine system
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Optional
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ml_engine.config import CONFIG
from ml_engine import ml_engine
from api_integration import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bridge_ml_engine.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BRIDGEMLEngineService:
    """Main service class for BRIDGE ML-Engine"""
    
    def __init__(self):
        self.ml_engine = ml_engine
        self.api_server = None
        self.shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize all ML-Engine components"""
        logger.info("🚀 Starting BRIDGE ML-Engine initialization...")
        
        try:
            # Initialize ML Engine
            logger.info("Initializing ML Engine components...")
            await self.ml_engine.initialize()
            
            logger.info("✅ ML-Engine initialization completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML-Engine initialization failed: {e}")
            return False
    
    async def start_api_server(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the API server"""
        logger.info(f"🌐 Starting API server on {host}:{port}")
        
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        
        try:
            await server.serve()
        except Exception as e:
            logger.error(f"❌ API server error: {e}")
            raise
    
    async def run_health_monitor(self):
        """Run periodic health monitoring"""
        logger.info("🏥 Starting health monitor...")
        
        while not self.shutdown_event.is_set():
            try:
                # Check ML Engine health
                health = await self.ml_engine.health_check()
                
                if health["status"] != "healthy":
                    logger.warning(f"⚠️ ML Engine health check: {health['status']}")
                
                # Log statistics periodically
                stats = self.ml_engine.get_engine_stats()
                logger.info(
                    f"📊 Stats: {stats.total_requests} requests, "
                    f"{stats.successful_authentications} successful, "
                    f"{stats.blocked_sessions} blocked, "
                    f"{stats.average_processing_time:.2f}ms avg"
                )
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(30)  # Shorter wait on error
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down BRIDGE ML-Engine...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Shutdown ML Engine
        await self.ml_engine.shutdown()
        
        logger.info("✅ BRIDGE ML-Engine shutdown completed")
    
    async def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the complete ML-Engine service"""
        
        # Initialize
        if not await self.initialize():
            logger.error("❌ Failed to initialize ML-Engine")
            return False
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start background services
        tasks = [
            asyncio.create_task(self.start_api_server(host, port)),
            asyncio.create_task(self.run_health_monitor())
        ]
        
        try:
            # Wait for shutdown signal
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("🔄 Received interrupt signal")
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
        finally:
            # Cleanup
            await self.shutdown()
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return True

async def main():
    """Main entry point"""
    
    # Print startup banner
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗                              ║
║  ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝                              ║
║  ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗                                ║
║  ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝                                ║
║  ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗                              ║
║  ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝                              ║
║                                                                              ║
║  Behavioral Risk Intelligence for Dynamic Guarded Entry                     ║
║  ML-Engine for Continuous Behavioral Authentication                         ║
║                                                                              ║
║  🏆 SuRaksha Cyber Hackathon - Team "Five"                                  ║
║  📱 Mobile Banking Security Enhancement                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Environment setup
    logger.info("🔧 Environment Setup:")
    logger.info(f"   Python: {sys.version}")
    logger.info(f"   Working Directory: {os.getcwd()}")
    logger.info(f"   Config Environment: {os.getenv('BRIDGE_ENV', 'development')}")
    logger.info(f"   Model Path: {CONFIG.MODEL_BASE_PATH}")
    logger.info(f"   Max Concurrent Sessions: {CONFIG.MAX_CONCURRENT_SESSIONS}")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="BRIDGE ML-Engine Service")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8001, help="API server port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Set environment
    if args.dev:
        os.environ["BRIDGE_ENV"] = "development"
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create and run service
    service = BRIDGEMLEngineService()
    
    logger.info("🚀 Starting BRIDGE ML-Engine Service...")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   Log Level: {args.log_level}")
    
    success = await service.run(args.host, args.port)
    
    if success:
        logger.info("✅ BRIDGE ML-Engine Service completed successfully")
        return 0
    else:
        logger.error("❌ BRIDGE ML-Engine Service failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🔄 Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
