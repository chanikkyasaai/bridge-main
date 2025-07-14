"""
BRIDGE ML-Engine Industry-Grade Startup Script

This script starts the complete ML-Engine with all components properly initialized
for industry-grade banking behavioral authentication.

Pipeline Order:
1. Input Validation & Preprocessing
2. Layer 1: FAISS Fast Verification
3. Layer 2: Adaptive Context Analysis  
4. Drift Detection & Profile Adaptation
5. Risk Assessment & Aggregation
6. Policy Decision Engine
7. Response Generation & Logging

Features:
- Session lifecycle management
- Real-time behavioral processing
- Banking compliance & audit trails
- High performance & reliability
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from mlengine.core.industry_engine import IndustryGradeMLEngine
from mlengine.api.session_integration import ml_session_integrator
from mlengine.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'bridge_ml_engine_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class MLEngineStartup:
    """ML-Engine startup and management"""
    
    def __init__(self, config_overrides: dict = None):
        self.config = {**CONFIG, **(config_overrides or {})}
        self.ml_engine = IndustryGradeMLEngine
        self.integrator = ml_session_integrator
        self.is_running = False
        
    async def startup(self) -> bool:
        """Complete ML-Engine startup sequence"""
        try:
            logger.info("🚀 Starting BRIDGE Industry-Grade ML-Engine...")
            logger.info("=" * 60)
            
            # 1. Validate environment
            if not await self._validate_environment():
                logger.error("❌ Environment validation failed")
                return False
            
            # 2. Initialize ML-Engine core
            logger.info("🧠 Initializing ML-Engine core components...")
            if not await self.ml_engine.initialize():
                logger.error("❌ ML-Engine initialization failed")
                return False
            
            # 3. Initialize session integration
            logger.info("🔗 Initializing session lifecycle integration...")
            if not await self.integrator.initialize():
                logger.error("❌ Session integration initialization failed")
                return False
            
            # 4. Register callbacks and handlers
            logger.info("📡 Registering event handlers...")
            await self._register_event_handlers()
            
            # 5. Health check
            logger.info("🏥 Performing health check...")
            if not await self._health_check():
                logger.error("❌ Health check failed")
                return False
            
            self.is_running = True
            logger.info("✅ BRIDGE ML-Engine startup completed successfully!")
            logger.info("=" * 60)
            logger.info(f"🏦 Banking-Grade Behavioral Authentication ACTIVE")
            logger.info(f"🔒 Security Level: Industry-Grade")
            logger.info(f"⚡ Processing Pipeline: 7-Stage Ordered")
            logger.info(f"📊 Session Lifecycle: Fully Integrated")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ML-Engine startup failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Graceful ML-Engine shutdown"""
        try:
            logger.info("🔄 Shutting down BRIDGE ML-Engine...")
            
            # 1. Shutdown session integration
            await self.integrator.shutdown()
            
            # 2. Shutdown ML-Engine core
            await self.ml_engine.shutdown()
            
            self.is_running = False
            logger.info("✅ ML-Engine shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during ML-Engine shutdown: {e}")
            return False
    
    async def run_forever(self):
        """Run ML-Engine until interrupted"""
        try:
            logger.info("🔄 ML-Engine running... Press Ctrl+C to stop")
            while self.is_running:
                await asyncio.sleep(60)  # Health check every minute
                if not await self._periodic_health_check():
                    logger.warning("⚠️ Health check warning detected")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            await self.shutdown()
    
    async def _validate_environment(self) -> bool:
        """Validate ML-Engine environment and dependencies"""
        try:
            # Check required directories
            required_dirs = ['models', 'data', 'logs', 'cache']
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    logger.info(f"📁 Creating directory: {dir_name}")
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            # Check configuration
            required_config = ['FAISS_DIMENSION', 'MAX_CONCURRENT_SESSIONS']
            for key in required_config:
                if key not in self.config:
                    logger.error(f"❌ Missing required config: {key}")
                    return False
            
            # Check model files (would check actual model files in production)
            logger.info("✅ Environment validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation error: {e}")
            return False
    
    async def _register_event_handlers(self):
        """Register event handlers for various ML events"""
        
        async def on_session_start(session_id: str, user_id: str, context):
            logger.info(f"🎯 ML Session started: {session_id} (User: {user_id})")
        
        async def on_session_end(session_id: str, session_data):
            logger.info(f"🏁 ML Session ended: {session_id}")
        
        async def on_authentication(response):
            if response.risk_level.value in ['high', 'critical']:
                logger.warning(f"⚠️ High risk authentication: {response.session_id} - {response.decision.value}")
            else:
                logger.debug(f"✅ Authentication: {response.session_id} - {response.decision.value}")
        
        # Register callbacks
        self.integrator.register_session_start_callback(on_session_start)
        self.integrator.register_session_end_callback(on_session_end)
        self.integrator.register_authentication_callback(on_authentication)
        
        logger.info("✅ Event handlers registered")
    
    async def _health_check(self) -> bool:
        """Comprehensive health check"""
        try:
            # Check ML-Engine status
            stats = await self.ml_engine.get_engine_stats()
            if not stats.get('is_initialized', False):
                logger.error("❌ ML-Engine not properly initialized")
                return False
            
            # Check integration status
            integration_stats = await self.integrator.get_ml_engine_stats()
            if not integration_stats.get('ml_enabled', False):
                logger.error("❌ ML integration not enabled")
                return False
            
            logger.info("✅ Health check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def _periodic_health_check(self) -> bool:
        """Periodic health monitoring"""
        try:
            stats = await self.ml_engine.get_engine_stats()
            
            # Check error rate
            error_rate = stats.get('error_rate', 0)
            if error_rate > 0.05:  # 5% error rate threshold
                logger.warning(f"⚠️ High error rate detected: {error_rate:.2%}")
                return False
            
            # Check processing time
            avg_time = stats.get('average_processing_time_ms', 0)
            if avg_time > 200:  # 200ms threshold
                logger.warning(f"⚠️ High processing time detected: {avg_time:.1f}ms")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Periodic health check error: {e}")
            return False

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='BRIDGE Industry-Grade ML-Engine')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--max-sessions', type=int, default=1000, help='Maximum concurrent sessions')
    parser.add_argument('--backend-host', default='localhost', help='Backend host')
    parser.add_argument('--backend-port', type=int, default=8000, help='Backend port')
    
    args = parser.parse_args()
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Configuration overrides
    config_overrides = {
        'MAX_CONCURRENT_SESSIONS': args.max_sessions,
        'BACKEND_HOST': args.backend_host,
        'BACKEND_PORT': args.backend_port
    }
    
    # Create and start ML-Engine
    startup = MLEngineStartup(config_overrides)
    
    if await startup.startup():
        await startup.run_forever()
    else:
        logger.error("❌ Failed to start ML-Engine")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 ML-Engine stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
