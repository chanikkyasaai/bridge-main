"""
BRIDGE ML-Engine Initialization Script
Sets up models, indexes, and initial configurations
"""

import asyncio
import logging
import os
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlengine.config import CONFIG
from mlengine.utils.behavioral_vectors import BehavioralVectorProcessor
from mlengine.adapters.faiss.verifier.layer1_verifier import FAISSVerifier
from mlengine.adapters.level2.layer2_verifier import Layer2Verifier
from mlengine.core.drift_detection import BehavioralDriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_ml_engine():
    """Initialize all ML-Engine components"""
    
    logger.info("🚀 Initializing BRIDGE ML-Engine...")
    
    # Create model directories
    model_dirs = [
        CONFIG.MODEL_BASE_PATH,
        CONFIG.FAISS_INDEX_PATH,
        CONFIG.TRANSFORMER_MODEL_PATH,
        CONFIG.GNN_MODEL_PATH
    ]
    
    for model_dir in model_dirs:
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"📁 Created model directory: {model_dir}")
    
    # Initialize components
    logger.info("🧠 Initializing Behavioral Vector Processor...")
    vector_processor = BehavioralVectorProcessor()
    await vector_processor.initialize()
    
    logger.info("🔍 Initializing FAISS Verifier...")
    faiss_verifier = FAISSVerifier()
    await faiss_verifier.initialize()
    
    logger.info("🤖 Initializing Layer 2 Verifier...")
    l2_verifier = Layer2Verifier()
    await l2_verifier.initialize()
    
    logger.info("📊 Initializing Drift Detector...")
    drift_detector = BehavioralDriftDetector()
    await drift_detector.initialize()
    
    logger.info("✅ BRIDGE ML-Engine initialization completed!")
    
    return {
        "vector_processor": vector_processor,
        "faiss_verifier": faiss_verifier,
        "l2_verifier": l2_verifier,
        "drift_detector": drift_detector
    }

if __name__ == "__main__":
    asyncio.run(initialize_ml_engine())
