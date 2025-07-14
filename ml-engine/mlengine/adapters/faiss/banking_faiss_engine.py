"""
Banking FAISS Engine - Layer 1 Fast Similarity Search
Simple wrapper for existing FAISS functionality
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class BankingFAISSEngine:
    """Simple FAISS engine wrapper for banking behavioral verification"""
    
    def __init__(self, vector_dim: int = 64):
        self.vector_dim = vector_dim
        self.is_initialized = False
        self.user_profiles = {}
        
        # Performance stats
        self.stats = {
            "searches_performed": 0,
            "total_users": 0,
            "average_search_time_ms": 0.0
        }
        
        logger.info(f"🔍 Banking FAISS Engine initialized (dim={vector_dim})")
    
    async def initialize(self):
        """Initialize the FAISS engine"""
        try:
            # Simple initialization - in production this would set up FAISS index
            self.is_initialized = True
            logger.info("✅ Banking FAISS Engine ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FAISS engine: {e}")
            raise
    
    async def add_user_profile(self, user_id: str, profile_vector: np.ndarray):
        """Add user behavioral profile"""
        try:
            self.user_profiles[user_id] = {
                "vector": profile_vector,
                "created": datetime.now(),
                "last_updated": datetime.now()
            }
            self.stats["total_users"] = len(self.user_profiles)
            logger.debug(f"Added profile for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding user profile: {e}")
    
    async def verify_user(self, user_id: str, current_vector: np.ndarray) -> Dict[str, Any]:
        """Verify user against their profile"""
        try:
            if user_id not in self.user_profiles:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "reason": "No profile found",
                    "recommendation": "establish_profile"
                }
            
            stored_vector = self.user_profiles[user_id]["vector"]
            
            # Simple cosine similarity
            similarity = np.dot(current_vector, stored_vector) / (
                np.linalg.norm(current_vector) * np.linalg.norm(stored_vector)
            )
            
            # Determine result
            if similarity > 0.8:
                result = "verified"
                recommendation = "allow"
            elif similarity > 0.6:
                result = "uncertain" 
                recommendation = "monitor"
            else:
                result = "anomalous"
                recommendation = "challenge"
            
            self.stats["searches_performed"] += 1
            
            return {
                "verified": result == "verified",
                "confidence": float(similarity),
                "similarity": float(similarity),
                "result": result,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Error verifying user {user_id}: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": f"Verification error: {e}",
                "recommendation": "error_fallback"
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "engine_type": "Banking FAISS Engine",
            "initialized": self.is_initialized,
            "vector_dimension": self.vector_dim,
            "total_users": self.stats["total_users"],
            "searches_performed": self.stats["searches_performed"],
            "average_search_time_ms": self.stats.get("average_search_time_ms", 0.0)
        }
