"""
Learning Manager for Behavioral Authentication
Handles the learning phase for users with <6 sessions
Performs clustering when 6th session completes
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class LearningManager:
    """Manages the learning phase for new users"""
    
    def __init__(self, database_manager, feature_extractor, vector_storage_manager, bot_detector):
        self.db = database_manager
        self.feature_extractor = feature_extractor
        self.vector_storage_manager = vector_storage_manager
        self.bot_detector = bot_detector
        self.learning_threshold = 6  # Number of sessions needed to complete learning
        self.min_clusters = 2  # Minimum clusters for user profile
        self.max_clusters = 4  # Maximum clusters for user profile
        
    async def process_events(self, session_data: Dict[str, Any], new_events: List[Dict]) -> Dict[str, Any]:
        """
        Process events during learning phase
        Simply collects and stores data, no authentication decisions
        """
        try:
            user_id = session_data["user_id"]
            session_id = session_data["session_id"]
            
            logger.info(f"Processing learning phase events for session {session_id}")
            
            # Extract features from current events window
            current_vector = await self.feature_extractor.extract_features(new_events)
            
            # 🔍 DETAILED VECTOR LOGGING FOR LEARNING PHASE
            logger.info(f"📊 LEARNING PHASE VECTOR for session {session_id}:")
            logger.info(f"   📐 Vector dimensions: {len(current_vector)}")
            logger.info(f"   🔢 Complete vector: {current_vector.tolist()}")
            logger.info(f"   📈 Vector norm: {np.linalg.norm(current_vector):.6f}")
            logger.info(f"   📋 Events processed: {len(new_events)}")
            
            # Store vector in session data (will be used for final session vector)
            session_data["vectors"].append(current_vector)
            
            # 🤖 BOT DETECTION - Analyze events for robotic patterns
            if self.bot_detector:
                logger.info(f"🛡️ Running bot detection on {len(new_events)} events for session {session_id}")
                bot_result = await self.bot_detector.analyze_events(new_events)
                
                if bot_result["is_bot"]:
                    logger.warning(f"🚫 BOT DETECTED in learning phase for session {session_id}")
                    logger.warning(f"   🔍 Detection reasons: {bot_result.get('reasons', [])}")
                    logger.warning(f"   📊 Bot score: {bot_result.get('confidence', 0.0):.3f}")
                    
                    # Convert numpy types to JSON-safe Python types
                    json_safe_bot_result = self._make_json_safe(bot_result)
                    
                    return {
                        "status": "blocked",
                        "decision": "block", 
                        "phase": "learning",
                        "confidence": bot_result.get('confidence', 0.0),
                        "message": "Bot behavior detected during learning phase",
                        "bot_detection": json_safe_bot_result,
                        "session_vector_count": len(session_data["vectors"]),
                        "details": {
                            "vector_dimensions": len(current_vector),
                            "events_processed": len(new_events),
                            "detection_reasons": bot_result.get('reasons', [])
                        }
                    }
                else:
                    logger.info(f"✅ Bot detection passed for session {session_id} (score: {bot_result.get('confidence', 0.0):.3f})")
            
            # Learning phase allows legitimate users (no blocking)
            return {
                "status": "success",
                "decision": "learn",
                "phase": "learning",
                "confidence": 1.0,
                "message": "Learning phase - data collected",
                "session_vector_count": len(session_data["vectors"]),
                "details": {
                    "vector_dimensions": len(current_vector),
                    "events_processed": len(new_events),
                    "feature_vector": current_vector.tolist()  # Complete vector for debugging
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process learning events for session {session_id}: {e}")
            return {
                "status": "error",
                "decision": "learn",
                "phase": "learning",
                "confidence": 0.0,
                "message": f"Learning phase error: {str(e)}"
            }
    
    async def process_session_end(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process end of learning session
        If this is the 6th session, perform clustering
        """
        try:
            user_id = session_data["user_id"]
            session_id = session_data["session_id"]
            
            logger.info(f"Processing learning session end for {session_id}")
            
            # Create final session vector from all collected vectors
            session_vector = await self._create_session_vector(session_data)
            
            # 🔍 DETAILED SESSION VECTOR LOGGING
            logger.info(f"🏁 SESSION END VECTOR PROCESSING for {session_id}:")
            logger.info(f"   📊 Collected window vectors: {len(session_data.get('vectors', []))}")
            logger.info(f"   📐 Final session vector dimensions: {len(session_vector)}")
            logger.info(f"   🔢 Final session vector: {session_vector.tolist()}")
            logger.info(f"   📈 Final vector norm: {np.linalg.norm(session_vector):.6f}")
            
            # Store session vector in database
            vector_stored = await self.db.store_session_vector(session_id, session_vector)
            logger.info(f"   💾 Vector storage: {'✅ SUCCESS' if vector_stored else '❌ FAILED'}")
            
            # Increment user session count
            new_session_count = await self.db.increment_user_session_count(user_id)
            logger.info(f"   📈 Session count incremented to: {new_session_count}")
            
            # Mark session as ended
            await self.db.mark_session_ended(session_id)
            
            # Check if we need to perform clustering (6th session completed)
            if new_session_count >= self.learning_threshold:
                clustering_result = await self._perform_user_clustering(user_id)
                
                logger.info(f"User {user_id} completed learning phase with {new_session_count} sessions")
                
                return {
                    "status": "success",
                    "learning_completed": True,
                    "session_count": new_session_count,
                    "clustering_result": clustering_result,
                    "message": f"Learning phase completed! Created {len(clustering_result.get('clusters', []))} behavioral clusters",
                    "next_phase": "authentication"
                }
            else:
                logger.info(f"User {user_id} learning continues: {new_session_count}/{self.learning_threshold} sessions")
                
                return {
                    "status": "success",
                    "learning_completed": False,
                    "session_count": new_session_count,
                    "sessions_remaining": self.learning_threshold - new_session_count,
                    "message": f"Learning continues: {new_session_count}/{self.learning_threshold} sessions completed",
                    "next_phase": "learning"
                }
            
        except Exception as e:
            logger.error(f"Failed to process learning session end for {session_id}: {e}")
            return {
                "status": "error",
                "message": f"Learning session end error: {str(e)}"
            }
    
    async def _create_session_vector(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Create final session vector by combining all vectors from the session
        Uses weighted average with more recent vectors having higher weight
        """
        try:
            vectors = session_data.get("vectors", [])
            events_buffer = session_data.get("events_buffer", [])
            
            if not vectors and events_buffer:
                # If no vectors but have events, create from all events
                return await self.feature_extractor.extract_session_vector(events_buffer)
            elif vectors:
                # Combine existing vectors with weighted average
                if len(vectors) == 1:
                    return vectors[0]
                
                # Weight recent vectors more heavily
                weights = np.array([1.0 + (i * 0.1) for i in range(len(vectors))])
                weights = weights / np.sum(weights)
                
                # Weighted average
                session_vector = np.zeros_like(vectors[0])
                for i, vector in enumerate(vectors):
                    session_vector += weights[i] * vector
                
                return session_vector
            else:
                # Fallback: zero vector
                return np.zeros(48)
                
        except Exception as e:
            logger.error(f"Failed to create session vector: {e}")
            return np.zeros(48)
    
    async def _perform_user_clustering(self, user_id: str) -> Dict[str, Any]:
        """
        Perform unsupervised clustering on user's behavioral data
        Creates user behavioral profile with clusters using FAISS
        """
        try:
            logger.info(f"Performing clustering for user {user_id}")
            
            # Get all session vectors for this user
            session_vectors = await self.db.get_latest_user_vectors(user_id, limit=10)
            
            if len(session_vectors) < 3:
                logger.warning(f"Not enough session vectors for clustering user {user_id}: {len(session_vectors)}")
                return {"status": "insufficient_data", "vectors_count": len(session_vectors)}
            
            # Convert to numpy array format
            session_vectors_array = []
            for vector in session_vectors:
                if isinstance(vector, list):
                    session_vectors_array.append(np.array(vector))
                else:
                    session_vectors_array.append(vector)
            
            # Use vector storage manager to create behavioral profile
            profile_result = await self.vector_storage_manager.create_user_behavioral_profile(
                user_id, session_vectors_array
            )
            
            if profile_result['status'] == 'success':
                logger.info(f"Successfully created behavioral profile for user {user_id} with {profile_result['n_clusters']} clusters")
                
                return {
                    "status": "success",
                    "clusters_created": profile_result['n_clusters'],
                    "total_vectors": len(session_vectors_array),
                    "faiss_integration": True,
                    "clusters": profile_result.get('cluster_labels', [])
                }
            else:
                logger.error(f"Failed to create behavioral profile: {profile_result.get('message')}")
                return profile_result
            
        except Exception as e:
            logger.error(f"Failed to perform clustering for user {user_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _determine_optimal_clusters(self, X: np.ndarray) -> int:
        """
        Determine optimal number of clusters using elbow method
        """
        try:
            n_samples = len(X)
            
            # Limit cluster range based on data size and our constraints
            min_k = max(2, self.min_clusters)
            max_k = min(min(n_samples - 1, 5), self.max_clusters)
            
            if min_k >= max_k:
                return min_k
            
            # Calculate inertia for different k values
            inertias = []
            k_range = range(min_k, max_k + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection: find largest drop in inertia
            if len(inertias) > 1:
                drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                max_drop_idx = np.argmax(drops)
                optimal_k = k_range[max_drop_idx]
            else:
                optimal_k = min_k
            
            # Ensure within bounds
            optimal_k = max(min_k, min(optimal_k, max_k))
            
            logger.info(f"Determined optimal clusters: {optimal_k} (range: {min_k}-{max_k})")
            return optimal_k
            
        except Exception as e:
            logger.error(f"Failed to determine optimal clusters: {e}")
            return self.min_clusters
    
    def _calculate_cluster_statistics(self, X: np.ndarray, labels: np.ndarray, 
                                    centroids: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics about the created clusters"""
        try:
            n_clusters = len(centroids)
            
            # Calculate silhouette score if possible
            try:
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(X, labels)
            except:
                silhouette_avg = None
            
            # Calculate intra-cluster distances
            cluster_variances = []
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroid = centroids[i]
                    distances = [np.linalg.norm(point - centroid) for point in cluster_points]
                    cluster_variances.append(np.var(distances))
                else:
                    cluster_variances.append(0)
            
            return {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_avg,
                "avg_intra_cluster_variance": np.mean(cluster_variances),
                "cluster_sizes": [int(np.sum(labels == i)) for i in range(n_clusters)],
                "total_variance": float(np.var(cluster_variances))
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate cluster statistics: {e}")
            return {"error": str(e)}
    
    async def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """Get learning progress for a user"""
        try:
            session_count = await self.db.get_user_session_count(user_id)
            
            return {
                "user_id": user_id,
                "current_session_count": session_count,
                "required_sessions": self.learning_threshold,
                "progress_percentage": (session_count / self.learning_threshold) * 100,
                "is_learning_complete": session_count >= self.learning_threshold,
                "sessions_remaining": max(0, self.learning_threshold - session_count)
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning progress for {user_id}: {e}")
            return {"error": str(e)}
    
    def get_learning_config(self) -> Dict[str, Any]:
        """Get learning manager configuration"""
        return {
            "learning_threshold": self.learning_threshold,
            "min_clusters": self.min_clusters,
            "max_clusters": self.max_clusters,
            "feature_dimensions": 48,
            "clustering_algorithm": "KMeans",
            "normalization": "StandardScaler"
        }
    
    def _make_json_safe(self, obj):
        """Convert numpy types to JSON-safe Python types"""
        if isinstance(obj, dict):
            return {key: self._make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        else:
            return obj
