"""
Authentication Manager for Behavioral Authentication
Handles authentication phase for users with >=6 sessions
Uses FAISS for real-time similarity matching
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio
from gnn_escalation import SessionEventGraph, GNNAnomalyDetector, detect_anomaly_with_user_adaptation

logger = logging.getLogger(__name__)

class AuthenticationManager:
    """Manages authentication phase for experienced users"""
    
    def __init__(self, database_manager, feature_extractor, vector_storage_manager, faiss_store, bot_detector):
        self.db = database_manager
        self.feature_extractor = feature_extractor
        self.vector_storage_manager = vector_storage_manager
        self.faiss_store = faiss_store
        self.bot_detector = bot_detector
        
        # Session state management
        self.session_vectors = {}  # Track session vectors for cumulative analysis
        # self.gnn_detector = GNNAnomalyDetector()
        # self.session_graphs = {}  # session_id -> SessionEventGraph
        
        # Configuration
        self.similarity_threshold = 0.75  # Default, but will use user-specific
        self.block_threshold = 0.6       # Below this = immediate block
        self.high_risk_threshold = 0.4   # High risk threshold for additional checks
        self.update_weight = 0.05        # Weight for cluster updates (5%)
        self.min_events_for_analysis = 5  # Minimum events needed for analysis
    
    async def process_events(self, session_data: Dict[str, Any], new_events: List[Dict]) -> Dict[str, Any]:
        """
        Process events during authentication phase
        Performs continuous behavioral authentication
        """
        try:
            user_id = session_data["user_id"]
            session_id = session_data["session_id"]
            
            logger.info(f"Processing authentication events for session {session_id}")
            
            # # Fetch user threshold_variance
            # user_profile = await self.db.get_user_profile(user_id)
            # threshold_variance = user_profile.get('threshold_variance', 0.0)
            # user_threshold = 0.75 - threshold_variance
            
            # Check if we have enough events for analysis
            if len(new_events) < self.min_events_for_analysis:
                return {
                    "status": "success",
                    "decision": "allow",
                    "phase": "authentication",
                    "confidence": 0.8,
                    "message": "Insufficient events for analysis - allowing",
                    "events_count": len(new_events)
                }
            
            # Extract features from current events window
            current_vector = await self.feature_extractor.extract_features(new_events)
            
            # 🔍 DETAILED VECTOR LOGGING FOR AUTHENTICATION PHASE
            logger.info(f"🔐 AUTHENTICATION PHASE VECTOR for session {session_id}:")
            logger.info(f"   📐 Vector dimensions: {len(current_vector)}")
            logger.info(f"   🔢 Current vector: {current_vector.tolist()}")
            logger.info(f"   📈 Vector norm: {np.linalg.norm(current_vector):.6f}")
            logger.info(f"   📋 Events processed: {len(new_events)}")
            
            # 🤖 BOT DETECTION - Critical Security Check
            logger.info(f"🤖 RUNNING BOT DETECTION for session {session_id}")
            bot_result = await self.bot_detector.analyze_events(new_events)
            logger.info(f"   🎯 Bot detection result: {bot_result}")
            
            # Block if bot behavior detected
            if bot_result.get("is_bot", False):
                logger.warning(f"🚫 BOT BEHAVIOR DETECTED - session {session_id} blocked")
                logger.warning(f"   🔍 Detection reasons: {bot_result.get('reasons', [])}")
                logger.warning(f"   📊 Bot confidence: {bot_result.get('confidence', 0.0):.3f}")
                
                # Convert numpy types to JSON-safe Python types
                json_safe_bot_result = self._make_json_safe(bot_result)
                
                return {
                    "status": "success",
                    "decision": "block",
                    "phase": "authentication",
                    "confidence": 0.95,
                    "similarity": 0.0,
                    "bot_detection": json_safe_bot_result,
                    "reason": "bot_behavior_detected",
                    "message": f"Automated behavior detected (confidence: {bot_result.get('confidence', 0.0):.3f})",
                    "events_count": len(new_events)
                }
            
            # Update cumulative session vector
            self._update_session_vector(session_data, current_vector)
            
            # Get current cumulative vector for comparison
            cumulative_vector = self._get_cumulative_vector(session_data)
            
            logger.info(f"   📊 Cumulative vector norm: {np.linalg.norm(cumulative_vector):.6f}")
            logger.info(f"   📊 Session vectors count: {len(session_data.get('vectors', []))}")
            
            # Perform similarity matching with user clusters
            auth_result = await self._perform_authentication(user_id, cumulative_vector)
            
            # similarity = auth_result.get('similarity', 0.0)
            
            # Escalation logic
            if similarity < user_threshold:
                # Escalate to Level 2 (GNN with Supabase historical data)
                logger.warning(f"🚨 Escalating session {session_id} to Level 2 (GNN) due to low similarity: {similarity:.3f} < {user_threshold:.3f}")
                
                # Prepare current session data for GNN analysis
                current_session_data = {
                    "logs": new_events
                }
                
                # Perform user-adapted anomaly detection using Supabase historical data
                try:
                    # Import the global managers from main
                    from main import gnn_storage_manager, gnn_caching_manager
                    
                    gnn_result = await detect_anomaly_with_user_adaptation(
                        current_session_data, 
                        user_id,
                        storage_manager=gnn_storage_manager,
                        caching_manager=gnn_caching_manager
                    )
                    
                    base_anomaly_score = gnn_result.get("base_anomaly_score", 0.0)
                    adapted_anomaly_score = gnn_result.get("adapted_anomaly_score")
                    historical_sessions_used = gnn_result.get("historical_sessions_used", 0)
                    
                    logger.info(f"GNN escalation completed for session {session_id}:")
                    logger.info(f"   📊 Base anomaly score: {base_anomaly_score:.4f}")
                    logger.info(f"   📊 Adapted anomaly score: {adapted_anomaly_score:.4f if adapted_anomaly_score else 'N/A'}")
                    logger.info(f"   📊 Historical sessions used: {historical_sessions_used}")
                    
                    # Use adapted score if available, otherwise use base score
                    final_anomaly_score = adapted_anomaly_score if adapted_anomaly_score is not None else base_anomaly_score
                    
                    return {
                        **auth_result,
                        "status": "escalated",
                        "decision": "escalate",
                        "phase": "escalation",
                        "similarity": similarity,
                        "user_threshold": user_threshold,
                        "anomaly_score": final_anomaly_score,
                        "base_anomaly_score": base_anomaly_score,
                        "adapted_anomaly_score": adapted_anomaly_score,
                        "historical_sessions_used": historical_sessions_used,
                        "user_profile_available": gnn_result.get("user_profile_available", False),
                        "message": f"Escalated to Level 2 (GNN) due to low similarity. Anomaly score: {final_anomaly_score:.4f}",
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to perform GNN escalation for session {session_id}: {e}")
                    # Fallback to basic GNN without historical data
                    if session_id not in self.session_graphs:
                        self.session_graphs[session_id] = SessionEventGraph()
                    for event in new_events:
                        self.session_graphs[session_id].add_event(event)
                    session_graph = self.session_graphs[session_id]
                    pyg_data = session_graph.to_pyg_data()
                    fallback_anomaly_score = self.gnn_detector.predict_anomaly(pyg_data)
                    
                    return {
                        **auth_result,
                        "status": "escalated",
                        "decision": "escalate", 
                        "phase": "escalation",
                        "similarity": similarity,
                        "user_threshold": user_threshold,
                        "anomaly_score": fallback_anomaly_score,
                        "error": f"Supabase GNN failed, using fallback: {str(e)}",
                        "message": f"Escalated to Level 2 (GNN fallback) due to low similarity. Anomaly score: {fallback_anomaly_score:.4f}",
                    }
            
            # Update cluster if authentication passed (incremental learning)
            if auth_result["decision"] == "allow" and similarity > user_threshold:
                await self._update_nearest_cluster(user_id, cumulative_vector, auth_result["nearest_cluster"])
            
            # Store vector in session data for final processing
            session_data["vectors"].append(current_vector)
            
            # Enhanced result with authentication details
            result = {
                **auth_result,
                "status": "success",
                "phase": "authentication",
                "session_vector_count": len(session_data["vectors"]),
                "cumulative_vector_strength": float(np.linalg.norm(cumulative_vector)),
                "events_processed": len(new_events)
            }
            
            logger.info(f"Authentication result for {session_id}: {auth_result['decision']} "
                       f"(similarity: {auth_result['similarity']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process auth events for session {session_id}: {e}")
            return {
                "status": "error",
                "decision": "block",  # Fail securely
                "phase": "authentication",
                "confidence": 0.0,
                "message": f"Authentication error: {str(e)}"
            }
    
    async def process_session_end(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process end of authentication session
        Stores final session vector and updates user profile
        """
        try:
            user_id = session_data["user_id"]
            session_id = session_data["session_id"]
            
            logger.info(f"Processing authentication session end for {session_id}")
            
            # Create final session vector
            session_vector = await self._create_final_session_vector(session_data)
            
            # Store session vector in database
            await self.db.store_session_vector(session_id, session_vector)
            
            # Find and update the nearest cluster with final vector
            nearest_cluster = await self.db.find_nearest_cluster(user_id, session_vector)
            
            if nearest_cluster:
                # Update cluster with final session vector (skip update for now - clusters don't have 'id')
                # await self.db.update_cluster_with_vector(
                #     nearest_cluster["id"], 
                #     session_vector, 
                #     alpha=self.vector_update_weight
                # )
                
                logger.info(f"Updated cluster {nearest_cluster['cluster_label']} for user {user_id}")
            
            # Mark session as ended
            await self.db.mark_session_ended(session_id)
            
            # Increment session count
            new_session_count = await self.db.increment_user_session_count(user_id)
            
            # Clean up session state
            if session_id in self.session_vectors:
                del self.session_vectors[session_id]
            
            return {
                "status": "success",
                "session_count": new_session_count,
                "cluster_updated": nearest_cluster is not None,
                "cluster_info": {
                    "label": nearest_cluster["cluster_label"] if nearest_cluster else None,
                    "similarity": nearest_cluster["similarity"] if nearest_cluster else None
                } if nearest_cluster else None,
                "message": "Authentication session completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to process auth session end for {session_id}: {e}")
            return {
                "status": "error",
                "message": f"Authentication session end error: {str(e)}"
            }
    
    def _update_session_vector(self, session_data: Dict[str, Any], new_vector: np.ndarray):
        """Update cumulative session vector with new data"""
        session_id = session_data["session_id"]
        
        if session_id not in self.session_vectors:
            self.session_vectors[session_id] = {
                "cumulative_vector": new_vector.copy(),
                "vector_count": 1,
                "start_time": datetime.utcnow()
            }
        else:
            # Weighted average with decay for older data
            session_state = self.session_vectors[session_id]
            weight = 1.0 / (session_state["vector_count"] + 1)
            
            # Update cumulative vector
            session_state["cumulative_vector"] = (
                (1 - weight) * session_state["cumulative_vector"] + 
                weight * new_vector
            )
            session_state["vector_count"] += 1
    
    def _get_cumulative_vector(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Get current cumulative vector for session"""
        session_id = session_data["session_id"]
        
        if session_id in self.session_vectors:
            return self.session_vectors[session_id]["cumulative_vector"]
        else:
            # Fallback: use latest vector from session data
            vectors = session_data.get("vectors", [])
            if vectors:
                return vectors[-1]
            else:
                return np.zeros(48)
    
    async def _perform_authentication(self, user_id: str, vector: np.ndarray) -> Dict[str, Any]:
        """
        Perform behavioral authentication against user clusters
        """
        try:
            # Get user clusters
            clusters = await self.db.get_user_clusters(user_id)
            
            if not clusters:
                logger.warning(f"No clusters found for user {user_id} - should not happen in auth phase")
                return {
                    "decision": "block",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "reason": "no_user_clusters",
                    "message": "User profile not found"
                }
            
            # 🔍 DETAILED SIMILARITY LOGGING
            logger.info(f"🎯 SIMILARITY ANALYSIS for user {user_id}:")
            logger.info(f"   📊 Available clusters: {len(clusters)}")
            logger.info(f"   📐 Input vector norm: {np.linalg.norm(vector):.6f}")
            
            # Calculate similarities for all clusters
            cluster_similarities = []
            for i, cluster in enumerate(clusters):
                centroid = np.array(cluster["centroid"])
                similarity = self._calculate_similarity(vector, centroid)
                cluster_similarities.append({
                    "cluster_id": i,
                    "cluster_label": cluster.get("cluster_label", i),
                    "similarity": similarity,
                    "centroid_norm": np.linalg.norm(centroid)
                })
                logger.info(f"   🎯 Cluster {i}: similarity={similarity:.6f}, centroid_norm={np.linalg.norm(centroid):.6f}")
            
            # Find best matching cluster
            best_match = self._find_best_cluster_match(clusters, vector)
            
            logger.info(f"   ⭐ BEST MATCH: similarity={best_match['similarity']:.6f}")
            logger.info(f"   ⚖️  Threshold: {self.similarity_threshold:.6f}")
            
            # Make authentication decision
            decision_result = self._make_auth_decision(best_match)
            
            return {
                **decision_result,
                "nearest_cluster": best_match,
                "total_clusters": len(clusters),
                "cluster_similarities": [
                    {
                        "label": cluster["cluster_label"],
                        "similarity": float(self._calculate_similarity(vector, cluster["centroid"]))
                    }
                    for cluster in clusters
                ]
            }
            
        except Exception as e:
            logger.error(f"Authentication failed for user {user_id}: {e}")
            return {
                "decision": "block",
                "confidence": 0.0,
                "similarity": 0.0,
                "reason": "authentication_error",
                "message": str(e)
            }
    
    def _find_best_cluster_match(self, clusters: List[Dict], vector: np.ndarray) -> Dict[str, Any]:
        """Find the best matching cluster for authentication"""
        best_similarity = -1
        best_cluster = None
        
        for cluster in clusters:
            similarity = self._calculate_similarity(vector, cluster["centroid"])
            
            if similarity > best_similarity:
                best_similarity = similarity
                # Create a JSON-serializable cluster object
                best_cluster = {
                    "cluster_label": cluster["cluster_label"],
                    "created_at": cluster["created_at"],
                    "similarity": float(similarity),
                    "distance": float(1 - similarity)
                }
        
        return best_cluster
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(vector1, vector2) / (norm1 * norm2)
            
            # Clamp to [0, 1] range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _make_auth_decision(self, best_match: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make authentication decision based on cluster similarity
        """
        if not best_match:
            return {
                "decision": "block",
                "confidence": 0.0,
                "similarity": 0.0,
                "reason": "no_cluster_match",
                "message": "No behavioral clusters match"
            }
        
        similarity = float(best_match["similarity"])  # Ensure JSON serializable
        
        # High risk - immediate block
        if similarity < (1 - self.high_risk_threshold):
            return {
                "decision": "block",
                "confidence": 0.95,
                "similarity": similarity,
                "reason": "high_risk_behavior",
                "message": f"Behavioral pattern highly anomalous (similarity: {similarity:.3f})"
            }
        
        # Below threshold - block
        elif similarity < self.similarity_threshold:
            return {
                "decision": "block",
                "confidence": 0.8,
                "similarity": similarity,
                "reason": "low_similarity",
                "message": f"Behavioral pattern does not match user profile (similarity: {similarity:.3f})"
            }
        
        # Low confidence - allow but flag
        elif similarity < (self.similarity_threshold + 0.1):
            return {
                "decision": "allow",
                "confidence": 0.6,
                "similarity": similarity,
                "reason": "marginal_match",
                "message": f"Behavioral pattern marginally matches (similarity: {similarity:.3f})"
            }
        
        # Good match - allow
        else:
            confidence = min(0.95, 0.7 + (similarity - self.similarity_threshold) * 2)
            return {
                "decision": "allow",
                "confidence": float(confidence),
                "similarity": similarity,
                "reason": "good_match",
                "message": f"Behavioral pattern matches user profile (similarity: {similarity:.3f})"
            }
    
    async def _update_nearest_cluster(self, user_id: str, vector: np.ndarray, 
                                    cluster_info: Dict[str, Any]) -> bool:
        """Update the nearest cluster with new behavioral data"""
        try:
            if not cluster_info:
                return False
            
            # Skip cluster update for now as database schema doesn't include 'id' field
            # This would require database schema changes to support incremental updates
            logger.debug(f"Skipping cluster update for cluster {cluster_info.get('cluster_label', 'unknown')} (user {user_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update cluster for user {user_id}: {e}")
            return False
    
    async def _create_final_session_vector(self, session_data: Dict[str, Any]) -> np.ndarray:
        """Create final session vector from all session data"""
        try:
            session_id = session_data["session_id"]
            
            # Use cumulative vector if available
            if session_id in self.session_vectors:
                return self.session_vectors[session_id]["cumulative_vector"]
            
            # Fallback: combine all vectors
            vectors = session_data.get("vectors", [])
            events_buffer = session_data.get("events_buffer", [])
            
            if vectors:
                # Weighted average of all vectors (recent vectors weighted more)
                if len(vectors) == 1:
                    return vectors[0]
                
                weights = np.array([1.0 + (i * 0.1) for i in range(len(vectors))])
                weights = weights / np.sum(weights)
                
                final_vector = np.zeros_like(vectors[0])
                for i, vector in enumerate(vectors):
                    final_vector += weights[i] * vector
                
                return final_vector
            
            elif events_buffer:
                # Create vector from all events
                return await self.feature_extractor.extract_session_vector(events_buffer)
            
            else:
                return np.zeros(48)
                
        except Exception as e:
            logger.error(f"Failed to create final session vector: {e}")
            return np.zeros(48)
    
    async def get_user_auth_status(self, user_id: str) -> Dict[str, Any]:
        """Get authentication status and statistics for a user"""
        try:
            # Get user profile
            profile = await self.db.get_user_profile(user_id)
            
            # Get clusters
            clusters = await self.db.get_user_clusters(user_id)
            
            return {
                "user_id": user_id,
                "is_auth_ready": len(clusters) > 0,
                "cluster_count": len(clusters),
                "session_count": profile.get("sessions_count", 0),
                "clusters_info": [
                    {
                        "label": cluster["label"],
                        "created_at": cluster["created_at"],
                        "dimensions": len(cluster["centroid"])
                    }
                    for cluster in clusters
                ],
                "auth_thresholds": {
                    "similarity_threshold": self.similarity_threshold,
                    "high_risk_threshold": self.high_risk_threshold,
                    "low_confidence_threshold": self.block_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get auth status for {user_id}: {e}")
            return {"error": str(e)}
    
    def get_auth_config(self) -> Dict[str, Any]:
        """Get authentication manager configuration"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "high_risk_threshold": self.high_risk_threshold,
            "low_confidence_threshold": self.block_threshold,
            "vector_update_weight": self.update_weight,
            "min_events_for_analysis": self.min_events_for_analysis,
            "feature_dimensions": 48,
            "similarity_metric": "cosine_similarity"
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


class ContinuousAuthenticator:
    """
    Orchestrates continuous authentication process
    Handles the n-second window processing and decision making
    """
    
    def __init__(self, learning_manager, auth_manager, database_manager):
        self.learning_manager = learning_manager
        self.auth_manager = auth_manager
        self.db = database_manager
        self.window_duration = 15  # seconds
        
    async def process_continuous_authentication(self, user_id: str, session_id: str, 
                                              new_events: List[Dict], session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for continuous authentication
        Routes to learning or authentication manager based on user's session count
        """
        try:
            # Get user session count to determine phase
            session_count = await self.db.get_user_session_count(user_id)
            
            # Add events to session buffer
            if "events_buffer" not in session_data:
                session_data["events_buffer"] = []
            if "vectors" not in session_data:
                session_data["vectors"] = []
            
            session_data["events_buffer"].extend(new_events)
            
            # Route to appropriate manager
            if session_count < 6:
                # Learning phase
                result = await self.learning_manager.process_events(session_data, new_events)
                result["session_count"] = session_count
                result["phase"] = "learning"
            else:
                # Authentication phase  
                result = await self.auth_manager.process_events(session_data, new_events)
                result["session_count"] = session_count
                result["phase"] = "authentication"
            
            # Add timing information
            result["window_duration"] = self.window_duration
            result["timestamp"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Failed continuous authentication for user {user_id}: {e}")
            return {
                "status": "error",
                "decision": "block",  # Fail-safe
                "confidence": 0.0,
                "message": f"Continuous authentication error: {str(e)}"
            }
    
    async def end_session_processing(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session end processing
        Routes to appropriate manager for final processing
        """
        try:
            user_id = session_data["user_id"]
            session_count = await self.db.get_user_session_count(user_id)
            
            if session_count < 6:
                # Learning phase end
                return await self.learning_manager.process_session_end(session_data)
            else:
                # Authentication phase end
                return await self.auth_manager.process_session_end(session_data)
                
        except Exception as e:
            logger.error(f"Failed session end processing: {e}")
            return {
                "status": "error",
                "message": f"Session end error: {str(e)}"
            }
