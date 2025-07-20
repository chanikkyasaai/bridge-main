"""
FAISS Vector Store for Behavioral Authentication
Handles user cluster storage, similarity matching, and vector operations
"""

try:
    import faiss
except ImportError:
    print("FAISS not available. Installing faiss-cpu...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss

import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """FAISS-based vector store for user behavioral clusters"""
    
    def __init__(self, vector_dim: int = 48, storage_path: str = "./faiss_data"):
        self.vector_dim = vector_dim
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # FAISS indices for different users
        self.user_indices = {}  # user_id -> faiss.Index
        self.user_metadata = {}  # user_id -> metadata about clusters
        self.similarity_threshold = 0.75  # For banking security
        
        # Initialize FAISS index factory
        self.index_factory_string = f"Flat"  # Simple L2 distance for now
        
        logger.info(f"Initialized FAISS Vector Store with {vector_dim} dimensions")
    
    def _get_user_index_path(self, user_id: str) -> Path:
        """Get path for user's FAISS index file"""
        return self.storage_path / f"user_{user_id}.index"
    
    def _get_user_metadata_path(self, user_id: str) -> Path:
        """Get path for user's metadata file"""
        return self.storage_path / f"user_{user_id}_metadata.json"
    
    async def create_user_clusters(self, user_id: str, cluster_vectors: np.ndarray, 
                                 cluster_labels: List[int]) -> Dict[str, Any]:
        """
        Create FAISS index for user's behavioral clusters
        
        Args:
            user_id: User identifier
            cluster_vectors: Cluster centroids (n_clusters x vector_dim)
            cluster_labels: Labels for each cluster
            
        Returns:
            Result dictionary with status and cluster info
        """
        try:
            logger.info(f"Creating FAISS clusters for user {user_id}")
            
            # Validate input
            if cluster_vectors.shape[1] != self.vector_dim:
                raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {cluster_vectors.shape[1]}")
            
            # Convert cluster vectors to float32 format for both FAISS and fallback
            cluster_vectors_f32 = cluster_vectors.astype(np.float32)
            
            # Create FAISS index
            try:
                index = faiss.IndexFlatL2(self.vector_dim)
            except AttributeError:
                try:
                    # Try different FAISS index types
                    index = faiss.IndexFlatIP(self.vector_dim)  # Inner product
                except AttributeError:
                    # Simple fallback - use numpy-based similarity
                    logger.warning("FAISS index creation failed, using numpy fallback")
                    index = None
            
            if index is not None:
                # Add cluster centroids to index
                index.add(cluster_vectors_f32)
                
                # Store index and metadata
                self.user_indices[user_id] = index
            else:
                # Store clusters without FAISS for numpy fallback
                self.user_indices[user_id] = cluster_vectors_f32
            self.user_metadata[user_id] = {
                'cluster_labels': cluster_labels,
                'n_clusters': len(cluster_labels),
                'created_at': datetime.now().isoformat(),
                'vector_dim': self.vector_dim,
                'cluster_centroids': cluster_vectors.tolist()
            }
            
            # Persist to disk
            await self._save_user_index(user_id)
            await self._save_user_metadata(user_id)
            
            logger.info(f"Successfully created {len(cluster_labels)} clusters for user {user_id}")
            
            return {
                'status': 'success',
                'user_id': user_id,
                'n_clusters': len(cluster_labels),
                'cluster_labels': cluster_labels,
                'vector_dim': self.vector_dim
            }
            
        except Exception as e:
            logger.error(f"Failed to create user clusters for {user_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def find_nearest_cluster(self, user_id: str, query_vector: np.ndarray) -> Dict[str, Any]:
        """
        Find nearest cluster for a query vector using FAISS
        
        Args:
            user_id: User identifier
            query_vector: Vector to match against user's clusters
            
        Returns:
            Dictionary with similarity results and decision
        """
        try:
            # Load user index if not in memory
            if user_id not in self.user_indices:
                await self._load_user_index(user_id)
            
            if user_id not in self.user_indices:
                return {
                    'status': 'error',
                    'message': f'No clusters found for user {user_id}',
                    'decision': 'block',
                    'confidence': 0.0
                }
            
            index = self.user_indices[user_id]
            metadata = self.user_metadata[user_id]
            
            # Ensure query vector is correct shape and type
            query_vector_flat = query_vector.flatten().astype(np.float32)
            
            if len(query_vector_flat) != self.vector_dim:
                raise ValueError(f"Query vector dimension mismatch: expected {self.vector_dim}, got {len(query_vector_flat)}")
            
            # Handle both FAISS index and numpy array fallback
            if hasattr(index, 'search'):
                # FAISS index case
                query_reshaped = query_vector_flat.reshape(1, -1)
                distances, indices = index.search(query_reshaped, k=1)  # Find closest cluster
                
                nearest_distance = float(distances[0][0])
                nearest_cluster_idx = int(indices[0][0])
                nearest_cluster_label = metadata['cluster_labels'][nearest_cluster_idx]
                
                # Convert L2 distance to similarity score (0-1 scale)
                max_distance = 10.0  # Reasonable max for normalized vectors
                similarity_score = max(0.0, 1.0 - (nearest_distance / max_distance))
            else:
                # Numpy fallback case - index is actually cluster_vectors_f32
                cluster_centroids = index  # This is the numpy array of centroids
                
                # Calculate distances to all centroids
                distances = []
                for centroid in cluster_centroids:
                    # L2 distance
                    dist = np.linalg.norm(query_vector_flat - centroid)
                    distances.append(dist)
                
                # Find nearest
                nearest_cluster_idx = np.argmin(distances)
                nearest_distance = distances[nearest_cluster_idx]
                nearest_cluster_label = metadata['cluster_labels'][nearest_cluster_idx]
                
                # Convert L2 distance to similarity score
                max_distance = 10.0
                similarity_score = max(0.0, 1.0 - (nearest_distance / max_distance))
            
            # Make authentication decision
            if similarity_score >= self.similarity_threshold:
                decision = 'allow'
                confidence = similarity_score
            else:
                decision = 'block'
                confidence = 1.0 - similarity_score
            
            logger.info(f"User {user_id} similarity: {similarity_score:.3f}, decision: {decision}")
            
            return {
                'status': 'success',
                'user_id': user_id,
                'decision': decision,
                'confidence': confidence,
                'similarity_score': similarity_score,
                'nearest_cluster_label': nearest_cluster_label,
                'nearest_distance': nearest_distance,
                'threshold_used': self.similarity_threshold,
                'cluster_info': {
                    'total_clusters': metadata['n_clusters'],
                    'matched_cluster': nearest_cluster_label
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to find nearest cluster for user {user_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'decision': 'block',
                'confidence': 0.0
            }
    
    async def update_cluster(self, user_id: str, cluster_label: int, 
                           new_vector: np.ndarray, weight: float = 0.1) -> Dict[str, Any]:
        """
        Update cluster centroid with new vector (exponential moving average)
        
        Args:
            user_id: User identifier
            cluster_label: Which cluster to update
            new_vector: New vector to incorporate
            weight: Weight for new vector (0.1 = 10% new, 90% old)
        """
        try:
            if user_id not in self.user_indices:
                await self._load_user_index(user_id)
            
            if user_id not in self.user_indices:
                return {'status': 'error', 'message': 'User clusters not found'}
            
            metadata = self.user_metadata[user_id]
            cluster_centroids = np.array(metadata['cluster_centroids'])
            
            # Find cluster index
            if cluster_label not in metadata['cluster_labels']:
                return {'status': 'error', 'message': 'Cluster label not found'}
            
            cluster_idx = metadata['cluster_labels'].index(cluster_label)
            
            # Update centroid with exponential moving average
            old_centroid = cluster_centroids[cluster_idx]
            new_centroid = (1 - weight) * old_centroid + weight * new_vector
            cluster_centroids[cluster_idx] = new_centroid
            
            # Recreate FAISS index with updated centroids
            index = faiss.IndexFlatL2(self.vector_dim)
            index.add(cluster_centroids.astype(np.float32))
            
            # Update stored data
            self.user_indices[user_id] = index
            metadata['cluster_centroids'] = cluster_centroids.tolist()
            metadata['updated_at'] = datetime.now().isoformat()
            
            # Persist changes
            await self._save_user_index(user_id)
            await self._save_user_metadata(user_id)
            
            logger.info(f"Updated cluster {cluster_label} for user {user_id}")
            
            return {
                'status': 'success',
                'updated_cluster': cluster_label,
                'weight_used': weight
            }
            
        except Exception as e:
            logger.error(f"Failed to update cluster for user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_user_cluster_info(self, user_id: str) -> Dict[str, Any]:
        """Get information about user's clusters"""
        try:
            if user_id not in self.user_metadata:
                await self._load_user_metadata(user_id)
            
            if user_id not in self.user_metadata:
                return {
                    'status': 'not_found',
                    'message': f'No cluster data found for user {user_id}'
                }
            
            metadata = self.user_metadata[user_id]
            return {
                'status': 'success',
                'user_id': user_id,
                **metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster info for user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _save_user_index(self, user_id: str) -> None:
        """Save user's FAISS index or numpy arrays to disk"""
        if user_id in self.user_indices:
            index = self.user_indices[user_id]
            index_path = self._get_user_index_path(user_id)
            
            if hasattr(index, 'search'):
                # FAISS index case
                faiss.write_index(index, str(index_path))
                logger.debug(f"Saved FAISS index for user {user_id}")
            else:
                # Numpy array fallback case
                np.save(str(index_path.with_suffix('.npy')), index)
                logger.debug(f"Saved numpy array for user {user_id}")
    
    async def _save_user_metadata(self, user_id: str) -> None:
        """Save user's metadata to disk"""
        if user_id in self.user_metadata:
            metadata_path = self._get_user_metadata_path(user_id)
            with open(metadata_path, 'w') as f:
                json.dump(self.user_metadata[user_id], f, indent=2)
            logger.debug(f"Saved metadata for user {user_id}")
    
    async def _load_user_index(self, user_id: str) -> None:
        """Load user's FAISS index or numpy arrays from disk"""
        try:
            index_path = self._get_user_index_path(user_id)
            npy_path = index_path.with_suffix('.npy')
            
            if index_path.exists():
                # FAISS index case
                index = faiss.read_index(str(index_path))
                self.user_indices[user_id] = index
                logger.debug(f"Loaded FAISS index for user {user_id}")
            elif npy_path.exists():
                # Numpy array fallback case
                index = np.load(str(npy_path))
                self.user_indices[user_id] = index
                logger.debug(f"Loaded numpy array for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to load index for user {user_id}: {e}")
    
    async def _load_user_metadata(self, user_id: str) -> None:
        """Load user's metadata from disk"""
        try:
            metadata_path = self._get_user_metadata_path(user_id)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.user_metadata[user_id] = json.load(f)
                logger.debug(f"Loaded metadata for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to load metadata for user {user_id}: {e}")
    
    async def delete_user_clusters(self, user_id: str) -> Dict[str, Any]:
        """Delete all clusters for a user"""
        try:
            # Remove from memory
            if user_id in self.user_indices:
                del self.user_indices[user_id]
            if user_id in self.user_metadata:
                del self.user_metadata[user_id]
            
            # Remove files
            index_path = self._get_user_index_path(user_id)
            metadata_path = self._get_user_metadata_path(user_id)
            
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Deleted all clusters for user {user_id}")
            return {'status': 'success', 'message': 'User clusters deleted'}
            
        except Exception as e:
            logger.error(f"Failed to delete clusters for user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}

    def set_similarity_threshold(self, threshold: float) -> None:
        """Update similarity threshold for all users"""
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Updated similarity threshold to {self.similarity_threshold}")

class VectorStorageManager:
    """Manager for handling vector storage operations with database integration"""
    
    def __init__(self, faiss_store: FAISSVectorStore, database_manager):
        self.faiss_store = faiss_store
        self.db = database_manager
        
    async def store_session_vector_with_cluster_update(self, user_id: str, session_id: str, 
                                                      session_vector: np.ndarray) -> Dict[str, Any]:
        """
        Store session vector and update nearest cluster
        This is called at the end of authenticated sessions
        """
        try:
            # Store session vector in database
            await self.db.store_session_vector(session_id, session_vector.tolist())
            
            # Find nearest cluster and update it
            cluster_result = await self.faiss_store.find_nearest_cluster(user_id, session_vector)
            
            if cluster_result['status'] == 'success' and cluster_result['decision'] == 'allow':
                # Update the nearest cluster with this session's vector
                cluster_label = cluster_result['nearest_cluster_label']
                await self.faiss_store.update_cluster(user_id, cluster_label, session_vector)
                
                logger.info(f"Stored session vector and updated cluster {cluster_label} for user {user_id}")
                
                return {
                    'status': 'success',
                    'session_vector_stored': True,
                    'cluster_updated': True,
                    'updated_cluster': cluster_label,
                    'similarity_score': cluster_result['similarity_score']
                }
            else:
                # Just store the vector, don't update clusters if it was too dissimilar
                return {
                    'status': 'success',
                    'session_vector_stored': True,
                    'cluster_updated': False,
                    'reason': 'Vector too dissimilar to existing clusters'
                }
                
        except Exception as e:
            logger.error(f"Failed to store session vector with cluster update: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def create_user_behavioral_profile(self, user_id: str, session_vectors: List[np.ndarray], session_vector_ids: List[str] = None) -> Dict[str, Any]:
        """
        Create user's behavioral profile by clustering session vectors
        Called after learning phase completion (6+ sessions)
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            if len(session_vectors) < 3:
                return {
                    'status': 'error',
                    'message': 'Insufficient session vectors for clustering'
                }
            # Convert to numpy array and normalize
            X = np.array(session_vectors)
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            # Determine optimal number of clusters (2-4 for banking)
            n_clusters = min(4, max(2, len(session_vectors) // 2))
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_normalized)
            # Get cluster centroids in original space
            centroids_normalized = kmeans.cluster_centers_
            centroids_original = scaler.inverse_transform(centroids_normalized)
            # Create cluster labels list
            unique_labels = list(range(n_clusters))
            # Store clusters in FAISS
            faiss_result = await self.faiss_store.create_user_clusters(
                user_id, centroids_original, unique_labels
            )
            # Store clusters in database - convert to the expected format (List[Tuple[int, np.ndarray, List[str]]])
            clusters_data = []
            if session_vector_ids is None:
                session_vector_ids = [str(i) for i in range(len(session_vectors))]
            for i, centroid in enumerate(centroids_original):
                # Collect session_vector_ids for this cluster
                ids_for_cluster = [session_vector_ids[j] for j, label in enumerate(cluster_labels) if label == i]
                clusters_data.append((i, centroid, ids_for_cluster))
            await self.db.store_user_clusters(user_id, clusters_data)
            logger.info(f"Created behavioral profile for user {user_id} with {n_clusters} clusters")
            return {
                'status': 'success',
                'user_id': user_id,
                'n_clusters': n_clusters,
                'cluster_labels': unique_labels,
                'faiss_result': faiss_result,
                'session_vectors_used': len(session_vectors)
            }
        except Exception as e:
            logger.error(f"Failed to create behavioral profile for user {user_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
