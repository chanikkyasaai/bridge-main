"""
Behavioral Drift Detection Module
Monitors and adapts to long-term changes in user behavior
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from scipy import stats
from sklearn.cluster import DBSCAN
from river import drift

from ml_engine.config import CONFIG
from ml_engine.utils.behavioral_vectors import BehavioralVector

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """Result from drift detection analysis"""
    user_id: str
    drift_detected: bool
    drift_type: str  # 'gradual', 'sudden', 'concept', 'none'
    drift_magnitude: float  # 0-1 scale
    confidence: float
    affected_features: List[str]
    recommendation: str  # 'update_profile', 'retrain', 'flag_review', 'none'
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class UserBehaviorProfile:
    """Comprehensive user behavioral profile"""
    user_id: str
    base_vectors: np.ndarray  # Baseline behavior vectors
    recent_vectors: np.ndarray  # Recent behavior vectors
    cluster_centers: np.ndarray  # Behavior mode clusters
    cluster_labels: List[str]  # Cluster mode names
    feature_distributions: Dict[str, Dict[str, float]]  # Feature statistics
    drift_history: List[DriftDetectionResult]
    last_updated: datetime
    creation_date: datetime
    total_samples: int
    stability_score: float  # How stable the user's behavior is

class StatisticalDriftDetector:
    """Statistical methods for drift detection"""
    
    def __init__(self, window_size: int = 100, significance_level: float = 0.05):
        self.window_size = window_size
        self.significance_level = significance_level
    
    def detect_distribution_shift(self, baseline: np.ndarray, recent: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect distribution shifts using statistical tests
        
        Returns:
            (drift_detected, p_value, test_type)
        """
        if len(baseline) < 10 or len(recent) < 10:
            return False, 1.0, "insufficient_data"
        
        try:
            # Kolmogorov-Smirnov test for distribution differences
            ks_statistic, ks_p_value = stats.ks_2samp(
                baseline.flatten(), 
                recent.flatten()
            )
            
            # Mann-Whitney U test for median differences
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                baseline.flatten(), 
                recent.flatten(),
                alternative='two-sided'
            )
            
            # Use the more conservative p-value
            min_p_value = min(ks_p_value, mw_p_value)
            drift_detected = min_p_value < self.significance_level
            
            test_type = "ks_mannwhitney"
            
            return drift_detected, min_p_value, test_type
            
        except Exception as e:
            logger.error(f"Statistical drift detection failed: {e}")
            return False, 1.0, "error"
    
    def detect_mean_shift(self, baseline: np.ndarray, recent: np.ndarray) -> Tuple[bool, float]:
        """Detect shifts in feature means"""
        if len(baseline) == 0 or len(recent) == 0:
            return False, 0.0
        
        baseline_mean = np.mean(baseline, axis=0)
        recent_mean = np.mean(recent, axis=0)
        
        # Calculate normalized difference
        baseline_std = np.std(baseline, axis=0) + 1e-8
        mean_shift = np.abs(recent_mean - baseline_mean) / baseline_std
        
        # Use average shift across features
        avg_shift = np.mean(mean_shift)
        shift_detected = avg_shift > 1.0  # More than 1 standard deviation
        
        return shift_detected, float(avg_shift)
    
    def detect_variance_change(self, baseline: np.ndarray, recent: np.ndarray) -> Tuple[bool, float]:
        """Detect changes in behavioral variance"""
        if len(baseline) < 5 or len(recent) < 5:
            return False, 0.0
        
        baseline_var = np.var(baseline, axis=0)
        recent_var = np.var(recent, axis=0)
        
        # Calculate variance ratio
        var_ratio = recent_var / (baseline_var + 1e-8)
        
        # Detect significant variance changes
        significant_change = np.any((var_ratio > 2.0) | (var_ratio < 0.5))
        avg_var_change = float(np.mean(np.abs(np.log(var_ratio + 1e-8))))
        
        return significant_change, avg_var_change

class RiverDriftDetector:
    """Online drift detection using River library"""
    
    def __init__(self):
        # ADWIN (Adaptive Windowing) detector
        self.adwin_detector = drift.ADWIN(delta=0.002)
        
        # DDM (Drift Detection Method) detector
        self.ddm_detector = drift.DDM()
        
        # Page-Hinkley detector
        self.ph_detector = drift.PageHinkley(
            min_instances=30,
            delta=0.005,
            threshold=50,
            alpha=0.9999
        )
        
        self.detectors = {
            'adwin': self.adwin_detector,
            'ddm': self.ddm_detector,
            'page_hinkley': self.ph_detector
        }
    
    def update_and_detect(self, similarity_score: float) -> Dict[str, bool]:
        """
        Update detectors with new similarity score and check for drift
        
        Args:
            similarity_score: Similarity score between current and baseline behavior
            
        Returns:
            Dict of detector_name -> drift_detected
        """
        results = {}
        
        # Convert similarity to error (drift when similarity decreases)
        error = 1.0 - similarity_score
        
        # Update ADWIN
        self.adwin_detector.update(error)
        results['adwin'] = self.adwin_detector.drift_detected
        
        # Update DDM (needs binary error signal)
        binary_error = 1 if error > 0.5 else 0
        self.ddm_detector.update(binary_error)
        results['ddm'] = self.ddm_detector.drift_detected
        
        # Update Page-Hinkley
        self.ph_detector.update(error)
        results['page_hinkley'] = self.ph_detector.drift_detected
        
        return results

class ClusterDriftDetector:
    """Detect drift through changes in behavioral clusters"""
    
    def __init__(self, eps: float = 0.3, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
    
    def detect_cluster_drift(self, baseline_vectors: np.ndarray, 
                           recent_vectors: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect drift through clustering analysis
        
        Returns:
            (drift_detected, drift_magnitude, metadata)
        """
        if len(baseline_vectors) < self.min_samples or len(recent_vectors) < self.min_samples:
            return False, 0.0, {"reason": "insufficient_data"}
        
        try:
            # Cluster baseline vectors
            baseline_clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(baseline_vectors)
            baseline_labels = baseline_clusters.labels_
            
            # Cluster recent vectors
            recent_clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(recent_vectors)
            recent_labels = recent_clusters.labels_
            
            # Calculate cluster statistics
            baseline_n_clusters = len(set(baseline_labels)) - (1 if -1 in baseline_labels else 0)
            recent_n_clusters = len(set(recent_labels)) - (1 if -1 in recent_labels else 0)
            
            # Detect changes in cluster structure
            cluster_count_change = abs(baseline_n_clusters - recent_n_clusters)
            
            # Calculate noise ratio (outliers)
            baseline_noise_ratio = np.sum(baseline_labels == -1) / len(baseline_labels)
            recent_noise_ratio = np.sum(recent_labels == -1) / len(recent_labels)
            noise_change = abs(recent_noise_ratio - baseline_noise_ratio)
            
            # Combine metrics
            drift_magnitude = (cluster_count_change / max(baseline_n_clusters, 1) + noise_change) / 2
            drift_detected = drift_magnitude > 0.3  # Threshold for significant change
            
            metadata = {
                "baseline_clusters": baseline_n_clusters,
                "recent_clusters": recent_n_clusters,
                "baseline_noise_ratio": baseline_noise_ratio,
                "recent_noise_ratio": recent_noise_ratio,
                "cluster_count_change": cluster_count_change,
                "noise_change": noise_change
            }
            
            return drift_detected, float(drift_magnitude), metadata
            
        except Exception as e:
            logger.error(f"Cluster drift detection failed: {e}")
            return False, 0.0, {"error": str(e)}

class BehavioralDriftMonitor:
    """Main behavioral drift monitoring system"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.user_profiles = {}  # user_id -> UserBehaviorProfile
        self.user_detectors = {}  # user_id -> RiverDriftDetector
        
        # Drift detection components
        self.statistical_detector = StatisticalDriftDetector(window_size)
        self.cluster_detector = ClusterDriftDetector()
        
        # Storage for recent vectors
        self.recent_vectors = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_vectors = {}  # user_id -> np.ndarray
    
    def add_behavioral_vector(self, vector: BehavioralVector):
        """Add new behavioral vector for a user"""
        user_id = vector.user_id
        
        # Initialize user if new
        if user_id not in self.user_profiles:
            self._initialize_user(user_id)
        
        # Add to recent vectors
        self.recent_vectors[user_id].append(vector.vector)
        
        # Update user profile
        self._update_user_profile(user_id, vector)
    
    def detect_drift(self, user_id: str) -> DriftDetectionResult:
        """Detect behavioral drift for a user"""
        if user_id not in self.user_profiles:
            return self._create_no_drift_result(user_id, "user_not_found")
        
        profile = self.user_profiles[user_id]
        
        # Need minimum samples for drift detection
        if len(self.recent_vectors[user_id]) < 20:
            return self._create_no_drift_result(user_id, "insufficient_recent_data")
        
        if user_id not in self.baseline_vectors or len(self.baseline_vectors[user_id]) < 20:
            return self._create_no_drift_result(user_id, "insufficient_baseline_data")
        
        # Get baseline and recent vectors
        baseline = self.baseline_vectors[user_id]
        recent = np.array(list(self.recent_vectors[user_id]))
        
        # Perform multiple drift detection methods
        results = self._perform_drift_analysis(user_id, baseline, recent)
        
        # Combine results
        drift_result = self._combine_drift_results(user_id, results)
        
        # Update profile with result
        profile.drift_history.append(drift_result)
        if len(profile.drift_history) > 50:  # Keep limited history
            profile.drift_history = profile.drift_history[-50:]
        
        # Update stability score
        self._update_stability_score(user_id)
        
        return drift_result
    
    def _initialize_user(self, user_id: str):
        """Initialize a new user profile"""
        now = datetime.now()
        
        self.user_profiles[user_id] = UserBehaviorProfile(
            user_id=user_id,
            base_vectors=np.array([]),
            recent_vectors=np.array([]),
            cluster_centers=np.array([]),
            cluster_labels=[],
            feature_distributions={},
            drift_history=[],
            last_updated=now,
            creation_date=now,
            total_samples=0,
            stability_score=1.0
        )
        
        self.user_detectors[user_id] = RiverDriftDetector()
        self.baseline_vectors[user_id] = np.array([]).reshape(0, CONFIG.BEHAVIORAL_VECTOR_DIM)
    
    def _update_user_profile(self, user_id: str, vector: BehavioralVector):
        """Update user profile with new vector"""
        profile = self.user_profiles[user_id]
        profile.total_samples += 1
        profile.last_updated = datetime.now()
        
        # Add to baseline if we don't have enough samples
        if len(self.baseline_vectors[user_id]) < 200:  # Build baseline with first 200 vectors
            if len(self.baseline_vectors[user_id]) == 0:
                self.baseline_vectors[user_id] = vector.vector.reshape(1, -1)
            else:
                self.baseline_vectors[user_id] = np.vstack([
                    self.baseline_vectors[user_id], 
                    vector.vector.reshape(1, -1)
                ])
    
    def _perform_drift_analysis(self, user_id: str, baseline: np.ndarray, 
                               recent: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive drift analysis"""
        results = {}
        
        # Statistical drift detection
        dist_shift, p_value, test_type = self.statistical_detector.detect_distribution_shift(baseline, recent)
        mean_shift, mean_shift_magnitude = self.statistical_detector.detect_mean_shift(baseline, recent)
        var_shift, var_change_magnitude = self.statistical_detector.detect_variance_change(baseline, recent)
        
        results['statistical'] = {
            'distribution_shift': dist_shift,
            'p_value': p_value,
            'test_type': test_type,
            'mean_shift': mean_shift,
            'mean_shift_magnitude': mean_shift_magnitude,
            'variance_shift': var_shift,
            'variance_change_magnitude': var_change_magnitude
        }
        
        # Cluster-based drift detection
        cluster_drift, cluster_magnitude, cluster_metadata = self.cluster_detector.detect_cluster_drift(baseline, recent)
        
        results['cluster'] = {
            'drift_detected': cluster_drift,
            'drift_magnitude': cluster_magnitude,
            'metadata': cluster_metadata
        }
        
        # Online drift detection using River
        if user_id in self.user_detectors:
            # Calculate average similarity between recent and baseline
            similarities = []
            for recent_vec in recent[-10:]:  # Use last 10 vectors
                baseline_similarities = np.dot(baseline, recent_vec) / (
                    np.linalg.norm(baseline, axis=1) * np.linalg.norm(recent_vec) + 1e-8
                )
                similarities.append(np.max(baseline_similarities))
            
            avg_similarity = np.mean(similarities) if similarities else 0.5
            
            river_results = self.user_detectors[user_id].update_and_detect(avg_similarity)
            results['online'] = river_results
        
        return results
    
    def _combine_drift_results(self, user_id: str, analysis_results: Dict[str, Any]) -> DriftDetectionResult:
        """Combine results from different drift detection methods"""
        now = datetime.now()
        
        # Extract drift indicators
        statistical = analysis_results.get('statistical', {})
        cluster = analysis_results.get('cluster', {})
        online = analysis_results.get('online', {})
        
        # Count drift detections
        drift_indicators = [
            statistical.get('distribution_shift', False),
            statistical.get('mean_shift', False),
            statistical.get('variance_shift', False),
            cluster.get('drift_detected', False),
            any(online.values()) if online else False
        ]
        
        drift_count = sum(drift_indicators)
        
        # Determine overall drift status
        drift_detected = drift_count >= 2  # Majority vote
        
        # Calculate confidence
        confidence = drift_count / len(drift_indicators)
        
        # Determine drift type
        if statistical.get('distribution_shift', False) and statistical.get('mean_shift', False):
            drift_type = 'sudden'
        elif statistical.get('variance_shift', False):
            drift_type = 'gradual'
        elif cluster.get('drift_detected', False):
            drift_type = 'concept'
        else:
            drift_type = 'none'
        
        # Calculate drift magnitude
        magnitudes = [
            statistical.get('mean_shift_magnitude', 0.0),
            statistical.get('variance_change_magnitude', 0.0),
            cluster.get('drift_magnitude', 0.0)
        ]
        drift_magnitude = np.mean([m for m in magnitudes if m > 0]) if any(m > 0 for m in magnitudes) else 0.0
        
        # Determine affected features (simplified)
        affected_features = []
        if statistical.get('mean_shift', False):
            affected_features.append('behavioral_patterns')
        if statistical.get('variance_shift', False):
            affected_features.append('behavioral_consistency')
        if cluster.get('drift_detected', False):
            affected_features.append('behavioral_modes')
        
        # Determine recommendation
        if drift_detected:
            if drift_magnitude > 0.7:
                recommendation = 'retrain'
            elif drift_magnitude > 0.4:
                recommendation = 'update_profile'
            else:
                recommendation = 'flag_review'
        else:
            recommendation = 'none'
        
        return DriftDetectionResult(
            user_id=user_id,
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_magnitude=float(drift_magnitude),
            confidence=float(confidence),
            affected_features=affected_features,
            recommendation=recommendation,
            timestamp=now,
            metadata={
                'analysis_results': analysis_results,
                'drift_indicators': drift_indicators,
                'drift_count': drift_count
            }
        )
    
    def _create_no_drift_result(self, user_id: str, reason: str) -> DriftDetectionResult:
        """Create a no-drift result with reason"""
        return DriftDetectionResult(
            user_id=user_id,
            drift_detected=False,
            drift_type='none',
            drift_magnitude=0.0,
            confidence=0.0,
            affected_features=[],
            recommendation='none',
            timestamp=datetime.now(),
            metadata={'reason': reason}
        )
    
    def _update_stability_score(self, user_id: str):
        """Update user stability score based on drift history"""
        profile = self.user_profiles[user_id]
        
        if not profile.drift_history:
            return
        
        # Look at recent drift events
        recent_history = profile.drift_history[-10:]  # Last 10 drift checks
        drift_events = [r for r in recent_history if r.drift_detected]
        
        # Calculate stability score
        stability = 1.0 - (len(drift_events) / len(recent_history))
        
        # Adjust based on drift magnitudes
        if drift_events:
            avg_magnitude = np.mean([d.drift_magnitude for d in drift_events])
            stability *= (1.0 - avg_magnitude * 0.5)
        
        profile.stability_score = max(0.0, min(1.0, stability))
    
    def get_user_stability_report(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive stability report for a user"""
        if user_id not in self.user_profiles:
            return {"error": "User not found"}
        
        profile = self.user_profiles[user_id]
        
        # Recent drift events
        recent_drifts = [d for d in profile.drift_history[-30:] if d.drift_detected]
        
        # Drift frequency
        total_checks = len(profile.drift_history)
        drift_frequency = len(recent_drifts) / max(total_checks, 1)
        
        return {
            'user_id': user_id,
            'stability_score': profile.stability_score,
            'total_samples': profile.total_samples,
            'total_drift_checks': total_checks,
            'recent_drift_events': len(recent_drifts),
            'drift_frequency': drift_frequency,
            'last_drift': recent_drifts[-1].timestamp.isoformat() if recent_drifts else None,
            'profile_age_days': (datetime.now() - profile.creation_date).days,
            'recommendations': self._get_stability_recommendations(profile)
        }
    
    def _get_stability_recommendations(self, profile: UserBehaviorProfile) -> List[str]:
        """Get recommendations based on user stability"""
        recommendations = []
        
        if profile.stability_score < 0.3:
            recommendations.append("High instability detected - consider additional authentication")
        elif profile.stability_score < 0.6:
            recommendations.append("Moderate instability - monitor closely")
        
        if len(profile.drift_history) > 0:
            recent_drift = profile.drift_history[-1]
            if recent_drift.drift_detected and recent_drift.drift_magnitude > 0.5:
                recommendations.append(f"Recent significant drift - {recent_drift.recommendation}")
        
        if profile.total_samples < 100:
            recommendations.append("Insufficient data for reliable drift detection")
        
        return recommendations
    
    def update_user_baseline(self, user_id: str):
        """Update user baseline with recent stable behavior"""
        if user_id not in self.user_profiles:
            return False
        
        recent_vectors = list(self.recent_vectors[user_id])
        if len(recent_vectors) < 50:
            return False
        
        # Use recent vectors as new baseline
        self.baseline_vectors[user_id] = np.array(recent_vectors)
        
        # Clear recent vectors
        self.recent_vectors[user_id].clear()
        
        # Reset drift detector
        self.user_detectors[user_id] = RiverDriftDetector()
        
        logger.info(f"Updated baseline for user {user_id} with {len(recent_vectors)} vectors")
        return True

class BehavioralDriftDetector:
    """Main behavioral drift detection system with async support"""
    
    def __init__(self):
        self.drift_monitor = BehavioralDriftMonitor()
        self.statistical_detector = StatisticalDriftDetector()
        self.cluster_detector = ClusterDriftDetector()
        self.user_profiles = {}  # user_id -> UserBehaviorProfile
        
    async def initialize(self):
        """Initialize drift detection system"""
        logger.info("Initializing Behavioral Drift Detector...")
        
        # Initialize components
        self.drift_monitor = BehavioralDriftMonitor()
        self.statistical_detector = StatisticalDriftDetector()
        self.cluster_detector = ClusterDriftDetector()
        
        logger.info("✓ Behavioral Drift Detector initialized")
    
    async def detect_drift(self, vectors: List[BehavioralVector], user_id: str) -> DriftDetectionResult:
        """Detect behavioral drift for user vectors"""
        if not vectors:
            return self._create_no_drift_result(user_id, "no_vectors_provided")
        
        # Add vectors to monitor
        for vector in vectors:
            self.drift_monitor.add_behavioral_vector(vector)
        
        # Perform drift detection
        result = self.drift_monitor.detect_drift(user_id)
        
        return result
    
    async def update_user_baseline(self, user_id: str, vectors: List[BehavioralVector]):
        """Update user baseline with new vectors"""
        if not vectors:
            return
        
        # Convert to numpy array
        vector_array = np.array([v.vector for v in vectors])
        
        # Update baseline
        self.drift_monitor.baseline_vectors[user_id] = vector_array
        
        # Initialize or update user profile
        if user_id not in self.drift_monitor.user_profiles:
            self.drift_monitor._initialize_user(user_id)
        
        profile = self.drift_monitor.user_profiles[user_id]
        profile.base_vectors = vector_array
        profile.total_samples += len(vectors)
        profile.last_updated = datetime.now()
        
        logger.info(f"Updated baseline for user {user_id} with {len(vectors)} vectors")
    
    async def get_user_profile(self, user_id: str) -> Optional[UserBehaviorProfile]:
        """Get user behavioral profile"""
        return self.drift_monitor.user_profiles.get(user_id)
    
    async def save_profiles(self):
        """Save user profiles"""
        # In a full implementation, this would save to persistent storage
        logger.info("Drift detection profiles saved")
    
    def _create_no_drift_result(self, user_id: str, reason: str) -> DriftDetectionResult:
        """Create no-drift result"""
        return DriftDetectionResult(
            user_id=user_id,
            drift_detected=False,
            drift_type="none",
            drift_magnitude=0.0,
            confidence=0.0,
            affected_features=[],
            recommendation="none",
            timestamp=datetime.now(),
            metadata={"reason": reason}
        )
