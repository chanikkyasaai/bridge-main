"""
Drift Detection System for Behavioral Authentication
Detects behavioral pattern changes and adaptation needs
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects behavioral drift and triggers adaptation
    """
    
    def __init__(self, window_size: int = 30, drift_threshold: float = 0.3):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.window_size = window_size  # Days to analyze
        self.drift_threshold = drift_threshold
        
        # Statistical tracking
        self.baseline_stats = {}
        self.current_stats = {}
        self.drift_history = defaultdict(list)
        
        # Drift detection parameters
        self.statistical_methods = [
            "kolmogorov_smirnov",
            "mann_whitney",
            "population_stability_index"
        ]
        
        self.logger.info(f"Drift Detector initialized (window: {window_size} days)")
    
    async def detect_behavioral_drift(self, user_id: str, recent_behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect drift in user's behavioral patterns
        
        Args:
            user_id: User identifier
            recent_behaviors: Recent behavioral data
            
        Returns:
            Drift detection results
        """
        try:
            # Get baseline behavioral profile
            baseline = await self._get_baseline_profile(user_id)
            
            # Extract features from recent behaviors
            current_features = self._extract_behavioral_features(recent_behaviors)
            
            # Calculate drift metrics
            drift_scores = await self._calculate_drift_scores(baseline, current_features)
            
            # Analyze drift significance
            drift_analysis = self._analyze_drift_significance(drift_scores)
            
            # Generate adaptation recommendations
            recommendations = await self._generate_adaptation_recommendations(drift_analysis)
            
            result = {
                "user_id": user_id,
                "drift_detected": drift_analysis["significant_drift"],
                "drift_severity": drift_analysis["severity"],
                "drift_scores": drift_scores,
                "affected_features": drift_analysis["affected_features"],
                "baseline_comparison": {
                    "baseline_period": baseline.get("period"),
                    "current_period": self._get_current_period(),
                    "sample_sizes": {
                        "baseline": baseline.get("sample_count", 0),
                        "current": len(recent_behaviors)
                    }
                },
                "adaptation_needed": recommendations["adaptation_needed"],
                "recommendations": recommendations["actions"],
                "confidence": drift_analysis["confidence"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store drift result for history
            self.drift_history[user_id].append(result)
            
            self.logger.info(f"Drift detection: {'YES' if result['drift_detected'] else 'NO'} "
                           f"(severity: {result['drift_severity']})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting behavioral drift: {e}")
            return {
                "error": str(e),
                "drift_detected": True,  # Fail secure
                "adaptation_needed": True
            }
    
    async def _get_baseline_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's baseline behavioral profile"""
        
        # Simulate baseline profile retrieval
        # In real implementation, fetch from historical data
        baseline_end = datetime.now() - timedelta(days=self.window_size)
        baseline_start = baseline_end - timedelta(days=60)  # 60-day baseline
        
        return {
            "user_id": user_id,
            "period": {
                "start": baseline_start.isoformat(),
                "end": baseline_end.isoformat()
            },
            "sample_count": 450,  # Simulated baseline sample size
            "features": {
                "typing_speed": {"mean": 180.5, "std": 25.3, "distribution": "normal"},
                "mouse_velocity": {"mean": 1.2, "std": 0.4, "distribution": "log_normal"},
                "session_duration": {"mean": 15.8, "std": 8.2, "distribution": "gamma"},
                "click_patterns": {"mean": 0.85, "std": 0.12, "distribution": "beta"},
                "navigation_entropy": {"mean": 2.3, "std": 0.6, "distribution": "normal"},
                "error_rate": {"mean": 0.03, "std": 0.01, "distribution": "exponential"},
                "authentication_times": {"mean": 2.1, "std": 0.7, "distribution": "normal"},
                "device_consistency": {"mean": 0.95, "std": 0.05, "distribution": "beta"}
            },
            "behavioral_vector_stats": {
                "mean_vector": np.random.normal(0.5, 0.1, 90).tolist(),
                "covariance_matrix": "compressed",  # Would store compressed covariance
                "principal_components": np.random.normal(0, 1, 10).tolist()
            }
        }
    
    def _extract_behavioral_features(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract statistical features from recent behaviors"""
        
        if not behaviors:
            return {}
        
        # Simulate feature extraction
        # In real implementation, process actual behavioral data
        
        features = {
            "typing_speed": [],
            "mouse_velocity": [],
            "session_duration": [],
            "click_patterns": [],
            "navigation_entropy": [],
            "error_rate": [],
            "authentication_times": [],
            "device_consistency": []
        }
        
        # Simulate extraction from behaviors
        for behavior in behaviors:
            features["typing_speed"].append(np.random.normal(175, 30))  # Slightly different
            features["mouse_velocity"].append(np.random.lognormal(0.1, 0.5))
            features["session_duration"].append(np.random.gamma(2, 8))
            features["click_patterns"].append(np.random.beta(8, 2))
            features["navigation_entropy"].append(np.random.normal(2.1, 0.7))
            features["error_rate"].append(np.random.exponential(0.04))
            features["authentication_times"].append(np.random.normal(2.3, 0.8))
            features["device_consistency"].append(np.random.beta(18, 2))
        
        # Calculate statistics
        feature_stats = {}
        for feature_name, values in features.items():
            if values:
                feature_stats[feature_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "values": values
                }
        
        return feature_stats
    
    async def _calculate_drift_scores(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, float]:
        """Calculate drift scores using multiple statistical methods"""
        
        drift_scores = {}
        baseline_features = baseline.get("features", {})
        
        for feature_name in baseline_features.keys():
            if feature_name in current:
                # Get baseline and current distributions
                baseline_stats = baseline_features[feature_name]
                current_stats = current[feature_name]
                
                # Calculate multiple drift metrics
                scores = {}
                
                # 1. Statistical distance (mean shift)
                baseline_mean = baseline_stats["mean"]
                current_mean = current_stats["mean"]
                baseline_std = baseline_stats["std"]
                
                mean_shift = abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
                scores["mean_shift"] = min(1.0, mean_shift / 2.0)  # Normalize
                
                # 2. Variance change
                baseline_var = baseline_stats["std"] ** 2
                current_var = current_stats["std"] ** 2
                variance_ratio = max(current_var / baseline_var, baseline_var / current_var) if baseline_var > 0 else 1.0
                scores["variance_change"] = min(1.0, (variance_ratio - 1.0) / 3.0)
                
                # 3. Population Stability Index (PSI)
                psi_score = self._calculate_psi(baseline_stats, current_stats)
                scores["psi"] = min(1.0, psi_score / 0.25)  # PSI > 0.25 indicates significant drift
                
                # 4. Distribution shape change (simplified KS test simulation)
                ks_score = self._simulate_ks_test(baseline_stats, current_stats)
                scores["distribution_change"] = ks_score
                
                # Combined drift score
                drift_scores[feature_name] = np.mean(list(scores.values()))
        
        # Overall drift score
        if drift_scores:
            drift_scores["overall"] = np.mean(list(drift_scores.values()))
        else:
            drift_scores["overall"] = 0.0
        
        return drift_scores
    
    def _calculate_psi(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate Population Stability Index"""
        
        # Simplified PSI calculation
        baseline_mean = baseline["mean"]
        current_mean = current["mean"]
        baseline_std = baseline["std"]
        current_std = current["std"]
        
        # Simulate PSI calculation
        mean_diff = abs(current_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0
        std_diff = abs(current_std - baseline_std) / baseline_std if baseline_std != 0 else 0
        
        psi = mean_diff + std_diff
        return min(1.0, psi)
    
    def _simulate_ks_test(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Simulate Kolmogorov-Smirnov test for distribution comparison"""
        
        # Generate sample distributions based on statistics
        baseline_samples = np.random.normal(baseline["mean"], baseline["std"], 100)
        current_samples = np.array(current.get("values", np.random.normal(current["mean"], current["std"], 50)))
        
        # Simple distribution comparison
        # In real implementation, use scipy.stats.ks_2samp
        baseline_sorted = np.sort(baseline_samples)
        current_sorted = np.sort(current_samples)
        
        # Simplified KS statistic approximation
        if len(current_sorted) > 10:
            # Compare distribution shapes
            baseline_cdf = np.arange(len(baseline_sorted)) / len(baseline_sorted)
            current_cdf = np.arange(len(current_sorted)) / len(current_sorted)
            
            # Interpolate to compare
            common_points = np.linspace(
                max(baseline_sorted[0], current_sorted[0]),
                min(baseline_sorted[-1], current_sorted[-1]),
                50
            )
            
            baseline_interp = np.interp(common_points, baseline_sorted, baseline_cdf)
            current_interp = np.interp(common_points, current_sorted, current_cdf)
            
            ks_stat = np.max(np.abs(baseline_interp - current_interp))
            return min(1.0, ks_stat * 2)  # Scale to 0-1
        
        return 0.0
    
    def _analyze_drift_significance(self, drift_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze the significance of detected drift"""
        
        overall_drift = drift_scores.get("overall", 0.0)
        feature_drifts = {k: v for k, v in drift_scores.items() if k != "overall"}
        
        # Determine significance
        significant_drift = overall_drift > self.drift_threshold
        
        # Categorize severity
        if overall_drift < 0.2:
            severity = "minimal"
        elif overall_drift < 0.4:
            severity = "moderate"
        elif overall_drift < 0.7:
            severity = "significant"
        else:
            severity = "critical"
        
        # Identify most affected features
        affected_features = []
        for feature, score in feature_drifts.items():
            if score > self.drift_threshold:
                affected_features.append({
                    "feature": feature,
                    "drift_score": score,
                    "severity": "high" if score > 0.6 else "medium"
                })
        
        # Calculate confidence based on data quality
        confidence = min(1.0, 0.5 + (len(feature_drifts) / 10))  # More features = higher confidence
        
        return {
            "significant_drift": significant_drift,
            "severity": severity,
            "affected_features": affected_features,
            "confidence": confidence,
            "drift_pattern": self._identify_drift_pattern(feature_drifts)
        }
    
    def _identify_drift_pattern(self, feature_drifts: Dict[str, float]) -> str:
        """Identify the pattern of drift"""
        
        high_drift_features = [f for f, score in feature_drifts.items() if score > 0.5]
        
        if len(high_drift_features) == 0:
            return "stable"
        elif len(high_drift_features) == 1:
            return "isolated_change"
        elif len(high_drift_features) <= len(feature_drifts) / 2:
            return "partial_drift"
        else:
            return "comprehensive_drift"
    
    async def _generate_adaptation_recommendations(self, drift_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for handling detected drift"""
        
        recommendations = {
            "adaptation_needed": drift_analysis["significant_drift"],
            "actions": [],
            "priority": "normal",
            "timeline": "within_7_days"
        }
        
        severity = drift_analysis["severity"]
        affected_features = drift_analysis["affected_features"]
        
        if not drift_analysis["significant_drift"]:
            recommendations["actions"].append("continue_monitoring")
            return recommendations
        
        # Set priority based on severity
        if severity == "critical":
            recommendations["priority"] = "urgent"
            recommendations["timeline"] = "immediate"
        elif severity == "significant":
            recommendations["priority"] = "high"
            recommendations["timeline"] = "within_2_days"
        
        # Specific actions based on affected features
        if affected_features:
            recommendations["actions"].extend([
                "retrain_behavioral_model",
                "update_baseline_profile",
                "adjust_authentication_thresholds"
            ])
            
            # Feature-specific recommendations
            for feature_info in affected_features:
                feature = feature_info["feature"]
                if "typing" in feature:
                    recommendations["actions"].append("recalibrate_typing_patterns")
                elif "mouse" in feature:
                    recommendations["actions"].append("update_mouse_dynamics")
                elif "session" in feature:
                    recommendations["actions"].append("adjust_session_timeout")
        
        # General recommendations
        if severity in ["significant", "critical"]:
            recommendations["actions"].extend([
                "increase_monitoring_frequency",
                "temporarily_lower_authentication_confidence",
                "notify_security_team"
            ])
        
        return recommendations
    
    def _get_current_period(self) -> Dict[str, str]:
        """Get current analysis period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.window_size)
        
        return {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
    
    async def monitor_system_drift(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor system-wide drift in authentication patterns
        
        Args:
            system_metrics: System-wide behavioral metrics
            
        Returns:
            System drift analysis
        """
        try:
            # Analyze system-wide patterns
            system_analysis = {
                "global_accuracy_drift": self._calculate_accuracy_drift(system_metrics),
                "user_population_drift": self._calculate_population_drift(system_metrics),
                "feature_importance_drift": self._calculate_feature_drift(system_metrics),
                "performance_degradation": self._check_performance_drift(system_metrics),
                "recommendations": []
            }
            
            # Generate system-level recommendations
            if system_analysis["global_accuracy_drift"] > 0.1:
                system_analysis["recommendations"].append("system_wide_model_retrain")
            
            if system_analysis["user_population_drift"] > 0.15:
                system_analysis["recommendations"].append("update_user_segmentation")
            
            if system_analysis["performance_degradation"] > 0.2:
                system_analysis["recommendations"].append("infrastructure_optimization")
            
            return system_analysis
            
        except Exception as e:
            self.logger.error(f"Error monitoring system drift: {e}")
            return {"error": str(e)}
    
    def _calculate_accuracy_drift(self, metrics: Dict[str, Any]) -> float:
        """Calculate drift in system accuracy"""
        current_accuracy = metrics.get("current_accuracy", 0.95)
        baseline_accuracy = metrics.get("baseline_accuracy", 0.97)
        
        return max(0, baseline_accuracy - current_accuracy)
    
    def _calculate_population_drift(self, metrics: Dict[str, Any]) -> float:
        """Calculate drift in user population characteristics"""
        # Simulate population drift calculation
        return np.random.uniform(0, 0.2)
    
    def _calculate_feature_drift(self, metrics: Dict[str, Any]) -> float:
        """Calculate drift in feature importance"""
        # Simulate feature importance drift
        return np.random.uniform(0, 0.15)
    
    def _check_performance_drift(self, metrics: Dict[str, Any]) -> float:
        """Check for performance degradation"""
        current_latency = metrics.get("current_avg_latency", 150)
        baseline_latency = metrics.get("baseline_avg_latency", 120)
        
        relative_increase = (current_latency - baseline_latency) / baseline_latency if baseline_latency > 0 else 0
        return max(0, relative_increase)
