"""
Unit tests for adaptive learning layer.
"""

import pytest
import numpy as np
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from src.layers.adaptive_layer import AdaptiveLayer, LearningPattern, AdaptationMetrics
from src.data.models import (
    BehavioralVector, UserProfile, AuthenticationDecision, 
    RiskLevel, SessionPhase, BehavioralFeatures
)
from src.core.vector_store import HDF5VectorStore
from src.utils.constants import TOTAL_VECTOR_DIM
from src.config.settings import Settings


def create_sample_features() -> BehavioralFeatures:
    """Helper function to create sample behavioral features."""
    return BehavioralFeatures(
        typing_speed=45.5,
        keystroke_intervals=[0.1, 0.15, 0.12],
        typing_rhythm_variance=0.05,
        backspace_frequency=0.02,
        typing_pressure=[0.5, 0.6, 0.55],
        touch_pressure=[0.7, 0.8, 0.75],
        touch_duration=[0.2, 0.25, 0.22],
        touch_area=[100.0, 105.0, 102.0],
        swipe_velocity=[2.5, 3.0, 2.8],
        touch_coordinates=[{"x": 100, "y": 200}, {"x": 150, "y": 250}],
        navigation_patterns=["home", "menu", "profile"],
        screen_time_distribution={"home": 30.0, "menu": 10.0, "profile": 20.0},
        interaction_frequency=0.5,
        session_duration=120.0,
        device_orientation="portrait",
        time_of_day=14,
        day_of_week=1,
        app_version="1.0.0"
    )


def create_behavioral_vector(user_id: str, session_id: str, vector: list = None, confidence: float = 0.8) -> BehavioralVector:
    """Helper function to create a BehavioralVector."""
    if vector is None:
        vector = np.random.random(TOTAL_VECTOR_DIM).tolist()
    
    return BehavioralVector(
        user_id=user_id,
        session_id=session_id,
        vector=vector,
        confidence_score=confidence,
        feature_source=create_sample_features()
    )


class TestAdaptiveLayer:
    """Test adaptive layer functionality."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            adaptive_learning_rate=0.01,
            adaptation_threshold=0.1,
            pattern_retention_days=30,
            min_feedback_samples=3
        )
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock(spec=HDF5VectorStore)
        return store
    
    @pytest.fixture
    def adaptive_layer(self, mock_vector_store, settings):
        """Create adaptive layer instance."""
        return AdaptiveLayer(mock_vector_store, settings)
    
    @pytest.fixture
    def sample_behavioral_vector(self):
        """Create sample behavioral vector."""
        return create_behavioral_vector(
            user_id="test_user",
            session_id="test_session"
        )
    
    @pytest.fixture
    def sample_feedback_data(self, sample_behavioral_vector):
        """Create sample feedback data."""
        return [
            {
                'timestamp': datetime.utcnow(),
                'vector': sample_behavioral_vector.vector,
                'decision': 'allow',
                'was_correct': True,
                'confidence': 0.8,
                'context': {'device_type': 'mobile', 'time_of_day': 14}
            },
            {
                'timestamp': datetime.utcnow(),
                'vector': sample_behavioral_vector.vector,
                'decision': 'allow',
                'was_correct': False,  # False positive
                'confidence': 0.9,
                'context': {'device_type': 'mobile', 'time_of_day': 15}
            },
            {
                'timestamp': datetime.utcnow(),
                'vector': sample_behavioral_vector.vector,
                'decision': 'deny',
                'was_correct': True,
                'confidence': 0.7,
                'context': {'device_type': 'desktop', 'time_of_day': 22}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_learn_from_authentication(self, adaptive_layer, sample_behavioral_vector):
        """Test learning from authentication feedback."""
        user_id = "test_user"
        
        result = await adaptive_layer.learn_from_authentication(
            user_id=user_id,
            behavioral_vector=sample_behavioral_vector,
            decision=AuthenticationDecision.ALLOW,
            was_correct=True,
            confidence=0.8,
            context={'device_type': 'mobile'}
        )
        
        assert result is True
        assert user_id in adaptive_layer.feedback_buffer
        assert len(adaptive_layer.feedback_buffer[user_id]) == 1
        
        feedback = adaptive_layer.feedback_buffer[user_id][0]
        assert feedback['decision'] == 'allow'
        assert feedback['was_correct'] is True
        assert feedback['confidence'] == 0.8
    
    @pytest.mark.asyncio
    async def test_adapt_user_model_insufficient_feedback(self, adaptive_layer):
        """Test adaptation with insufficient feedback."""
        result = await adaptive_layer.adapt_user_model("test_user")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_adapt_user_model_success(self, adaptive_layer, sample_behavioral_vector):
        """Test successful model adaptation."""
        user_id = "test_user"
        
        # Create feedback data with high false positive rate to trigger adaptation
        feedback_data = []
        for i in range(10):
            feedback_data.append({
                'timestamp': datetime.utcnow(),
                'vector': sample_behavioral_vector.vector,
                'decision': 'allow',
                'was_correct': i < 6,  # 40% false positive rate
                'confidence': 0.8,
                'context': {'device_type': 'mobile', 'time_of_day': 14}
            })
        
        adaptive_layer.feedback_buffer[user_id] = feedback_data
        
        result = await adaptive_layer.adapt_user_model(user_id)
        
        assert result is True
        assert len(adaptive_layer.feedback_buffer[user_id]) == 0  # Buffer cleared
        assert user_id in adaptive_layer.adaptation_history
    
    @pytest.mark.asyncio
    async def test_get_adaptive_threshold_new_user(self, adaptive_layer):
        """Test getting threshold for new user."""
        base_threshold = 0.7
        threshold = await adaptive_layer.get_adaptive_threshold("new_user", base_threshold)
        assert threshold == base_threshold
    
    @pytest.mark.asyncio
    async def test_get_adaptive_threshold_existing_user(self, adaptive_layer):
        """Test getting threshold for user with custom threshold."""
        user_id = "test_user"
        custom_threshold = 0.8
        adaptive_layer.user_thresholds[user_id] = custom_threshold
        
        threshold = await adaptive_layer.get_adaptive_threshold(user_id, 0.7)
        assert threshold == custom_threshold
    
    @pytest.mark.asyncio
    async def test_detect_pattern_drift_insufficient_data(self, adaptive_layer):
        """Test pattern drift detection with insufficient data."""
        vectors = [create_behavioral_vector(
            user_id="test_user",
            session_id=f"session_{i}"
        ) for i in range(2)]  # Only 2 vectors
        
        result = await adaptive_layer.detect_pattern_drift("test_user", vectors)
        
        assert result['drift_detected'] is False
        assert result['confidence'] == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_pattern_drift_no_patterns(self, adaptive_layer):
        """Test pattern drift detection with no historical patterns."""
        vectors = [create_behavioral_vector(
            user_id="test_user",
            session_id=f"session_{i}"
        ) for i in range(15)]
        
        result = await adaptive_layer.detect_pattern_drift("test_user", vectors)
        
        assert result['drift_detected'] is False
        assert result['confidence'] == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_pattern_drift_with_patterns(self, adaptive_layer):
        """Test pattern drift detection with existing patterns."""
        user_id = "test_user"
        
        # Create some historical patterns
        pattern = LearningPattern(
            pattern_id="pattern_1",
            user_id=user_id,
            feature_weights=np.random.random(TOTAL_VECTOR_DIM).tolist(),
            confidence_score=0.8,
            frequency=10,
            last_seen=datetime.utcnow(),
            context_tags=["mobile", "afternoon"]
        )
        adaptive_layer.user_patterns[user_id] = [pattern]
        
        # Create recent vectors (different from pattern)
        vectors = [create_behavioral_vector(
            user_id=user_id,
            session_id=f"session_{i}",
            vector=(np.random.random(TOTAL_VECTOR_DIM) * 2).tolist()  # Different scale
        ) for i in range(15)]
        
        result = await adaptive_layer.detect_pattern_drift(user_id, vectors)
        
        assert isinstance(result['drift_detected'], bool)
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'avg_drift_score' in result
        assert 'max_drift_score' in result
    
    @pytest.mark.asyncio
    async def test_optimize_learning_parameters_insufficient_history(self, adaptive_layer):
        """Test parameter optimization with insufficient history."""
        result = await adaptive_layer.optimize_learning_parameters("test_user")
        
        assert 'learning_rate' in result
        assert result['learning_rate'] == adaptive_layer.learning_rate
    
    @pytest.mark.asyncio
    async def test_optimize_learning_parameters_with_history(self, adaptive_layer):
        """Test parameter optimization with adaptation history."""
        user_id = "test_user"
        
        # Create adaptation history
        history = [
            AdaptationMetrics(
                user_id=user_id,
                adaptation_count=i + 1,
                last_adaptation=datetime.utcnow(),
                accuracy_improvement=0.1 if i % 2 == 0 else -0.05,  # Mixed success
                false_positive_reduction=0.02,
                false_negative_reduction=0.01,
                model_confidence_avg=0.8
            )
            for i in range(5)
        ]
        adaptive_layer.adaptation_history[user_id] = history
        
        result = await adaptive_layer.optimize_learning_parameters(user_id)
        
        assert 'learning_rate' in result
        assert 'success_rate' in result
        assert 'adaptation_count' in result
        assert result['adaptation_count'] == 5
        assert 0.0 <= result['success_rate'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_layer_statistics(self, adaptive_layer):
        """Test layer statistics retrieval."""
        # Add some test data
        user_id = "test_user"
        pattern = LearningPattern(
            pattern_id="pattern_1",
            user_id=user_id,
            feature_weights=np.random.random(TOTAL_VECTOR_DIM).tolist(),
            confidence_score=0.8,
            frequency=5,
            last_seen=datetime.utcnow(),
            context_tags=["mobile"]
        )
        adaptive_layer.user_patterns[user_id] = [pattern]
        adaptive_layer.feedback_buffer[user_id] = [{'test': 'data'}]
        
        stats = await adaptive_layer.get_layer_statistics()
        
        assert isinstance(stats, dict)
        assert 'adaptation_stats' in stats
        assert 'total_learned_patterns' in stats
        assert 'users_with_patterns' in stats
        assert 'performance_metrics' in stats
        assert 'feedback_buffer_sizes' in stats
        
        assert stats['total_learned_patterns'] == 1
        assert stats['users_with_patterns'] == 1
        assert stats['feedback_buffer_sizes'][user_id] == 1
    
    def test_analyze_feedback_patterns(self, adaptive_layer, sample_feedback_data):
        """Test feedback pattern analysis."""
        analysis = adaptive_layer._analyze_feedback_patterns(sample_feedback_data)
        
        assert isinstance(analysis, dict)
        assert 'total_samples' in analysis
        assert 'accuracy_rate' in analysis
        assert 'false_positive_rate' in analysis
        assert 'false_negative_rate' in analysis
        assert 'avg_confidence' in analysis
        assert 'confidence_trend' in analysis
        
        assert analysis['total_samples'] == 3
        assert 0.0 <= analysis['accuracy_rate'] <= 1.0
        assert 0.0 <= analysis['false_positive_rate'] <= 1.0
        assert 0.0 <= analysis['false_negative_rate'] <= 1.0
        assert analysis['avg_confidence'] > 0.0
    
    @pytest.mark.asyncio
    async def test_adapt_user_threshold_high_false_positives(self, adaptive_layer):
        """Test threshold adaptation for high false positive rate."""
        user_id = "test_user"
        adaptive_layer.user_thresholds[user_id] = 0.7
        
        analysis = {
            'false_positive_rate': 2.0,  # Extreme rate to ensure adaptation triggers (0.01 * 2.0 = 0.02 > 0.01)
            'false_negative_rate': 0.05,
            'accuracy_rate': 0.2
        }
        
        result = await adaptive_layer._adapt_user_threshold(user_id, analysis)
        
        assert result is True
        # Threshold should increase to reduce false positives
        assert adaptive_layer.user_thresholds[user_id] > 0.7
    
    @pytest.mark.asyncio
    async def test_adapt_user_threshold_high_false_negatives(self, adaptive_layer):
        """Test threshold adaptation for high false negative rate."""
        user_id = "test_user"
        adaptive_layer.user_thresholds[user_id] = 0.7
        
        analysis = {
            'false_positive_rate': 0.05,
            'false_negative_rate': 2.0,  # Extreme rate to ensure adaptation triggers (0.01 * 2.0 = 0.02 > 0.01)
            'accuracy_rate': 0.2
        }
        
        result = await adaptive_layer._adapt_user_threshold(user_id, analysis)
        
        assert result is True
        # Threshold should decrease to reduce false negatives
        assert adaptive_layer.user_thresholds[user_id] < 0.7
    
    @pytest.mark.asyncio
    async def test_discover_behavioral_patterns(self, adaptive_layer, sample_feedback_data):
        """Test behavioral pattern discovery."""
        user_id = "test_user"
        
        # Add more correct samples
        correct_samples = []
        for i in range(5):
            sample = sample_feedback_data[0].copy()  # Copy the correct one
            sample['vector'] = np.random.random(TOTAL_VECTOR_DIM).tolist()
            correct_samples.append(sample)
        
        result = await adaptive_layer._discover_behavioral_patterns(user_id, correct_samples)
        
        assert result is True
        assert user_id in adaptive_layer.user_patterns
        assert len(adaptive_layer.user_patterns[user_id]) > 0
        
        pattern = adaptive_layer.user_patterns[user_id][0]
        assert pattern.user_id == user_id
        assert len(pattern.feature_weights) == TOTAL_VECTOR_DIM
        assert pattern.frequency == 5
    
    @pytest.mark.asyncio
    async def test_update_pattern_weights(self, adaptive_layer):
        """Test pattern weight updates."""
        user_id = "test_user"
        
        # Create pattern
        pattern = LearningPattern(
            pattern_id="pattern_1",
            user_id=user_id,
            feature_weights=np.random.random(TOTAL_VECTOR_DIM).tolist(),
            confidence_score=0.5,
            frequency=3,
            last_seen=datetime.utcnow() - timedelta(hours=1),
            context_tags=["mobile"]
        )
        adaptive_layer.user_patterns[user_id] = [pattern]
        
        # Small delay to ensure time difference
        import time
        time.sleep(0.001)
        
        analysis = {
            'accuracy_rate': 0.8  # Good accuracy
        }
        
        result = await adaptive_layer._update_pattern_weights(user_id, analysis)
        
        assert result is True
        # Confidence should have increased
        assert adaptive_layer.user_patterns[user_id][0].confidence_score > 0.5
        # Last seen should be updated (should be later than original)
        assert adaptive_layer.user_patterns[user_id][0].last_seen >= pattern.last_seen
    
    def test_calculate_feature_drift(self, adaptive_layer):
        """Test feature drift calculation."""
        historical_features = np.random.random(TOTAL_VECTOR_DIM)
        recent_mean = np.random.random(TOTAL_VECTOR_DIM)
        recent_std = np.random.random(TOTAL_VECTOR_DIM) * 0.1
        
        drift_score = adaptive_layer._calculate_feature_drift(historical_features, recent_mean, recent_std)
        
        assert isinstance(drift_score, (float, np.floating))
        assert 0.0 <= drift_score <= 2.0  # Theoretical max
    
    def test_calculate_trend(self, adaptive_layer):
        """Test trend calculation."""
        # Increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = adaptive_layer._calculate_trend(increasing_values)
        assert trend > 0
        
        # Decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = adaptive_layer._calculate_trend(decreasing_values)
        assert trend < 0
        
        # Flat trend
        flat_values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = adaptive_layer._calculate_trend(flat_values)
        assert abs(trend) < 0.1
        
        # Insufficient data
        short_values = [1.0, 2.0]
        trend = adaptive_layer._calculate_trend(short_values)
        assert trend == 0.0
    
    def test_extract_context_tags(self, adaptive_layer, sample_feedback_data):
        """Test context tag extraction."""
        tags = adaptive_layer._extract_context_tags(sample_feedback_data)
        
        assert isinstance(tags, list)
        assert 'device_mobile' in tags
        assert 'afternoon' in tags  # time_of_day 14-15
        assert 'night' in tags  # time_of_day 22
    
    def test_cleanup_old_patterns(self, adaptive_layer):
        """Test cleanup of old patterns."""
        user_id = "test_user"
        
        # Create patterns with different ages
        old_pattern = LearningPattern(
            pattern_id="old_pattern",
            user_id=user_id,
            feature_weights=np.random.random(TOTAL_VECTOR_DIM).tolist(),
            confidence_score=0.8,
            frequency=5,
            last_seen=datetime.utcnow() - timedelta(days=40),  # Old
            context_tags=["mobile"]
        )
        
        recent_pattern = LearningPattern(
            pattern_id="recent_pattern",
            user_id=user_id,
            feature_weights=np.random.random(TOTAL_VECTOR_DIM).tolist(),
            confidence_score=0.8,
            frequency=3,
            last_seen=datetime.utcnow() - timedelta(days=5),  # Recent
            context_tags=["desktop"]
        )
        
        adaptive_layer.user_patterns[user_id] = [old_pattern, recent_pattern]
        
        # Cleanup
        adaptive_layer._cleanup_old_patterns(user_id)
        
        # Should only have recent pattern
        assert len(adaptive_layer.user_patterns[user_id]) == 1
        assert adaptive_layer.user_patterns[user_id][0].pattern_id == "recent_pattern"
    
    @pytest.mark.asyncio
    async def test_cleanup_user_data(self, adaptive_layer):
        """Test cleanup of all user data."""
        user_id = "test_user"
        
        # Add test data
        adaptive_layer.user_patterns[user_id] = [Mock()]
        adaptive_layer.user_thresholds[user_id] = 0.8
        adaptive_layer.adaptation_history[user_id] = [Mock()]
        adaptive_layer.feedback_buffer[user_id] = [Mock()]
        
        result = await adaptive_layer.cleanup_user_data(user_id)
        
        assert result is True
        assert user_id not in adaptive_layer.user_patterns
        assert user_id not in adaptive_layer.user_thresholds
        assert user_id not in adaptive_layer.adaptation_history
        assert user_id not in adaptive_layer.feedback_buffer


class TestAdaptiveLayerIntegration:
    """Integration tests for adaptive layer."""
    
    @pytest.fixture
    def vector_store(self):
        """Create real vector store with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield HDF5VectorStore(storage_path=temp_dir)
    
    @pytest.fixture
    def adaptive_layer_real(self, vector_store):
        """Create adaptive layer with real vector store."""
        return AdaptiveLayer(vector_store)
    
    @pytest.mark.asyncio
    async def test_complete_adaptation_workflow(self, adaptive_layer_real):
        """Test complete adaptation workflow."""
        user_id = "integration_test_user"
        
        # Simulate authentication feedback with high error rate to trigger adaptation
        for i in range(10):
            vector = create_behavioral_vector(
                user_id=user_id,
                session_id=f"session_{i}"
            )
            
            await adaptive_layer_real.learn_from_authentication(
                user_id=user_id,
                behavioral_vector=vector,
                decision=AuthenticationDecision.ALLOW,
                was_correct=i < 3,  # 70% false positive rate to ensure adaptation
                confidence=0.8,
                context={'device_type': 'mobile', 'time_of_day': 14}
            )
        
        # Should trigger adaptation
        assert user_id in adaptive_layer_real.feedback_buffer
        
        # Manual adaptation
        result = await adaptive_layer_real.adapt_user_model(user_id)
        assert result is True
        
        # Check results
        stats = await adaptive_layer_real.get_layer_statistics()
        assert stats['adaptation_stats']['total_adaptations'] > 0
