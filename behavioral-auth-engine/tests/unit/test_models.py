"""
Unit tests for data models.
"""

import pytest
from datetime import datetime
from src.data.models import (
    BehavioralFeatures, BehavioralVector, UserProfile, 
    AuthenticationRequest, AuthenticationResponse,
    SessionPhase, RiskLevel, AuthenticationDecision,
    DriftDetectionResult, ModelPerformanceMetrics
)
from src.utils.constants import TOTAL_VECTOR_DIM


class TestBehavioralFeatures:
    """Test behavioral features model."""
    
    def test_valid_behavioral_features(self):
        """Test creating valid behavioral features."""
        features = BehavioralFeatures(
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
        
        assert features.typing_speed == 45.5
        assert len(features.keystroke_intervals) == 3
        assert features.device_orientation == "portrait"
    
    def test_negative_values_validation(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValueError):
            BehavioralFeatures(
                typing_speed=45.5,
                keystroke_intervals=[-0.1, 0.15, 0.12],  # negative value
                typing_rhythm_variance=0.05,
                backspace_frequency=0.02,
                touch_pressure=[0.7, 0.8, 0.75],
                touch_duration=[0.2, 0.25, 0.22],
                touch_area=[100.0, 105.0, 102.0],
                swipe_velocity=[2.5, 3.0, 2.8],
                touch_coordinates=[{"x": 100, "y": 200}],
                navigation_patterns=["home", "menu"],
                screen_time_distribution={"home": 30.0},
                interaction_frequency=0.5,
                session_duration=120.0,
                device_orientation="portrait",
                time_of_day=14,
                day_of_week=1,
                app_version="1.0.0"
            )


class TestBehavioralVector:
    """Test behavioral vector model."""
    
    def test_valid_vector_creation(self):
        """Test creating a valid behavioral vector."""
        features = BehavioralFeatures(
            typing_speed=45.5,
            keystroke_intervals=[0.1, 0.15, 0.12],
            typing_rhythm_variance=0.05,
            backspace_frequency=0.02,
            touch_pressure=[0.7, 0.8, 0.75],
            touch_duration=[0.2, 0.25, 0.22],
            touch_area=[100.0, 105.0, 102.0],
            swipe_velocity=[2.5, 3.0, 2.8],
            touch_coordinates=[{"x": 100, "y": 200}],
            navigation_patterns=["home", "menu"],
            screen_time_distribution={"home": 30.0},
            interaction_frequency=0.5,
            session_duration=120.0,
            device_orientation="portrait",
            time_of_day=14,
            day_of_week=1,
            app_version="1.0.0"
        )
        
        vector = BehavioralVector(
            user_id="user123",
            session_id="session456",
            vector=[0.1] * TOTAL_VECTOR_DIM,  # 90-dimensional vector
            feature_source=features
        )
        
        assert vector.user_id == "user123"
        assert vector.session_id == "session456"
        assert len(vector.vector) == TOTAL_VECTOR_DIM
        assert vector.confidence_score == 1.0
    
    def test_invalid_vector_dimension(self):
        """Test that incorrect vector dimensions are rejected."""
        features = BehavioralFeatures(
            typing_speed=45.5,
            keystroke_intervals=[0.1],
            typing_rhythm_variance=0.05,
            backspace_frequency=0.02,
            touch_pressure=[0.7],
            touch_duration=[0.2],
            touch_area=[100.0],
            swipe_velocity=[2.5],
            touch_coordinates=[{"x": 100, "y": 200}],
            navigation_patterns=["home"],
            screen_time_distribution={"home": 30.0},
            interaction_frequency=0.5,
            session_duration=120.0,
            device_orientation="portrait",
            time_of_day=14,
            day_of_week=1,
            app_version="1.0.0"
        )
        
        with pytest.raises(ValueError):
            BehavioralVector(
                user_id="user123",
                session_id="session456",
                vector=[0.1] * 50,  # Wrong dimension
                feature_source=features
            )


class TestUserProfile:
    """Test user profile model."""
    
    def test_profile_creation(self):
        """Test creating a user profile."""
        profile = UserProfile(user_id="user123")
        
        assert profile.user_id == "user123"
        assert profile.session_count == 0
        assert profile.current_phase == SessionPhase.LEARNING
        assert profile.risk_threshold == 0.8
    
    def test_add_session_vector(self):
        """Test adding session vectors to profile."""
        profile = UserProfile(user_id="user123")
        
        features = BehavioralFeatures(
            typing_speed=45.5,
            keystroke_intervals=[0.1],
            typing_rhythm_variance=0.05,
            backspace_frequency=0.02,
            touch_pressure=[0.7],
            touch_duration=[0.2],
            touch_area=[100.0],
            swipe_velocity=[2.5],
            touch_coordinates=[{"x": 100, "y": 200}],
            navigation_patterns=["home"],
            screen_time_distribution={"home": 30.0},
            interaction_frequency=0.5,
            session_duration=120.0,
            device_orientation="portrait",
            time_of_day=14,
            day_of_week=1,
            app_version="1.0.0"
        )
        
        vector = BehavioralVector(
            user_id="user123",
            session_id="session456",
            vector=[0.1] * TOTAL_VECTOR_DIM,
            feature_source=features
        )
        
        profile.add_session_vector(vector)
        
        assert profile.session_count == 1
        assert len(profile.recent_vectors) == 1
    
    def test_phase_updates(self):
        """Test learning phase updates."""
        profile = UserProfile(user_id="user123")
        
        # Test learning phase
        profile.session_count = 2
        profile.update_phase()
        assert profile.current_phase == SessionPhase.LEARNING
        
        # Test gradual risk phase
        profile.session_count = 5
        profile.update_phase()
        assert profile.current_phase == SessionPhase.GRADUAL_RISK
        
        # Test full auth phase
        profile.session_count = 15
        profile.update_phase()
        assert profile.current_phase == SessionPhase.FULL_AUTH


class TestModelPerformanceMetrics:
    """Test model performance metrics."""
    
    def test_metrics_calculation(self):
        """Test calculated metrics."""
        metrics = ModelPerformanceMetrics(
            model_name="test_model",
            true_positives=80,
            false_positives=10,
            true_negatives=85,
            false_negatives=5
        )
        
        assert metrics.accuracy == 0.9166666666666666  # (80+85)/(80+10+85+5)
        assert metrics.precision == 0.8888888888888888  # 80/(80+10)
        assert metrics.recall == 0.9411764705882353  # 80/(80+5)
        assert metrics.f1_score > 0.9  # Should be around 0.914
    
    def test_zero_division_handling(self):
        """Test that zero division is handled gracefully."""
        metrics = ModelPerformanceMetrics(
            model_name="test_model",
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0
        )
        
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
