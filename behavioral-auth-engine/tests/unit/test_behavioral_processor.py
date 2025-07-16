"""
Unit tests for behavioral data processor.
"""

import pytest
import numpy as np
from src.data.behavioral_processor import BehavioralProcessor
from src.data.models import BehavioralFeatures
from src.utils.constants import TOTAL_VECTOR_DIM


class TestBehavioralProcessor:
    """Test behavioral data processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a behavioral processor instance."""
        return BehavioralProcessor()
    
    @pytest.fixture
    def sample_behavioral_features(self):
        """Create sample behavioral features for testing."""
        return BehavioralFeatures(
            typing_speed=45.5,
            keystroke_intervals=[0.1, 0.15, 0.12, 0.08, 0.13],
            typing_rhythm_variance=0.05,
            backspace_frequency=0.02,
            typing_pressure=[0.5, 0.6, 0.55, 0.7, 0.65],
            touch_pressure=[0.7, 0.8, 0.75, 0.9, 0.85],
            touch_duration=[0.2, 0.25, 0.22, 0.18, 0.24],
            touch_area=[100.0, 105.0, 102.0, 98.0, 103.0],
            swipe_velocity=[2.5, 3.0, 2.8, 3.2, 2.9],
            touch_coordinates=[
                {"x": 100, "y": 200}, 
                {"x": 150, "y": 250}, 
                {"x": 120, "y": 180},
                {"x": 140, "y": 220}
            ],
            navigation_patterns=["home", "menu", "profile", "settings", "home"],
            screen_time_distribution={"home": 30.0, "menu": 10.0, "profile": 20.0, "settings": 15.0},
            interaction_frequency=0.5,
            session_duration=120.0,
            device_orientation="portrait",
            time_of_day=14,
            day_of_week=1,
            app_version="1.0.0"
        )
    
    @pytest.fixture
    def minimal_behavioral_features(self):
        """Create minimal behavioral features for testing edge cases."""
        return BehavioralFeatures(
            typing_speed=30.0,
            keystroke_intervals=[],
            typing_rhythm_variance=0.0,
            backspace_frequency=0.0,
            typing_pressure=[],
            touch_pressure=[0.5],
            touch_duration=[0.1],
            touch_area=[50.0],
            swipe_velocity=[1.0],
            touch_coordinates=[{"x": 50, "y": 100}],
            navigation_patterns=["home"],
            screen_time_distribution={"home": 10.0},
            interaction_frequency=0.1,
            session_duration=30.0,
            device_orientation="portrait",
            time_of_day=10,
            day_of_week=0,
            app_version="1.0.0"
        )
    
    @pytest.mark.asyncio
    async def test_process_behavioral_data(self, processor, sample_behavioral_features):
        """Test processing behavioral data into a vector."""
        user_id = "test_user_123"
        session_id = "session_456"
        
        vector = await processor.process_behavioral_data(
            sample_behavioral_features,
            user_id,
            session_id
        )
        
        assert vector.user_id == user_id
        assert vector.session_id == session_id
        assert len(vector.vector) == TOTAL_VECTOR_DIM
        assert vector.feature_source == sample_behavioral_features
        assert 0.0 <= vector.confidence_score <= 1.0
        
        # Check that all vector values are finite
        assert all(np.isfinite(v) for v in vector.vector)
    
    @pytest.mark.asyncio
    async def test_process_minimal_data(self, processor, minimal_behavioral_features):
        """Test processing minimal behavioral data."""
        user_id = "test_user_minimal"
        session_id = "session_minimal"
        
        vector = await processor.process_behavioral_data(
            minimal_behavioral_features,
            user_id,
            session_id
        )
        
        assert vector.user_id == user_id
        assert vector.session_id == session_id
        assert len(vector.vector) == TOTAL_VECTOR_DIM
        
        # Confidence should be lower for minimal data
        assert vector.confidence_score < 0.8
    
    def test_extract_typing_features(self, processor, sample_behavioral_features):
        """Test typing feature extraction."""
        features = processor._extract_typing_features(sample_behavioral_features)
        
        assert len(features) == 25  # TYPING_FEATURES_DIM
        assert all(np.isfinite(f) for f in features)
        
        # First feature should be typing speed
        assert features[0] == sample_behavioral_features.typing_speed
    
    def test_extract_typing_features_empty_data(self, processor):
        """Test typing feature extraction with empty data."""
        empty_features = BehavioralFeatures(
            typing_speed=0.0,
            keystroke_intervals=[],
            typing_rhythm_variance=0.0,
            backspace_frequency=0.0,
            typing_pressure=[],
            touch_pressure=[],
            touch_duration=[],
            touch_area=[],
            swipe_velocity=[],
            touch_coordinates=[],
            navigation_patterns=[],
            screen_time_distribution={},
            interaction_frequency=0.0,
            session_duration=0.0,
            device_orientation="unknown",
            time_of_day=0,
            day_of_week=0,
            app_version="0.0.0"
        )
        
        features = processor._extract_typing_features(empty_features)
        
        assert len(features) == 25
        assert all(np.isfinite(f) for f in features)
    
    def test_extract_touch_features(self, processor, sample_behavioral_features):
        """Test touch feature extraction."""
        features = processor._extract_touch_features(sample_behavioral_features)
        
        assert len(features) == 30  # TOUCH_FEATURES_DIM
        assert all(np.isfinite(f) for f in features)
    
    def test_extract_navigation_features(self, processor, sample_behavioral_features):
        """Test navigation feature extraction."""
        features = processor._extract_navigation_features(sample_behavioral_features)
        
        assert len(features) == 20  # NAVIGATION_FEATURES_DIM
        assert all(np.isfinite(f) for f in features)
        
        # First feature should be number of navigation patterns
        assert features[0] == len(sample_behavioral_features.navigation_patterns)
    
    def test_extract_contextual_features(self, processor, sample_behavioral_features):
        """Test contextual feature extraction."""
        features = processor._extract_contextual_features(sample_behavioral_features)
        
        assert len(features) == 15  # CONTEXTUAL_FEATURES_DIM
        assert all(np.isfinite(f) for f in features)
        
        # Check time normalization
        assert 0.0 <= features[0] <= 1.0  # time_of_day normalized
        assert 0.0 <= features[1] <= 1.0  # day_of_week normalized
    
    def test_calculate_confidence_score(self, processor, sample_behavioral_features, minimal_behavioral_features):
        """Test confidence score calculation."""
        # Full data should have high confidence
        full_confidence = processor._calculate_confidence_score(sample_behavioral_features)
        assert 0.5 <= full_confidence <= 1.0
        
        # Minimal data should have lower confidence
        minimal_confidence = processor._calculate_confidence_score(minimal_behavioral_features)
        assert minimal_confidence < full_confidence
        assert minimal_confidence >= 0.1  # Minimum confidence
    
    def test_pad_or_truncate(self, processor):
        """Test feature padding and truncation."""
        # Test padding
        short_features = [1.0, 2.0, 3.0]
        padded = processor._pad_or_truncate(short_features, 5)
        assert len(padded) == 5
        assert padded == [1.0, 2.0, 3.0, 0.0, 0.0]
        
        # Test truncation
        long_features = [1.0, 2.0, 3.0, 4.0, 5.0]
        truncated = processor._pad_or_truncate(long_features, 3)
        assert len(truncated) == 3
        assert truncated == [1.0, 2.0, 3.0]
        
        # Test exact size
        exact_features = [1.0, 2.0, 3.0]
        unchanged = processor._pad_or_truncate(exact_features, 3)
        assert len(unchanged) == 3
        assert unchanged == [1.0, 2.0, 3.0]
    
    def test_typing_rhythm_consistency(self, processor):
        """Test typing rhythm consistency calculation."""
        # Consistent intervals
        consistent_intervals = [0.1, 0.1, 0.1, 0.1]
        consistency = processor._calculate_typing_rhythm_consistency(consistent_intervals)
        assert consistency > 0.5
        
        # Inconsistent intervals
        inconsistent_intervals = [0.05, 0.3, 0.01, 0.5]
        inconsistency = processor._calculate_typing_rhythm_consistency(inconsistent_intervals)
        assert inconsistency < consistency
        
        # Empty intervals
        empty_consistency = processor._calculate_typing_rhythm_consistency([])
        assert empty_consistency == 0.0
    
    def test_navigation_entropy(self, processor):
        """Test navigation entropy calculation."""
        # Uniform distribution (high entropy)
        uniform_patterns = ["home", "menu", "profile", "settings"]
        high_entropy = processor._calculate_navigation_entropy(uniform_patterns)
        
        # Repeated pattern (low entropy)
        repeated_patterns = ["home", "home", "home", "menu"]
        low_entropy = processor._calculate_navigation_entropy(repeated_patterns)
        
        assert high_entropy > low_entropy
        
        # Empty patterns
        empty_entropy = processor._calculate_navigation_entropy([])
        assert empty_entropy == 0.0
    
    def test_touch_coordinate_patterns(self, processor):
        """Test touch coordinate pattern analysis."""
        # Close coordinates
        close_coords = [
            {"x": 100, "y": 100},
            {"x": 101, "y": 101},
            {"x": 102, "y": 102}
        ]
        close_distance = processor._calculate_touch_coordinate_patterns(close_coords)
        
        # Far coordinates
        far_coords = [
            {"x": 0, "y": 0},
            {"x": 100, "y": 100},
            {"x": 200, "y": 200}
        ]
        far_distance = processor._calculate_touch_coordinate_patterns(far_coords)
        
        assert far_distance > close_distance
        
        # Empty coordinates
        empty_distance = processor._calculate_touch_coordinate_patterns([])
        assert empty_distance == 0.0
    
    def test_normalize_features(self, processor):
        """Test feature normalization."""
        features = [1.0, 5.0, 10.0, 2.0, 8.0]
        normalized = processor._normalize_features(features)
        
        assert len(normalized) == len(features)
        assert all(np.isfinite(f) for f in normalized)
        
        # For min-max normalization, values should be between 0 and 1
        # (though exact range depends on scaler configuration)
        assert all(f >= 0.0 for f in normalized)
    
    def test_handle_outliers(self, processor):
        """Test outlier handling."""
        # Features with outliers
        features_with_outliers = [1.0, 2.0, 3.0, 100.0, 4.0, 5.0]  # 100.0 is an outlier
        cleaned = processor._handle_outliers(features_with_outliers)
        
        assert len(cleaned) == len(features_with_outliers)
        assert all(np.isfinite(f) for f in cleaned)
        
        # Outlier should be reduced/clipped
        assert max(cleaned) < 100.0
    
    def test_time_consistency_score(self, processor):
        """Test time consistency score calculation."""
        # Business hours
        business_score = processor._calculate_time_consistency_score(14)  # 2 PM
        assert business_score > 0.5
        
        # Night time
        night_score = processor._calculate_time_consistency_score(2)  # 2 AM
        assert night_score < business_score
    
    def test_usage_pattern_score(self, processor):
        """Test usage pattern score calculation."""
        # Weekday business hours
        weekday_business = processor._calculate_usage_pattern_score(14, 1)  # Tuesday 2 PM
        
        # Weekend night
        weekend_night = processor._calculate_usage_pattern_score(2, 6)  # Saturday 2 AM
        
        assert weekday_business > weekend_night
    
    def test_device_familiarity_score(self, processor):
        """Test device familiarity score calculation."""
        portrait_score = processor._calculate_device_familiarity_score("portrait")
        landscape_score = processor._calculate_device_familiarity_score("landscape")
        unknown_score = processor._calculate_device_familiarity_score("unknown")
        
        assert portrait_score > landscape_score > unknown_score
