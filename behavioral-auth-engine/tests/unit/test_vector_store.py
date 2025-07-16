"""
Unit tests for vector storage implementation.
"""

import os
import tempfile
import pytest
import asyncio
from datetime import datetime
from src.core.vector_store import HDF5VectorStore
from src.data.models import BehavioralFeatures, BehavioralVector, UserProfile
from src.utils.constants import TOTAL_VECTOR_DIM


class TestHDF5VectorStore:
    """Test HDF5 vector storage implementation."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def vector_store(self, temp_storage_path):
        """Create a vector store instance for testing."""
        return HDF5VectorStore(storage_path=temp_storage_path)
    
    @pytest.fixture
    def sample_behavioral_features(self):
        """Create sample behavioral features for testing."""
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
    
    @pytest.fixture
    def sample_behavioral_vector(self, sample_behavioral_features):
        """Create sample behavioral vector for testing."""
        return BehavioralVector(
            user_id="test_user_123",
            session_id="session_456",
            vector=[0.1] * TOTAL_VECTOR_DIM,
            feature_source=sample_behavioral_features
        )
    
    @pytest.mark.asyncio
    async def test_store_vector(self, vector_store, sample_behavioral_vector):
        """Test storing a behavioral vector."""
        result = await vector_store.store_vector(
            sample_behavioral_vector.user_id, 
            sample_behavioral_vector
        )
        
        assert result is True
        
        # Verify file was created
        file_path = vector_store._get_user_file_path(sample_behavioral_vector.user_id)
        assert os.path.exists(file_path)
    
    @pytest.mark.asyncio
    async def test_get_user_vectors_empty(self, vector_store):
        """Test retrieving vectors for non-existent user."""
        vectors = await vector_store.get_user_vectors("non_existent_user")
        assert vectors == []
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_vectors(self, vector_store, sample_behavioral_features):
        """Test storing and retrieving multiple vectors."""
        user_id = "test_user_store_retrieve"
        vectors_to_store = []
        
        # Create multiple vectors
        for i in range(3):
            vector = BehavioralVector(
                user_id=user_id,
                session_id=f"session_{i}",
                vector=[0.1 + i * 0.1] * TOTAL_VECTOR_DIM,
                feature_source=sample_behavioral_features
            )
            vectors_to_store.append(vector)
        
        # Store all vectors
        for vector in vectors_to_store:
            result = await vector_store.store_vector(user_id, vector)
            assert result is True
        
        # Retrieve vectors
        retrieved_vectors = await vector_store.get_user_vectors(user_id)
        
        assert len(retrieved_vectors) == 3
        assert all(v.user_id == user_id for v in retrieved_vectors)
    
    @pytest.mark.asyncio
    async def test_get_user_vectors_with_limit(self, vector_store, sample_behavioral_features):
        """Test retrieving vectors with limit."""
        user_id = "test_user_limit"
        
        # Store 5 vectors
        for i in range(5):
            vector = BehavioralVector(
                user_id=user_id,
                session_id=f"session_{i}",
                vector=[0.1 + i * 0.1] * TOTAL_VECTOR_DIM,
                feature_source=sample_behavioral_features
            )
            await vector_store.store_vector(user_id, vector)
        
        # Retrieve with limit
        retrieved_vectors = await vector_store.get_user_vectors(user_id, limit=3)
        
        assert len(retrieved_vectors) == 3
    
    @pytest.mark.asyncio
    async def test_get_user_profile_new_user(self, vector_store):
        """Test getting profile for new user."""
        profile = await vector_store.get_user_profile("new_user")
        assert profile is None
    
    @pytest.mark.asyncio
    async def test_get_user_profile_existing_user(self, vector_store, sample_behavioral_vector):
        """Test getting profile for existing user."""
        user_id = "test_user_profile"
        sample_behavioral_vector.user_id = user_id
        
        # Store a vector
        await vector_store.store_vector(user_id, sample_behavioral_vector)
        
        # Get profile
        profile = await vector_store.get_user_profile(user_id)
        
        assert profile is not None
        assert profile.user_id == user_id
        assert profile.session_count >= 1
    
    @pytest.mark.asyncio
    async def test_update_user_profile(self, vector_store, sample_behavioral_vector):
        """Test updating user profile."""
        user_id = "test_user_update_profile"
        sample_behavioral_vector.user_id = user_id
        
        # Store initial vector
        await vector_store.store_vector(user_id, sample_behavioral_vector)
        
        # Get and update profile
        profile = await vector_store.get_user_profile(user_id)
        assert profile is not None
        
        profile.risk_threshold = 0.5
        profile.false_positive_rate = 0.1
        
        result = await vector_store.update_user_profile(profile)
        assert result is True
        
        # Retrieve updated profile
        updated_profile = await vector_store.get_user_profile(user_id)
        assert updated_profile is not None
        assert updated_profile.risk_threshold == 0.5
        assert updated_profile.false_positive_rate == 0.1
    
    @pytest.mark.asyncio
    async def test_delete_user_data(self, vector_store, sample_behavioral_vector):
        """Test deleting user data."""
        user_id = "test_user_delete"
        sample_behavioral_vector.user_id = user_id
        
        # Store vector
        await vector_store.store_vector(user_id, sample_behavioral_vector)
        
        # Verify file exists
        file_path = vector_store._get_user_file_path(user_id)
        assert os.path.exists(file_path)
        
        # Delete user data
        result = await vector_store.delete_user_data(user_id)
        assert result is True
        
        # Verify file is deleted
        assert not os.path.exists(file_path)
    
    @pytest.mark.asyncio
    async def test_get_similar_vectors_empty(self, vector_store):
        """Test similarity search with no stored vectors."""
        query_vector = [0.5] * TOTAL_VECTOR_DIM
        similar = await vector_store.get_similar_vectors(query_vector, "empty_user")
        
        assert similar == []
    
    @pytest.mark.asyncio
    async def test_get_similar_vectors(self, vector_store, sample_behavioral_features):
        """Test similarity search with stored vectors."""
        user_id = "test_user_similarity"
        
        # Store some vectors with different patterns
        base_vector = [0.5] * TOTAL_VECTOR_DIM
        vectors_to_store = [
            base_vector,  # Identical
            [v + 0.1 for v in base_vector],  # Slightly different
            [v + 0.5 for v in base_vector],  # More different
        ]
        
        for i, vector_data in enumerate(vectors_to_store):
            vector = BehavioralVector(
                user_id=user_id,
                session_id=f"session_{i}",
                vector=vector_data,
                feature_source=sample_behavioral_features
            )
            await vector_store.store_vector(user_id, vector)
        
        # Search for similar vectors
        query_vector = [0.5] * TOTAL_VECTOR_DIM
        similar = await vector_store.get_similar_vectors(query_vector, user_id, top_k=2)
        
        assert len(similar) == 2
        # First result should be most similar (identical)
        assert similar[0][1] > similar[1][1]  # Higher similarity score
    
    @pytest.mark.asyncio
    async def test_get_storage_stats(self, vector_store, sample_behavioral_vector):
        """Test storage statistics."""
        # Store some data
        await vector_store.store_vector("user1", sample_behavioral_vector)
        
        sample_behavioral_vector.user_id = "user2"
        await vector_store.store_vector("user2", sample_behavioral_vector)
        
        stats = await vector_store.get_storage_stats()
        
        assert stats['total_users'] >= 2
        assert stats['total_size_mb'] > 0
        assert stats['storage_path'] == vector_store.storage_path
    
    def test_user_file_path_generation(self, vector_store):
        """Test user file path generation."""
        user_id = "test_user_123"
        expected_path = os.path.join(vector_store.storage_path, "user_test_user_123.h5")
        actual_path = vector_store._get_user_file_path(user_id)
        
        assert actual_path == expected_path
    
    def test_user_lock_creation(self, vector_store):
        """Test user-specific lock creation."""
        user_id = "test_user_lock"
        
        # Get lock for first time
        lock1 = vector_store._get_user_lock(user_id)
        
        # Get lock for second time (should be same instance)
        lock2 = vector_store._get_user_lock(user_id)
        
        assert lock1 is lock2
