"""
Unit tests for FAISS similarity search layer.
"""

import pytest
import numpy as np
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.layers.faiss_layer import FAISSLayer
from src.data.models import BehavioralVector, UserProfile, AuthenticationDecision, RiskLevel, SessionPhase, BehavioralFeatures
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
        vector = [0.5] * TOTAL_VECTOR_DIM
    
    return BehavioralVector(
        user_id=user_id,
        session_id=session_id,
        vector=vector,
        confidence_score=confidence,
        feature_source=create_sample_features()
    )


class TestFAISSLayer:
    """Test FAISS layer functionality."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            similarity_threshold=0.7,
            min_vectors_for_search=3,
            adaptive_learning_rate=0.01
        )
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock(spec=HDF5VectorStore)
        store.get_user_vectors = AsyncMock(return_value=[])
        return store
    
    @pytest.fixture
    def faiss_layer(self, mock_vector_store, settings):
        """Create FAISS layer instance."""
        try:
            return FAISSLayer(mock_vector_store, settings)
        except ImportError:
            pytest.skip("FAISS not available")
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample behavioral vectors."""
        vectors = []
        for i in range(5):
            vector = create_behavioral_vector(
                user_id="test_user",
                session_id=f"session_{i}",
                vector=[0.1 * i] * TOTAL_VECTOR_DIM
            )
            vectors.append(vector)
        return vectors
    
    @pytest.fixture
    def user_profile(self):
        """Create test user profile."""
        return UserProfile(
            user_id="test_user",
            session_count=5,
            current_phase=SessionPhase.GRADUAL_RISK
        )
    
    @pytest.mark.asyncio
    async def test_initialize_user_index_success(self, faiss_layer, mock_vector_store, sample_vectors):
        """Test successful user index initialization."""
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        
        result = await faiss_layer.initialize_user_index("test_user")
        
        assert result is True
        assert "test_user" in faiss_layer.user_indices
        assert "test_user" in faiss_layer.index_metadata
        assert faiss_layer.index_metadata["test_user"]["vector_count"] == 5
    
    @pytest.mark.asyncio
    async def test_initialize_user_index_insufficient_vectors(self, faiss_layer, mock_vector_store):
        """Test index initialization with insufficient vectors."""
        mock_vector_store.get_user_vectors.return_value = []  # No vectors
        
        result = await faiss_layer.initialize_user_index("test_user")
        
        assert result is False
        assert "test_user" not in faiss_layer.user_indices
    
    @pytest.mark.asyncio
    async def test_add_vector_to_index(self, faiss_layer, mock_vector_store, sample_vectors):
        """Test adding vector to existing index."""
        # Initialize index first
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        await faiss_layer.initialize_user_index("test_user")
        
        # Add new vector
        new_vector = create_behavioral_vector(
            user_id="test_user",
            session_id="new_session",
            vector=[0.5] * TOTAL_VECTOR_DIM,
            confidence=0.9
        )
        
        result = await faiss_layer.add_vector_to_index("test_user", new_vector)
        
        assert result is True
        assert faiss_layer.index_metadata["test_user"]["vector_count"] == 6
    
    @pytest.mark.asyncio
    async def test_compute_similarity_scores(self, faiss_layer, mock_vector_store, sample_vectors):
        """Test similarity score computation."""
        # Initialize index
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        await faiss_layer.initialize_user_index("test_user")
        
        # Query with similar vector
        query_vector = create_behavioral_vector(
            user_id="test_user",
            session_id="query",
            vector=[0.1] * TOTAL_VECTOR_DIM  # Similar to first vector
        )
        
        scores = await faiss_layer.compute_similarity_scores("test_user", query_vector, top_k=3)
        
        assert isinstance(scores, dict)
        assert len(scores) <= 3
        # Should have similarity scores (allow small floating point errors)
        for score in scores.values():
            assert -0.1 <= score <= 1.1  # Allow small floating point precision errors
    
    @pytest.mark.asyncio
    async def test_compute_similarity_scores_no_index(self, faiss_layer, mock_vector_store):
        """Test similarity computation with no existing index."""
        mock_vector_store.get_user_vectors.return_value = []
        
        query_vector = create_behavioral_vector(
            user_id="test_user",
            session_id="query",
            vector=[0.5] * TOTAL_VECTOR_DIM
        )
        
        scores = await faiss_layer.compute_similarity_scores("test_user", query_vector)
        
        assert scores == {}
    
    @pytest.mark.asyncio
    async def test_make_authentication_decision_learning(self, faiss_layer, mock_vector_store, user_profile):
        """Test authentication decision in learning phase."""
        # No historical data
        mock_vector_store.get_user_vectors.return_value = []
        
        user_profile.current_phase = SessionPhase.LEARNING
        
        query_vector = create_behavioral_vector(
            user_id="test_user",
            session_id="query",
            vector=[0.5] * TOTAL_VECTOR_DIM
        )
        
        decision, risk_level, risk_score, confidence, factors = await faiss_layer.make_authentication_decision(
            "test_user", query_vector, user_profile
        )
        
        assert decision == AuthenticationDecision.LEARN
        assert risk_level == RiskLevel.LOW
        assert 0.0 <= risk_score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert len(factors) > 0
    
    @pytest.mark.asyncio
    async def test_make_authentication_decision_allow(self, faiss_layer, mock_vector_store, sample_vectors, user_profile):
        """Test authentication decision with high similarity."""
        # Initialize with sample vectors
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        await faiss_layer.initialize_user_index("test_user")
        
        user_profile.current_phase = SessionPhase.FULL_AUTH
        
        # Query with very similar vector
        query_vector = create_behavioral_vector(
            user_id="test_user",
            session_id="query",
            vector=[0.1] * TOTAL_VECTOR_DIM  # Very similar to sample vectors
        )
        
        decision, risk_level, risk_score, confidence, factors = await faiss_layer.make_authentication_decision(
            "test_user", query_vector, user_profile
        )
        
        assert decision in [AuthenticationDecision.ALLOW, AuthenticationDecision.CHALLENGE]
        assert isinstance(risk_level, RiskLevel)
        assert 0.0 <= risk_score <= 1.0
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_optimize_user_index(self, faiss_layer, mock_vector_store, sample_vectors):
        """Test user index optimization."""
        # Initialize index
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        await faiss_layer.initialize_user_index("test_user")
        
        original_metadata = faiss_layer.index_metadata["test_user"].copy()
        
        # Add small delay to ensure timestamp difference
        import time
        time.sleep(0.001)
        
        # Optimize index
        result = await faiss_layer.optimize_user_index("test_user")
        
        assert result is True
        assert "test_user" in faiss_layer.user_indices
        # Should have updated metadata (timestamp should be different)
        assert faiss_layer.index_metadata["test_user"]["created_at"] >= original_metadata["created_at"]
    
    @pytest.mark.asyncio
    async def test_get_layer_statistics(self, faiss_layer, mock_vector_store, sample_vectors):
        """Test layer statistics retrieval."""
        # Initialize index
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        await faiss_layer.initialize_user_index("test_user")
        
        stats = await faiss_layer.get_layer_statistics()
        
        assert isinstance(stats, dict)
        assert "total_user_indices" in stats
        assert "search_stats" in stats
        assert "index_metadata" in stats
        assert "memory_usage_mb" in stats
        assert "settings" in stats
        
        assert stats["total_user_indices"] == 1
        assert isinstance(stats["memory_usage_mb"], (int, float))
    
    @pytest.mark.asyncio
    async def test_cleanup_user_index(self, faiss_layer, mock_vector_store, sample_vectors):
        """Test user index cleanup."""
        # Initialize index
        mock_vector_store.get_user_vectors.return_value = sample_vectors
        await faiss_layer.initialize_user_index("test_user")
        
        assert "test_user" in faiss_layer.user_indices
        
        # Cleanup
        result = await faiss_layer.cleanup_user_index("test_user")
        
        assert result is True
        assert "test_user" not in faiss_layer.user_indices
        assert "test_user" not in faiss_layer.index_metadata
        assert "test_user" not in faiss_layer.index_locks
    
    def test_normalize_vectors(self, faiss_layer):
        """Test vector normalization."""
        # Test vectors
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]  # Zero vector
        ])
        
        normalized = faiss_layer._normalize_vectors(vectors)
        
        assert normalized.shape == vectors.shape
        
        # Check normalization (except zero vector)
        for i in range(2):  # Skip zero vector
            norm = np.linalg.norm(normalized[i])
            assert abs(norm - 1.0) < 1e-6
        
        # Zero vector should remain as is (but normalized to avoid division by zero)
        assert np.allclose(normalized[2], [0.0, 0.0, 0.0])
    
    def test_update_search_stats(self, faiss_layer):
        """Test search statistics update."""
        start_time = datetime.utcnow()
        
        # Test successful search
        faiss_layer._update_search_stats(start_time, True)
        
        assert faiss_layer.search_stats['total_searches'] == 1
        assert faiss_layer.search_stats['successful_matches'] == 1
        assert faiss_layer.search_stats['failed_matches'] == 0
        assert faiss_layer.search_stats['avg_search_time_ms'] >= 0
        
        # Test failed search
        faiss_layer._update_search_stats(start_time, False)
        
        assert faiss_layer.search_stats['total_searches'] == 2
        assert faiss_layer.search_stats['successful_matches'] == 1
        assert faiss_layer.search_stats['failed_matches'] == 1
    
    def test_estimate_memory_usage(self, faiss_layer):
        """Test memory usage estimation."""
        # No indices initially
        memory_usage = faiss_layer._estimate_memory_usage()
        assert memory_usage == 0.0
        
        # Mock an index
        mock_index = Mock()
        mock_index.ntotal = 100  # 100 vectors
        faiss_layer.user_indices["test_user"] = mock_index
        
        memory_usage = faiss_layer._estimate_memory_usage()
        assert memory_usage > 0.0
    
    @pytest.mark.asyncio
    async def test_faiss_not_available(self, mock_vector_store, settings):
        """Test behavior when FAISS is not available."""
        with patch('src.layers.faiss_layer.faiss', None):
            with pytest.raises(ImportError, match="FAISS library is required"):
                FAISSLayer(mock_vector_store, settings)


class TestFAISSLayerIntegration:
    """Integration tests for FAISS layer with real vector store."""
    
    @pytest.fixture
    def vector_store(self):
        """Create real vector store with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield HDF5VectorStore(storage_path=temp_dir)
    
    @pytest.fixture
    def faiss_layer_real(self, vector_store):
        """Create FAISS layer with real vector store."""
        try:
            return FAISSLayer(vector_store)
        except ImportError:
            pytest.skip("FAISS not available")
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, faiss_layer_real, vector_store):
        """Test complete FAISS workflow with real data."""
        user_id = "integration_test_user"
        
        # Store some vectors
        vectors = []
        for i in range(5):
            vector = create_behavioral_vector(
                user_id=user_id,
                session_id=f"session_{i}",
                vector=np.random.random(TOTAL_VECTOR_DIM).tolist()
            )
            vectors.append(vector)
            await vector_store.store_vector(user_id, vector)
        
        # Initialize index
        result = await faiss_layer_real.initialize_user_index(user_id)
        assert result is True
        
        # Test similarity search
        query_vector = vectors[0]  # Use first vector as query
        scores = await faiss_layer_real.compute_similarity_scores(user_id, query_vector)
        
        assert len(scores) > 0
        # Should find itself with high similarity
        max_score = max(scores.values())
        assert max_score > 0.99  # Should be very similar to itself
        
        # Test authentication decision
        user_profile = UserProfile(user_id=user_id, session_count=5)
        decision, risk_level, risk_score, confidence, factors = await faiss_layer_real.make_authentication_decision(
            user_id, query_vector, user_profile
        )
        
        assert decision in [AuthenticationDecision.ALLOW, AuthenticationDecision.LEARN]
        assert isinstance(risk_level, RiskLevel)
        assert 0.0 <= risk_score <= 1.0
        assert 0.0 <= confidence <= 1.0
