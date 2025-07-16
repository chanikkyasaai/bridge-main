"""
Unit tests for session manager.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from src.core.session_manager import SessionManager, SessionContext, SessionStatus
from src.core.vector_store import HDF5VectorStore
from src.data.models import AuthenticationRequest, BehavioralFeatures, SessionPhase
import tempfile


class TestSessionManager:
    """Test session manager functionality."""
    
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
    def session_manager(self, vector_store):
        """Create a session manager instance for testing."""
        return SessionManager(vector_store)
    
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
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a new session."""
        await session_manager.start()
        try:
            user_id = "test_user_123"
            session_id = await session_manager.create_session(
                user_id=user_id,
                device_id="device_123",
                ip_address="192.168.1.1",
                user_agent="TestAgent/1.0"
            )
            
            assert session_id is not None
            assert len(session_id) > 0
            
            # Verify session exists
            session = await session_manager.get_session(session_id)
            assert session is not None
            assert session.user_id == user_id
            assert session.device_id == "device_123"
            assert session.ip_address == "192.168.1.1"
            assert session.user_agent == "TestAgent/1.0"
            assert session.status == SessionStatus.ACTIVE
        finally:
            await session_manager.stop()
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, session_manager):
        """Test getting a non-existent session."""
        session = await session_manager.get_session("non_existent_session")
        assert session is None
    
    @pytest.mark.asyncio
    async def test_update_session_activity(self, session_manager):
        """Test updating session activity."""
        user_id = "test_user_activity"
        session_id = await session_manager.create_session(user_id)
        
        # Get initial activity time
        session = await session_manager.get_session(session_id)
        initial_activity = session.last_activity
        
        # Small delay to ensure timestamp difference
        await asyncio.sleep(0.01)
        
        # Update activity
        result = await session_manager.update_session_activity(session_id)
        assert result is True
        
        # Verify activity was updated
        updated_session = await session_manager.get_session(session_id)
        assert updated_session.last_activity > initial_activity
    
    @pytest.mark.asyncio
    async def test_terminate_session(self, session_manager):
        """Test terminating a session."""
        user_id = "test_user_terminate"
        session_id = await session_manager.create_session(user_id)
        
        # Verify session is active
        session = await session_manager.get_session(session_id)
        assert session.status == SessionStatus.ACTIVE
        
        # Terminate session
        result = await session_manager.terminate_session(session_id)
        assert result is True
        
        # Verify session is terminated
        terminated_session = await session_manager.get_session(session_id)
        assert terminated_session.status == SessionStatus.TERMINATED
    
    @pytest.mark.asyncio
    async def test_terminate_user_sessions(self, session_manager):
        """Test terminating all sessions for a user."""
        user_id = "test_user_terminate_all"
        
        # Create multiple sessions for the user
        session_ids = []
        for i in range(3):
            session_id = await session_manager.create_session(user_id)
            session_ids.append(session_id)
        
        # Verify all sessions are active
        for session_id in session_ids:
            session = await session_manager.get_session(session_id)
            assert session.status == SessionStatus.ACTIVE
        
        # Terminate all user sessions
        terminated_count = await session_manager.terminate_user_sessions(user_id)
        assert terminated_count == 3
        
        # Verify all sessions are terminated
        for session_id in session_ids:
            session = await session_manager.get_session(session_id)
            assert session.status == SessionStatus.TERMINATED
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, session_manager):
        """Test getting all sessions for a user."""
        user_id = "test_user_get_sessions"
        
        # Create multiple sessions
        session_ids = []
        for i in range(2):
            session_id = await session_manager.create_session(user_id)
            session_ids.append(session_id)
        
        # Get user sessions
        user_sessions = await session_manager.get_user_sessions(user_id)
        assert len(user_sessions) == 2
        assert all(session.user_id == user_id for session in user_sessions)
        assert all(session.status == SessionStatus.ACTIVE for session in user_sessions)
    
    @pytest.mark.asyncio
    async def test_process_authentication_request(self, session_manager, sample_behavioral_features):
        """Test processing an authentication request."""
        user_id = "test_user_auth"
        session_id = await session_manager.create_session(user_id)
        
        # Create authentication request
        auth_request = AuthenticationRequest(
            user_id=user_id,
            session_id=session_id,
            behavioral_data=sample_behavioral_features,
            device_id="device_123",
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        
        # Process request
        response = await session_manager.process_authentication_request(auth_request)
        
        assert response is not None
        assert response.user_id == user_id
        assert response.request_id == auth_request.request_id
        assert response.processing_time_ms > 0
        assert response.session_phase == SessionPhase.LEARNING  # New user should be in learning phase
        assert response.session_count >= 0
        assert response.decision is not None
        assert response.risk_level is not None
        assert 0.0 <= response.risk_score <= 1.0
        assert 0.0 <= response.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_session_statistics(self, session_manager):
        """Test getting session statistics."""
        # Create some sessions
        await session_manager.create_session("user1")
        await session_manager.create_session("user2")
        session_id = await session_manager.create_session("user1")  # Second session for user1
        
        # Terminate one session
        await session_manager.terminate_session(session_id)
        
        # Get statistics
        stats = await session_manager.get_session_statistics()
        
        assert stats["total_sessions"] >= 3
        assert stats["active_sessions"] >= 2
        assert stats["terminated_sessions"] >= 1
        assert stats["unique_users"] >= 2
        assert stats["max_sessions_per_user"] >= 1
        assert stats["avg_sessions_per_user"] >= 1.0


class TestSessionContext:
    """Test session context functionality."""
    
    def test_session_context_creation(self):
        """Test creating a session context."""
        user_id = "test_user"
        session_id = "test_session"
        now = datetime.utcnow()
        
        context = SessionContext(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE
        )
        
        assert context.user_id == user_id
        assert context.session_id == session_id
        assert context.status == SessionStatus.ACTIVE
        assert context.risk_scores == []
        assert context.decision_history == []
    
    def test_update_activity(self):
        """Test updating session activity."""
        now = datetime.utcnow()
        context = SessionContext(
            user_id="test",
            session_id="test",
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE
        )
        
        initial_activity = context.last_activity
        context.update_activity()
        
        assert context.last_activity >= initial_activity
    
    def test_add_risk_score(self):
        """Test adding risk scores."""
        context = SessionContext(
            user_id="test",
            session_id="test",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            status=SessionStatus.ACTIVE
        )
        
        # Add some risk scores
        context.add_risk_score(0.1)
        context.add_risk_score(0.3)
        context.add_risk_score(0.2)
        
        assert len(context.risk_scores) == 3
        assert context.risk_scores == [0.1, 0.3, 0.2]
        
        # Test average
        avg_score = context.get_average_risk_score()
        assert abs(avg_score - 0.2) < 0.01  # (0.1 + 0.3 + 0.2) / 3 = 0.2
    
    def test_risk_score_limit(self):
        """Test risk score list size limit."""
        context = SessionContext(
            user_id="test",
            session_id="test",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            status=SessionStatus.ACTIVE
        )
        
        # Add more than 50 scores
        for i in range(55):
            context.add_risk_score(i / 100.0)
        
        # Should only keep last 50
        assert len(context.risk_scores) == 50
        assert context.risk_scores[0] == 0.05  # Should start from index 5
    
    def test_session_expiration(self):
        """Test session expiration logic."""
        # Create session that's old enough to be expired
        old_time = datetime.utcnow() - timedelta(minutes=35)
        context = SessionContext(
            user_id="test",
            session_id="test",
            created_at=old_time,
            last_activity=old_time,
            status=SessionStatus.ACTIVE
        )
        
        # Should be expired with default 30-minute timeout
        assert context.is_expired() is True
        
        # Should not be expired with longer timeout
        assert context.is_expired(timeout_minutes=60) is False
    
    def test_session_already_terminated(self):
        """Test that terminated sessions are considered expired."""
        context = SessionContext(
            user_id="test",
            session_id="test",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            status=SessionStatus.TERMINATED
        )
        
        assert context.is_expired() is True
