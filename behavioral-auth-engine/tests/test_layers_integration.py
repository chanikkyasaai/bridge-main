"""
Integration Tests for Layers G, H, and J
Testing Session Graph Generator, GNN Anomaly Detection, and Policy Orchestration
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.layers.session_graph_generator import (
    SessionGraphGenerator, SessionGraph, BehavioralNode, BehavioralEdge,
    ActionType, TransitionType
)
from src.layers.gnn_anomaly_detector import (
    GNNAnomalyDetector, GNNAnomalyResult, AnomalyType
)
from src.layers.policy_orchestration_engine import (
    PolicyOrchestrationEngine, PolicyLevel, ContextualRiskFactors,
    PolicyDecisionResult
)
from src.data.models import (
    BehavioralVector, UserProfile, AuthenticationDecision, 
    RiskLevel, SessionPhase
)


class TestSessionGraphGenerator:
    """Test Layer G: Session Graph Generator"""
    
    @pytest.fixture
    def graph_generator(self):
        return SessionGraphGenerator()
    
    @pytest.fixture
    def sample_behavioral_events(self):
        """Sample mobile behavioral events for testing"""
        base_time = 1700000000000  # Timestamp in milliseconds
        
        return [
            {
                'event_type': 'touch',
                'timestamp': base_time,
                'x': 100,
                'y': 200,
                'pressure': 0.5,
                'duration': 150,
                'screen_id': 'login_screen'
            },
            {
                'event_type': 'typing',
                'timestamp': base_time + 500,
                'key_count': 5,
                'typing_speed': 120,
                'dwell_time': 80,
                'duration': 400,
                'screen_id': 'login_screen'
            },
            {
                'event_type': 'scroll',
                'timestamp': base_time + 1200,
                'velocity': 250,
                'direction': 'down',
                'distance': 300,
                'duration': 200,
                'screen_id': 'main_screen'
            },
            {
                'event_type': 'transaction_start',
                'timestamp': base_time + 2000,
                'amount': 1000,
                'beneficiary_type': 'new',
                'transaction_type': 'transfer',
                'duration': 100,
                'screen_id': 'transaction_screen'
            },
            {
                'event_type': 'mpin_entry',
                'timestamp': base_time + 3000,
                'duration': 300,
                'screen_id': 'mpin_screen'
            }
        ]
    
    def test_generate_session_graph_basic(self, graph_generator, sample_behavioral_events):
        """Test basic session graph generation"""
        user_id = "test_user"
        session_id = "test_session"
        
        graph = graph_generator.generate_session_graph(
            user_id, session_id, sample_behavioral_events
        )
        
        assert graph.user_id == user_id
        assert graph.session_id == session_id
        assert len(graph.nodes) == 5  # 5 events should create 5 nodes
        assert len(graph.edges) >= 4  # At least sequential edges
    
    def test_node_creation_from_events(self, graph_generator, sample_behavioral_events):
        """Test node creation from behavioral events"""
        user_id = "test_user"
        session_id = "test_session"
        
        graph = graph_generator.generate_session_graph(
            user_id, session_id, sample_behavioral_events
        )
        
        # Check node types
        action_types = [node.action_type for node in graph.nodes.values()]
        expected_types = [ActionType.TAP, ActionType.TYPING, ActionType.SCROLL, 
                         ActionType.TRANSACTION_START, ActionType.MPIN_ENTRY]
        
        for expected_type in expected_types:
            assert expected_type in action_types
    
    def test_edge_creation_timing(self, graph_generator, sample_behavioral_events):
        """Test edge creation with correct timing analysis"""
        user_id = "test_user"
        session_id = "test_session"
        
        graph = graph_generator.generate_session_graph(
            user_id, session_id, sample_behavioral_events
        )
        
        # Check transition types based on timing
        transition_types = [edge.transition_type for edge in graph.edges.values()]
        
        # First edge (touch -> typing: 500ms) should be RAPID or SEQUENTIAL
        # Later edges should be SEQUENTIAL or DELAYED
        assert TransitionType.RAPID in transition_types or TransitionType.SEQUENTIAL in transition_types
    
    def test_graph_feature_vector_generation(self, graph_generator, sample_behavioral_events):
        """Test conversion of graph to feature vector"""
        user_id = "test_user"
        session_id = "test_session"
        
        graph = graph_generator.generate_session_graph(
            user_id, session_id, sample_behavioral_events
        )
        
        feature_vector = graph_generator.get_graph_features_vector(graph)
        
        assert isinstance(feature_vector, np.ndarray)
        assert len(feature_vector) > 0
        assert not np.all(feature_vector == 0)  # Should have meaningful features
    
    def test_gnn_export_format(self, graph_generator, sample_behavioral_events):
        """Test export format for GNN analysis"""
        user_id = "test_user"
        session_id = "test_session"
        
        graph = graph_generator.generate_session_graph(
            user_id, session_id, sample_behavioral_events
        )
        
        gnn_data = graph_generator.export_graph_for_gnn(graph)
        
        assert 'node_features' in gnn_data
        assert 'edge_indices' in gnn_data
        assert 'edge_attributes' in gnn_data
        assert 'num_nodes' in gnn_data
        assert 'num_edges' in gnn_data
        
        assert gnn_data['num_nodes'] == len(graph.nodes)
        assert isinstance(gnn_data['node_features'], np.ndarray)
    
    def test_empty_events_handling(self, graph_generator):
        """Test handling of empty behavioral events"""
        user_id = "test_user"
        session_id = "test_session"
        
        graph = graph_generator.generate_session_graph(
            user_id, session_id, []
        )
        
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.user_id == user_id
        assert graph.session_id == session_id


class TestGNNAnomalyDetector:
    """Test Layer H: GNN-Based Anomaly Detection"""
    
    @pytest.fixture
    def gnn_detector(self):
        config = {
            'node_features': 10,
            'edge_features': 4,
            'hidden_dim': 32,  # Smaller for testing
            'num_layers': 2
        }
        return GNNAnomalyDetector(config)
    
    @pytest.fixture
    def sample_session_graph(self):
        """Create a sample session graph for testing"""
        graph = SessionGraph(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now()
        )
        
        # Add sample nodes
        for i in range(3):
            node = BehavioralNode(
                node_id=f"node_{i}",
                action_type=ActionType.TAP if i % 2 == 0 else ActionType.SCROLL,
                timestamp=datetime.now() + timedelta(seconds=i),
                duration=100 + i * 50,
                coordinates=(100 + i * 20, 200 + i * 10),
                sequence_position=i
            )
            graph.add_node(node)
        
        # Add sample edges
        for i in range(2):
            edge = BehavioralEdge(
                edge_id=f"edge_{i}",
                source_node=f"node_{i}",
                target_node=f"node_{i+1}",
                transition_type=TransitionType.SEQUENTIAL,
                time_gap=500,
                confidence=1.0
            )
            graph.add_edge(edge)
        
        return graph
    
    @pytest.fixture
    def sample_user_profile(self):
        return UserProfile(
            user_id="test_user",
            current_phase=SessionPhase.FULL_AUTH,
            drift_score=0.1,
            false_positive_rate=0.05
        )
    
    def test_anomaly_detection_basic(self, gnn_detector, sample_session_graph, sample_user_profile):
        """Test basic anomaly detection functionality"""
        result = gnn_detector.detect_anomalies(
            sample_session_graph, sample_user_profile
        )
        
        assert isinstance(result, GNNAnomalyResult)
        assert 0.0 <= result.anomaly_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.decision, AuthenticationDecision)
        assert isinstance(result.risk_level, RiskLevel)
        assert isinstance(result.graph_embeddings, np.ndarray)
    
    def test_fraud_pattern_detection(self, gnn_detector, sample_session_graph, sample_user_profile):
        """Test fraud pattern detection"""
        # Create a graph with rapid transaction pattern
        fraud_graph = SessionGraph(
            session_id="fraud_session",
            user_id="test_user",
            start_time=datetime.now()
        )
        
        # Add multiple transaction nodes in rapid succession
        for i in range(4):
            node = BehavioralNode(
                node_id=f"txn_node_{i}",
                action_type=ActionType.TRANSACTION_START,
                timestamp=datetime.now() + timedelta(seconds=i * 10),  # Rapid succession
                duration=100,
                sequence_position=i
            )
            fraud_graph.add_node(node)
        
        result = gnn_detector.detect_anomalies(fraud_graph, sample_user_profile)
        
        # Should detect elevated risk due to rapid transactions
        assert result.anomaly_score > 0.3
    
    def test_automation_detection(self, gnn_detector, sample_user_profile):
        """Test automation/bot detection"""
        # Create graph with perfect timing (bot-like behavior)
        automation_graph = SessionGraph(
            session_id="automation_session",
            user_id="test_user",
            start_time=datetime.now()
        )
        
        # Add nodes with identical timing and coordinates (automation signature)
        for i in range(6):
            node = BehavioralNode(
                node_id=f"auto_node_{i}",
                action_type=ActionType.TAP,
                timestamp=datetime.now() + timedelta(milliseconds=i * 500),  # Perfect timing
                duration=100,  # Identical duration
                coordinates=(100, 200),  # Identical coordinates
                sequence_position=i
            )
            automation_graph.add_node(node)
        
        # Add edges with identical timing
        for i in range(5):
            edge = BehavioralEdge(
                edge_id=f"auto_edge_{i}",
                source_node=f"auto_node_{i}",
                target_node=f"auto_node_{i+1}",
                transition_type=TransitionType.SEQUENTIAL,
                time_gap=500,  # Identical gaps
                confidence=1.0
            )
            automation_graph.add_edge(edge)
        
        result = gnn_detector.detect_anomalies(automation_graph, sample_user_profile)
        
        # Should detect automation patterns
        assert result.anomaly_score > 0.4
    
    def test_empty_graph_handling(self, gnn_detector, sample_user_profile):
        """Test handling of empty session graphs"""
        empty_graph = SessionGraph(
            session_id="empty_session",
            user_id="test_user",
            start_time=datetime.now()
        )
        
        result = gnn_detector.detect_anomalies(empty_graph, sample_user_profile)
        
        assert isinstance(result, GNNAnomalyResult)
        assert result.confidence < 0.5  # Low confidence for empty graphs


class TestPolicyOrchestrationEngine:
    """Test Layer J: Policy Orchestration Engine"""
    
    @pytest.fixture
    def mock_faiss_layer(self):
        mock = AsyncMock()
        mock.make_authentication_decision = AsyncMock(return_value=(
            AuthenticationDecision.ALLOW,
            RiskLevel.LOW,
            0.2,
            0.8,
            ["High FAISS similarity"]
        ))
        return mock
    
    @pytest.fixture
    def mock_gnn_detector(self):
        mock = Mock()
        mock.detect_anomalies = Mock(return_value=GNNAnomalyResult(
            anomaly_score=0.2,
            anomaly_types=[],
            confidence=0.8,
            risk_level=RiskLevel.LOW,
            decision=AuthenticationDecision.ALLOW,
            graph_embeddings=np.random.rand(32),
            explanation="Normal behavioral patterns detected"
        ))
        return mock
    
    @pytest.fixture
    def mock_adaptive_layer(self):
        mock = AsyncMock()
        mock.analyze_behavioral_pattern = AsyncMock(return_value=Mock(
            confidence=0.7,
            drift_detected=False,
            decision=AuthenticationDecision.ALLOW,
            __dict__={'confidence': 0.7, 'drift_detected': False, 'decision': AuthenticationDecision.ALLOW}
        ))
        return mock
    
    @pytest.fixture
    def policy_engine(self, mock_faiss_layer, mock_gnn_detector, mock_adaptive_layer):
        return PolicyOrchestrationEngine(
            faiss_layer=mock_faiss_layer,
            gnn_detector=mock_gnn_detector,
            adaptive_layer=mock_adaptive_layer
        )
    
    @pytest.fixture
    def sample_behavioral_vector(self):
        return BehavioralVector(
            user_id="test_user",
            session_id="test_session",
            vector=np.random.rand(90).tolist(),
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_session_graph(self):
        return SessionGraph(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now()
        )
    
    @pytest.fixture
    def sample_user_profile(self):
        return UserProfile(
            user_id="test_user",
            current_phase=SessionPhase.FULL_AUTH,
            drift_score=0.1,
            false_positive_rate=0.05
        )
    
    @pytest.fixture
    def low_risk_contextual_factors(self):
        return ContextualRiskFactors(
            transaction_amount=1000,
            is_new_beneficiary=False,
            time_of_day_risk=0.1,
            location_risk=0.1,
            device_risk=0.1,
            transaction_frequency_risk=0.1,
            recent_failures=0,
            vpn_detected=False
        )
    
    @pytest.fixture
    def high_risk_contextual_factors(self):
        return ContextualRiskFactors(
            transaction_amount=75000,
            is_new_beneficiary=True,
            time_of_day_risk=0.8,
            location_risk=0.7,
            device_risk=0.6,
            transaction_frequency_risk=0.8,
            recent_failures=4,
            vpn_detected=True
        )
    
    @pytest.mark.asyncio
    async def test_level_1_basic_policy(
        self, 
        policy_engine, 
        sample_behavioral_vector,
        sample_session_graph,
        sample_user_profile,
        low_risk_contextual_factors
    ):
        """Test Level 1 Basic Policy execution"""
        result = await policy_engine.make_policy_decision(
            user_id="test_user",
            session_id="test_session", 
            behavioral_vector=sample_behavioral_vector,
            session_graph=sample_session_graph,
            user_profile=sample_user_profile,
            contextual_factors=low_risk_contextual_factors,
            requested_policy_level=PolicyLevel.LEVEL_1_BASIC
        )
        
        assert isinstance(result, PolicyDecisionResult)
        assert result.policy_level_used == PolicyLevel.LEVEL_1_BASIC
        assert result.faiss_result is not None
        assert result.gnn_result is None  # Level 1 doesn't use GNN
        assert result.final_decision in [AuthenticationDecision.ALLOW, AuthenticationDecision.CHALLENGE, AuthenticationDecision.BLOCK]
    
    @pytest.mark.asyncio
    async def test_level_4_maximum_policy(
        self,
        policy_engine,
        sample_behavioral_vector,
        sample_session_graph,
        sample_user_profile,
        high_risk_contextual_factors
    ):
        """Test Level 4 Maximum Policy execution"""
        result = await policy_engine.make_policy_decision(
            user_id="test_user",
            session_id="test_session",
            behavioral_vector=sample_behavioral_vector,
            session_graph=sample_session_graph,
            user_profile=sample_user_profile,
            contextual_factors=high_risk_contextual_factors,
            requested_policy_level=PolicyLevel.LEVEL_4_MAXIMUM
        )
        
        assert result.policy_level_used == PolicyLevel.LEVEL_4_MAXIMUM
        assert result.faiss_result is not None
        assert result.gnn_result is not None
        assert result.adaptive_result is not None
        assert result.contextual_risk is not None
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_automatic_policy_level_selection(
        self,
        policy_engine,
        sample_behavioral_vector,
        sample_session_graph,
        sample_user_profile,
        high_risk_contextual_factors
    ):
        """Test automatic policy level selection based on risk"""
        result = await policy_engine.make_policy_decision(
            user_id="test_user",
            session_id="test_session",
            behavioral_vector=sample_behavioral_vector,
            session_graph=sample_session_graph,
            user_profile=sample_user_profile,
            contextual_factors=high_risk_contextual_factors
            # No requested_policy_level - should auto-select Level 4 due to high risk
        )
        
        # High risk factors should trigger Level 4 Maximum
        assert result.policy_level_used == PolicyLevel.LEVEL_4_MAXIMUM
    
    @pytest.mark.asyncio
    async def test_contextual_risk_calculation(
        self,
        policy_engine,
        sample_behavioral_vector,
        sample_session_graph,
        sample_user_profile,
        high_risk_contextual_factors
    ):
        """Test contextual risk factor integration"""
        result = await policy_engine.make_policy_decision(
            user_id="test_user",
            session_id="test_session",
            behavioral_vector=sample_behavioral_vector,
            session_graph=sample_session_graph,
            user_profile=sample_user_profile,
            contextual_factors=high_risk_contextual_factors,
            requested_policy_level=PolicyLevel.LEVEL_3_ADVANCED
        )
        
        # High contextual risk should be reflected in the result
        contextual_risk = high_risk_contextual_factors.get_overall_risk()
        assert contextual_risk > 0.5
        assert result.contextual_risk is not None
        assert result.final_risk_score > 0.4  # Should be elevated due to context
    
    @pytest.mark.asyncio
    async def test_fraud_signature_override(
        self,
        policy_engine,
        sample_behavioral_vector,
        sample_session_graph,
        sample_user_profile,
        low_risk_contextual_factors
    ):
        """Test fraud signature detection overrides normal decisions"""
        # Mock GNN to return fraud signature
        policy_engine.gnn_detector.detect_anomalies = Mock(return_value=GNNAnomalyResult(
            anomaly_score=0.9,
            anomaly_types=[AnomalyType.FRAUD_SIGNATURE],
            confidence=0.9,
            risk_level=RiskLevel.HIGH,
            decision=AuthenticationDecision.BLOCK,
            graph_embeddings=np.random.rand(32),
            explanation="Fraud signature detected"
        ))
        
        result = await policy_engine.make_policy_decision(
            user_id="test_user",
            session_id="test_session",
            behavioral_vector=sample_behavioral_vector,
            session_graph=sample_session_graph,
            user_profile=sample_user_profile,
            contextual_factors=low_risk_contextual_factors,
            requested_policy_level=PolicyLevel.LEVEL_3_ADVANCED
        )
        
        # Fraud signature should override and force BLOCK decision
        assert result.final_decision == AuthenticationDecision.BLOCK
        assert result.final_risk_level == RiskLevel.HIGH
    
    def test_policy_statistics_tracking(self, policy_engine):
        """Test policy engine statistics tracking"""
        stats = policy_engine.get_policy_statistics()
        
        assert 'performance_metrics' in stats
        assert 'policy_thresholds' in stats
        assert 'layer_weights' in stats
        assert 'recent_decisions' in stats
        
        # Check structure of performance metrics
        performance = stats['performance_metrics']
        assert 'total_decisions' in performance
        assert 'level_usage' in performance
        assert 'avg_processing_time' in performance
        assert 'decision_distribution' in performance
    
    def test_threshold_adjustment(self, policy_engine):
        """Test dynamic threshold adjustment"""
        original_threshold = policy_engine.policy_thresholds[PolicyLevel.LEVEL_1_BASIC]['allow_threshold']
        
        new_thresholds = {'allow_threshold': 0.9}
        policy_engine.adjust_policy_thresholds(PolicyLevel.LEVEL_1_BASIC, new_thresholds)
        
        updated_threshold = policy_engine.policy_thresholds[PolicyLevel.LEVEL_1_BASIC]['allow_threshold']
        assert updated_threshold == 0.9
        assert updated_threshold != original_threshold


class TestLayerIntegration:
    """Test integration between all layers"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test complete pipeline from graph generation to policy decision"""
        # Setup components
        graph_generator = SessionGraphGenerator()
        gnn_detector = GNNAnomalyDetector({'hidden_dim': 16, 'num_layers': 1})  # Small for testing
        
        # Mock other layers
        mock_faiss = AsyncMock()
        mock_faiss.make_authentication_decision = AsyncMock(return_value=(
            AuthenticationDecision.ALLOW, RiskLevel.LOW, 0.2, 0.8, ["Test"]
        ))
        
        mock_adaptive = AsyncMock()
        mock_adaptive.analyze_behavioral_pattern = AsyncMock(return_value=Mock(
            confidence=0.7, drift_detected=False, decision=AuthenticationDecision.ALLOW,
            __dict__={'confidence': 0.7, 'drift_detected': False, 'decision': AuthenticationDecision.ALLOW}
        ))
        
        policy_engine = PolicyOrchestrationEngine(
            faiss_layer=mock_faiss,
            gnn_detector=gnn_detector,
            adaptive_layer=mock_adaptive
        )
        
        # Create test data
        behavioral_events = [
            {
                'event_type': 'touch',
                'timestamp': 1700000000000,
                'x': 100,
                'y': 200,
                'pressure': 0.5,
                'duration': 150
            },
            {
                'event_type': 'scroll',
                'timestamp': 1700000000500,
                'velocity': 250,
                'direction': 'down',
                'duration': 200
            }
        ]
        
        # Execute full pipeline
        # 1. Generate session graph
        session_graph = graph_generator.generate_session_graph(
            "test_user", "test_session", behavioral_events
        )
        
        # 2. Create behavioral vector
        behavioral_vector = BehavioralVector(
            user_id="test_user",
            session_id="test_session",
            vector=np.random.rand(90).tolist(),
            timestamp=datetime.now()
        )
        
        # 3. User profile
        user_profile = UserProfile(
            user_id="test_user",
            current_phase=SessionPhase.FULL_AUTH,
            drift_score=0.1,
            false_positive_rate=0.05
        )
        
        # 4. Contextual factors
        contextual_factors = ContextualRiskFactors(
            transaction_amount=5000,
            is_new_beneficiary=False
        )
        
        # 5. Make policy decision
        result = await policy_engine.make_policy_decision(
            user_id="test_user",
            session_id="test_session",
            behavioral_vector=behavioral_vector,
            session_graph=session_graph,
            user_profile=user_profile,
            contextual_factors=contextual_factors
        )
        
        # Verify complete pipeline execution
        assert isinstance(result, PolicyDecisionResult)
        assert result.final_decision in [AuthenticationDecision.ALLOW, AuthenticationDecision.CHALLENGE, AuthenticationDecision.BLOCK]
        assert result.gnn_result is not None
        assert isinstance(result.gnn_result.graph_embeddings, np.ndarray)
        assert result.processing_time_ms > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
