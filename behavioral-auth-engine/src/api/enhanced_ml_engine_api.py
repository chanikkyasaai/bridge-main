"""
Integration with ML Engine API Service
Connects the new behavioral layers with the existing system
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from src.layers.session_graph_generator import SessionGraphGenerator
from src.layers.policy_orchestration_engine import (
    PolicyOrchestrationEngine, PolicyLevel, ContextualRiskFactors,
    PolicyDecisionResult
)
from src.data.models import BehavioralVector, UserProfile, AuthenticationDecision, RiskLevel

logger = logging.getLogger(__name__)

class EnhancedMLEngineAPI:
    """
    Enhanced ML Engine API with complete Layer G, H, J integration
    
    Integrates:
    - Layer G: Session Graph Generator
    - Layer H: GNN Anomaly Detection (optional)  
    - Layer J: Policy Orchestration Engine
    """
    
    def __init__(self, faiss_layer, adaptive_layer, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize core layers
        self.faiss_layer = faiss_layer
        self.adaptive_layer = adaptive_layer
        
        # Initialize new layers
        self.session_graph_generator = SessionGraphGenerator()
        
        # Initialize GNN detector if PyTorch available
        self.gnn_detector = None
        try:
            from src.layers.gnn_anomaly_detector import GNNAnomalyDetector
            gnn_config = self.config.get('gnn_config', {
                'node_features': 10,
                'edge_features': 4,
                'hidden_dim': 64,
                'num_layers': 3
            })
            self.gnn_detector = GNNAnomalyDetector(gnn_config)
            self.logger.info("GNN Anomaly Detector initialized")
        except ImportError:
            self.logger.warning("PyTorch not available - GNN detection disabled")
        
        # Initialize Policy Orchestration Engine
        self.policy_engine = PolicyOrchestrationEngine(
            faiss_layer=self.faiss_layer,
            gnn_detector=self.gnn_detector,
            adaptive_layer=self.adaptive_layer,
            config=self.config.get('policy_config', {})
        )
        
        self.logger.info("Enhanced ML Engine API initialized with all layers")
    
    async def analyze_enhanced_behavioral_data(
        self,
        user_id: str,
        session_id: str,
        behavioral_data: Dict[str, Any],
        user_profile: Optional[UserProfile] = None,
        contextual_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete behavioral analysis using all implemented layers
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            behavioral_data: Mobile behavioral data
            user_profile: User's behavioral profile
            contextual_factors: Transaction and environmental context
            
        Returns:
            Complete analysis result with policy decision
        """
        start_time = datetime.now()
        
        try:
            # Extract behavioral events for graph generation
            behavioral_events = behavioral_data.get('events', [])
            
            # Layer G: Generate session graph
            session_graph = self.session_graph_generator.generate_session_graph(
                user_id, session_id, behavioral_events
            )
            
            # Create behavioral vector (using existing enhanced processor)
            behavioral_vector = BehavioralVector(
                user_id=user_id,
                session_id=session_id,
                vector=behavioral_data.get('vector', np.random.rand(90).tolist()),
                timestamp=datetime.now()
            )
            
            # Create contextual risk factors
            contextual_risk = self._create_contextual_risk_factors(
                contextual_factors or {}
            )
            
            # Get or create user profile
            if not user_profile:
                user_profile = self._get_default_user_profile(user_id)
            
            # Layer J: Policy orchestration decision
            policy_result = await self.policy_engine.make_policy_decision(
                user_id=user_id,
                session_id=session_id,
                behavioral_vector=behavioral_vector,
                session_graph=session_graph,
                user_profile=user_profile,
                contextual_factors=contextual_risk
            )
            
            # Compile comprehensive result
            result = {
                'status': 'success',
                'decision': policy_result.final_decision.value,
                'risk_level': policy_result.final_risk_level.value,
                'risk_score': policy_result.final_risk_score,
                'confidence': policy_result.confidence,
                'policy_level_used': policy_result.policy_level_used.value,
                
                # Session graph metrics
                'session_graph': {
                    'node_count': len(session_graph.nodes),
                    'edge_count': len(session_graph.edges),
                    'action_types': list(set(node.action_type.value for node in session_graph.nodes.values())),
                    'transition_types': list(set(edge.transition_type.value for edge in session_graph.edges.values()))
                },
                
                # Layer-specific results
                'layer_results': {
                    'faiss': policy_result.faiss_result,
                    'gnn': policy_result.gnn_result.__dict__ if policy_result.gnn_result else None,
                    'adaptive': policy_result.adaptive_result,
                    'contextual': policy_result.contextual_risk.__dict__ if policy_result.contextual_risk else None
                },
                
                # Decision explanation
                'explanation': policy_result.explanation,
                'primary_reasons': [reason.value for reason in policy_result.primary_reasons],
                'risk_factors': policy_result.risk_factors,
                'recommendations': policy_result.recommendations,
                
                # Performance metrics
                'processing_time_ms': policy_result.processing_time_ms,
                'layer_timings': policy_result.layer_timings,
                
                # Metadata
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'user_id': user_id
            }
            
            self.logger.info(
                f"Enhanced analysis complete for {user_id}: {policy_result.final_decision.value} "
                f"(risk: {policy_result.final_risk_score:.3f}, level: {policy_result.policy_level_used.value})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced behavioral analysis: {e}")
            return self._create_fallback_response(str(e))
    
    def _create_contextual_risk_factors(self, context: Dict[str, Any]) -> ContextualRiskFactors:
        """Create contextual risk factors from request context"""
        return ContextualRiskFactors(
            transaction_amount=float(context.get('transaction_amount', 0)),
            is_new_beneficiary=context.get('is_new_beneficiary', False),
            time_of_day_risk=float(context.get('time_of_day_risk', 0.1)),
            location_risk=float(context.get('location_risk', 0.1)),
            device_risk=float(context.get('device_risk', 0.1)),
            transaction_frequency_risk=float(context.get('transaction_frequency_risk', 0.1)),
            recent_failures=int(context.get('recent_failures', 0)),
            vpn_detected=context.get('vpn_detected', False)
        )
    
    def _get_default_user_profile(self, user_id: str) -> UserProfile:
        """Create default user profile for testing"""
        from src.data.models import SessionPhase
        return UserProfile(
            user_id=user_id,
            current_phase=SessionPhase.FULL_AUTH,
            drift_score=0.1,
            false_positive_rate=0.05
        )
    
    def _create_fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Create safe fallback response for errors"""
        return {
            'status': 'error',
            'decision': 'challenge',
            'risk_level': 'medium',
            'risk_score': 0.7,
            'confidence': 0.3,
            'explanation': f"Fallback decision due to error: {error_message}",
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'layers_status': {
                'session_graph_generator': 'active',
                'gnn_anomaly_detector': 'active' if self.gnn_detector else 'disabled',
                'policy_orchestration': 'active',
                'faiss_layer': 'active',
                'adaptive_layer': 'active'
            },
            'policy_statistics': self.policy_engine.get_policy_statistics(),
            'layer_statistics': {
                'faiss': await self.faiss_layer.get_layer_statistics() if hasattr(self.faiss_layer, 'get_layer_statistics') else {},
                'gnn': self.gnn_detector.get_layer_statistics() if self.gnn_detector else {},
                'policy': self.policy_engine.get_policy_statistics()
            },
            'system_info': {
                'gnn_available': self.gnn_detector is not None,
                'total_policy_levels': len(PolicyLevel),
                'supported_action_types': len([action for action in self.session_graph_generator.__class__.__dict__ if 'ACTION' in str(action)])
            }
        }
    
    async def analyze_session_graph_only(
        self,
        user_id: str,
        session_id: str,
        behavioral_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze only session graph for testing/debugging"""
        try:
            session_graph = self.session_graph_generator.generate_session_graph(
                user_id, session_id, behavioral_events
            )
            
            # Generate graph features
            feature_vector = self.session_graph_generator.get_graph_features_vector(session_graph)
            gnn_export = self.session_graph_generator.export_graph_for_gnn(session_graph)
            
            return {
                'status': 'success',
                'session_graph': {
                    'user_id': session_graph.user_id,
                    'session_id': session_graph.session_id,
                    'node_count': len(session_graph.nodes),
                    'edge_count': len(session_graph.edges),
                    'nodes': {
                        node_id: {
                            'action_type': node.action_type.value,
                            'timestamp': node.timestamp.isoformat(),
                            'duration': node.duration,
                            'coordinates': node.coordinates,
                            'sequence_position': node.sequence_position
                        }
                        for node_id, node in session_graph.nodes.items()
                    },
                    'edges': {
                        edge_id: {
                            'source': edge.source_node,
                            'target': edge.target_node,
                            'transition_type': edge.transition_type.value,
                            'time_gap': edge.time_gap,
                            'confidence': edge.confidence
                        }
                        for edge_id, edge in session_graph.edges.items()
                    }
                },
                'graph_features': {
                    'feature_vector_size': len(feature_vector),
                    'feature_vector': feature_vector.tolist(),
                    'gnn_export': {
                        'num_nodes': gnn_export['num_nodes'],
                        'num_edges': gnn_export['num_edges'],
                        'node_features_shape': list(gnn_export['node_features'].shape),
                        'edge_features_shape': list(gnn_export['edge_attributes'].shape)
                    }
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'session_graph': None
            }
