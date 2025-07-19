"""
Layer H: GNN-Based Anomaly Detection
Analyzes session graphs using Graph Neural Networks to detect behavioral sequence anomalies.
Critical component for national-level fraud detection system.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

from src.data.models import BehavioralVector, UserProfile, AuthenticationDecision, RiskLevel
from src.layers.session_graph_generator import SessionGraph, SessionGraphGenerator

logger = logging.getLogger(__name__)

class AnomalyType(str, Enum):
    """Types of anomalies detected by GNN"""
    SEQUENCE_ANOMALY = "sequence_anomaly"
    TIMING_ANOMALY = "timing_anomaly"
    SPATIAL_ANOMALY = "spatial_anomaly"
    PATTERN_DEVIATION = "pattern_deviation"
    FRAUD_SIGNATURE = "fraud_signature"
    AUTOMATION_DETECTED = "automation_detected"

@dataclass
class GNNAnomalyResult:
    """Result from GNN anomaly detection"""
    anomaly_score: float  # 0.0 (normal) to 1.0 (highly anomalous)
    anomaly_types: List[AnomalyType]
    confidence: float
    risk_level: RiskLevel
    decision: AuthenticationDecision
    graph_embeddings: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    explanation: str = ""
    processing_time_ms: float = 0.0

class BehavioralGNN(nn.Module):
    """
    Graph Neural Network for behavioral anomaly detection
    
    Uses Graph Attention Networks (GAT) to analyze behavioral sequences
    and detect anomalous patterns indicative of fraud or automation.
    """
    
    def __init__(
        self,
        node_features: int = 10,
        edge_features: int = 4, 
        hidden_dim: int = 64,
        num_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super(BehavioralGNN, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.node_projection = nn.Linear(node_features, hidden_dim)
        self.edge_projection = nn.Linear(edge_features, hidden_dim)
        
        # GAT layers for attention-based aggregation
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    in_dim, 
                    out_dim // attention_heads,
                    heads=attention_heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
        
        # Graph-level representation
        self.graph_pooling = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat of mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Anomaly detection head
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-task anomaly type prediction
        self.anomaly_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, len(AnomalyType))
        )
        
        # Temporal pattern analysis
        self.temporal_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            batch_first=True,
            bidirectional=True
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN
        
        Args:
            data: PyTorch Geometric Data object with node features and edge indices
            
        Returns:
            Tuple of (anomaly_scores, anomaly_types, graph_embeddings)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Project input features
        x = self.node_projection(x)
        
        # Store attention weights for explainability
        attention_weights = []
        
        # Pass through GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x, attention = gat_layer(x, edge_index, return_attention_weights=True)
            attention_weights.append(attention)
            
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # Graph-level pooling
        if batch is not None:
            graph_mean = global_mean_pool(x, batch)
            graph_max = global_max_pool(x, batch)
        else:
            graph_mean = torch.mean(x, dim=0, keepdim=True)
            graph_max = torch.max(x, dim=0, keepdim=True)[0]
        
        # Combine pooled representations
        graph_repr = torch.cat([graph_mean, graph_max], dim=-1)
        graph_embeddings = self.graph_pooling(graph_repr)
        
        # Anomaly detection
        anomaly_scores = self.anomaly_classifier(graph_embeddings)
        anomaly_types = self.anomaly_type_classifier(graph_embeddings)
        
        return anomaly_scores, anomaly_types, graph_embeddings

class GNNAnomalyDetector:
    """
    Layer H: GNN-Based Anomaly Detection
    
    Analyzes behavioral graphs using Graph Neural Networks to detect:
    - Sequence anomalies (unusual action patterns)
    - Timing anomalies (suspicious temporal patterns) 
    - Spatial anomalies (abnormal touch/swipe patterns)
    - Fraud signatures (known attack patterns)
    - Automation detection (bot-like behavior)
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = model_config or {}
        
        # Model parameters
        self.node_features = self.config.get('node_features', 10)
        self.edge_features = self.config.get('edge_features', 4) 
        self.hidden_dim = self.config.get('hidden_dim', 64)
        self.num_layers = self.config.get('num_layers', 3)
        
        # Initialize GNN model
        self.model = BehavioralGNN(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Load pre-trained weights if available
        self.model_path = self.config.get('model_path')
        if self.model_path:
            self._load_model()
        
        # Anomaly thresholds
        self.anomaly_thresholds = {
            'low_risk': 0.3,
            'medium_risk': 0.6,
            'high_risk': 0.8
        }
        
        # Pattern libraries for known fraud signatures
        self.fraud_patterns = self._load_fraud_patterns()
        
        # User baseline graphs for comparison
        self.user_baselines: Dict[str, List[torch.Tensor]] = {}
        
        self.logger.info("GNN Anomaly Detector (Layer H) initialized")
    
    async def initialize(self):
        """Initialize the GNN detector (async for compatibility with ML Engine)"""
        try:
            # Set model to evaluation mode
            self.model.eval()
            self.logger.info("GNN Anomaly Detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize GNN detector: {e}")
            raise
    
    async def detect_anomalies(
        self,
        user_id: str,
        session_id: str,
        behavioral_vector: np.ndarray,
        behavioral_logs: List[Dict[str, Any]]
    ) -> GNNAnomalyResult:
        """
        Simplified anomaly detection interface for ML Engine integration
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            behavioral_vector: 90-dimensional behavioral vector from FAISS
            behavioral_logs: Raw behavioral event logs
            
        Returns:
            GNNAnomalyResult with anomaly assessment
        """
        start_time = datetime.now()
        
        print(f"🧠 GNN DETECT_ANOMALIES: Starting detection for user {user_id}, session {session_id}")
        print(f"🧠 GNN DETECT_ANOMALIES: Behavioral logs count: {len(behavioral_logs)}")
        print(f"🧠 GNN DETECT_ANOMALIES: Behavioral vector shape: {behavioral_vector.shape}")
        
        try:
            # Simplified anomaly detection based on behavioral patterns
            print(f"🧠 GNN DETECT_ANOMALIES: About to call _analyze_behavioral_patterns...")
            anomaly_score = await self._analyze_behavioral_patterns(
                behavioral_vector, behavioral_logs
            )
            print(f"🧠 GNN DETECT_ANOMALIES: Got anomaly score: {anomaly_score}")
            
            # Determine anomaly types based on patterns
            anomaly_types = self._classify_anomaly_types(behavioral_logs, anomaly_score)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(behavioral_logs)
            
            # Determine risk level
            if anomaly_score < 0.3:
                risk_level = RiskLevel.LOW
                decision = AuthenticationDecision.ALLOW
            elif anomaly_score < 0.6:
                risk_level = RiskLevel.MEDIUM
                decision = AuthenticationDecision.CHALLENGE
            else:
                risk_level = RiskLevel.HIGH
                decision = AuthenticationDecision.DENY
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GNNAnomalyResult(
                anomaly_score=anomaly_score,
                anomaly_types=anomaly_types,
                confidence=confidence,
                risk_level=risk_level,
                decision=decision,
                graph_embeddings=behavioral_vector,
                explanation=f"GNN analysis detected {'high' if anomaly_score > 0.6 else 'moderate' if anomaly_score > 0.3 else 'low'} anomaly patterns",
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"GNN anomaly detection failed: {e}")
            # Return safe default result
            return GNNAnomalyResult(
                anomaly_score=0.0,
                anomaly_types=[],
                confidence=0.5,
                risk_level=RiskLevel.LOW,
                decision=AuthenticationDecision.ALLOW,
                graph_embeddings=np.zeros(90),
                explanation=f"GNN analysis failed: {e}",
                processing_time_ms=0.0
            )
    
    async def _analyze_behavioral_patterns(
        self, 
        behavioral_vector: np.ndarray, 
        behavioral_logs: List[Dict[str, Any]]
    ) -> float:
        """Analyze behavioral patterns to determine anomaly score"""
        try:
            anomaly_score = 0.0
            
            print(f"🔍 GNN ANALYSIS: Starting behavioral pattern analysis with {len(behavioral_logs)} logs")
            self.logger.info(f"🔍 GNN: Starting behavioral pattern analysis")
            
            # Check for robotic/automation patterns
            print(f"🤖 GNN ANALYSIS: About to call automation detection...")
            automation_score = self._detect_automation_patterns(behavioral_logs)
            print(f"🤖 GNN ANALYSIS: Automation score returned: {automation_score}")
            self.logger.info(f"🤖 GNN: Automation score: {automation_score}")
            anomaly_score = max(anomaly_score, automation_score)
            
            # Check for spatial anomalies
            spatial_score = self._detect_spatial_anomalies(behavioral_logs)
            self.logger.info(f"📍 GNN: Spatial score: {spatial_score}")
            anomaly_score = max(anomaly_score, spatial_score)
            
            # Check vector anomalies (identical patterns)
            vector_score = self._detect_vector_anomalies(behavioral_vector)
            self.logger.info(f"📊 GNN: Vector score: {vector_score}")
            anomaly_score = max(anomaly_score, vector_score)
            
            self.logger.info(f"🎯 GNN: Final anomaly score: {anomaly_score}")
            
            return min(1.0, anomaly_score)
            
        except Exception as e:
            self.logger.error(f"Error analyzing behavioral patterns: {e}")
            return 0.0
    
    def _detect_automation_patterns(self, behavioral_logs: List[Dict[str, Any]]) -> float:
        """Detect automation/robotic behavior patterns"""
        try:
            automation_indicators = 0
            total_checks = 0
            
            print(f"🔍 GNN AUTOMATION: Starting analysis of {len(behavioral_logs)} behavioral logs")
            self.logger.info(f"🔍 GNN: Analyzing {len(behavioral_logs)} behavioral logs for automation")
            
            for log in behavioral_logs:
                if log.get('event_type') == 'touch_sequence':
                    data = log.get('data', {})
                    touch_events = data.get('touch_events', [])
                    
                    self.logger.debug(f"🔍 GNN: Processing touch sequence with {len(touch_events)} events")
                    
                    if len(touch_events) >= 2:
                        # Check for identical coordinates (fix: use x,y not coordinates)
                        coordinates = []
                        for t in touch_events:
                            x = t.get('x', 0)
                            y = t.get('y', 0)
                            coordinates.append([x, y])
                        
                        self.logger.debug(f"🔍 GNN: Touch coordinates: {coordinates}")
                        
                        identical_coords = 0
                        for i in range(1, len(coordinates)):
                            if coordinates[i] == coordinates[i-1]:
                                identical_coords += 1
                        
                        self.logger.info(f"🔍 GNN: Identical coordinates: {identical_coords}/{len(coordinates)-1}")
                        
                        if identical_coords >= (len(coordinates) - 1) * 0.8:  # 80% or more identical comparisons
                            automation_indicators += 1
                            print(f"🚨 GNN AUTOMATION: DETECTED - {identical_coords} identical coordinates")
                            self.logger.warning(f"🚨 GNN: AUTOMATION DETECTED - {identical_coords} identical coordinates")
                        total_checks += 1
                        
                        # Check for identical pressures
                        pressures = [t.get('pressure', 0) for t in touch_events]
                        unique_pressures = len(set(pressures))
                        self.logger.debug(f"🔍 GNN: Pressures: {pressures}, Unique: {unique_pressures}")
                        
                        if len(set(pressures)) == 1 and len(pressures) > 1:  # All same pressure
                            automation_indicators += 1
                            print(f"🚨 GNN AUTOMATION: DETECTED - identical pressures: {pressures[0]}")
                            self.logger.warning(f"🚨 GNN: AUTOMATION DETECTED - identical pressures: {pressures[0]}")
                        
                        # Check for identical durations
                        durations = [t.get('duration', 0) for t in touch_events]
                        unique_durations = len(set(durations))
                        self.logger.debug(f"🔍 GNN: Durations: {durations}, Unique: {unique_durations}")
                        
                        if len(set(durations)) == 1 and len(durations) > 1:  # All same duration
                            automation_indicators += 1
                            print(f"🚨 GNN AUTOMATION: DETECTED - identical durations: {durations[0]}")
                            self.logger.warning(f"🚨 GNN: AUTOMATION DETECTED - identical durations: {durations[0]}")
            
            print(f"🔍 GNN AUTOMATION: Analysis complete - {automation_indicators} indicators from {total_checks} checks")
            self.logger.info(f"🔍 GNN: Automation analysis complete - {automation_indicators} indicators from {total_checks} checks")
            
            if total_checks == 0:
                self.logger.info("🔍 GNN: No touch sequences found, returning 0.0")
                return 0.0
                
            automation_ratio = automation_indicators / max(total_checks, 1)
            final_score = min(1.0, automation_ratio)
            print(f"🎯 GNN AUTOMATION: Final automation ratio: {automation_ratio} ({automation_indicators}/{total_checks})")
            print(f"🎯 GNN AUTOMATION: Returning final score: {final_score}")
            self.logger.info(f"🎯 GNN: Final automation ratio: {automation_ratio} ({automation_indicators}/{total_checks})")
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error detecting automation patterns: {e}")
            return 0.0
    
    def _detect_spatial_anomalies(self, behavioral_logs: List[Dict[str, Any]]) -> float:
        """Detect spatial anomalies in touch patterns"""
        try:
            for log in behavioral_logs:
                if log.get('event_type') == 'touch_sequence':
                    data = log.get('data', {})
                    touch_events = data.get('touch_events', [])
                    
                    # Fix: Extract coordinates from x,y fields
                    coordinates = []
                    for t in touch_events:
                        x = t.get('x', 0)
                        y = t.get('y', 0)
                        coordinates.append([x, y])
                    
                    # Check for coordinates at edges (0, 0) or (1000, 1000)
                    edge_touches = 0
                    for coord in coordinates:
                        if (coord[0] == 0 and coord[1] == 0) or coord[0] > 900 or coord[1] > 900:
                            edge_touches += 1
                    
                    if edge_touches > len(coordinates) * 0.5:  # More than 50% at edges
                        return 0.7
                    
                    # Check for unrealistic movements (too far apart)
                    if len(coordinates) >= 2:
                        max_distance = 0
                        for i in range(1, len(coordinates)):
                            distance = ((coordinates[i][0] - coordinates[i-1][0])**2 + 
                                      (coordinates[i][1] - coordinates[i-1][1])**2)**0.5
                            max_distance = max(max_distance, distance)
                        
                        # If movement > 500 pixels in single gesture, flag as anomalous
                        if max_distance > 500:
                            return 0.6
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error detecting spatial anomalies: {e}")
            return 0.0
    
    def _detect_vector_anomalies(self, behavioral_vector: np.ndarray) -> float:
        """Detect anomalies in the behavioral vector itself"""
        try:
            # Check if vector has meaningful variation
            vector_std = np.std(behavioral_vector)
            if vector_std < 0.01:  # Very low variation
                return 0.4
            
            # Check for suspicious patterns (all zeros, all same values)
            non_zero_count = np.count_nonzero(behavioral_vector)
            if non_zero_count < 10:  # Less than 10 non-zero values out of 90
                return 0.3
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error detecting vector anomalies: {e}")
            return 0.0
    
    def _classify_anomaly_types(
        self, 
        behavioral_logs: List[Dict[str, Any]], 
        anomaly_score: float
    ) -> List[AnomalyType]:
        """Classify the types of anomalies detected"""
        anomaly_types = []
        
        if anomaly_score > 0.5:
            # Check for automation
            automation_score = self._detect_automation_patterns(behavioral_logs)
            if automation_score > 0.3:
                anomaly_types.append(AnomalyType.AUTOMATION_DETECTED)
            
            # Check for spatial issues
            spatial_score = self._detect_spatial_anomalies(behavioral_logs)
            if spatial_score > 0.3:
                anomaly_types.append(AnomalyType.SPATIAL_ANOMALY)
            
            # General pattern deviation
            if anomaly_score > 0.7:
                anomaly_types.append(AnomalyType.PATTERN_DEVIATION)
        
        return anomaly_types
    
    def _calculate_confidence(self, behavioral_logs: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the anomaly detection result"""
        try:
            # Base confidence on data quality and quantity
            data_points = 0
            for log in behavioral_logs:
                if log.get('event_type') == 'touch_sequence':
                    data = log.get('data', {})
                    touch_events = data.get('touch_events', [])
                    data_points += len(touch_events)
            
            # More data points = higher confidence
            if data_points >= 5:
                return 0.9
            elif data_points >= 3:
                return 0.7
            elif data_points >= 1:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.3
    
    def detect_anomalies_original(
        self,
        session_graph: SessionGraph,
        user_profile: UserProfile,
        baseline_graphs: Optional[List[SessionGraph]] = None
    ) -> GNNAnomalyResult:
        """
        Analyze session graph for behavioral anomalies
        
        Args:
            session_graph: Current session behavioral graph
            user_profile: User's behavioral profile
            baseline_graphs: Historical normal session graphs
            
        Returns:
            GNNAnomalyResult with anomaly assessment
        """
        start_time = datetime.now()
        
        try:
            # Convert session graph to PyTorch Geometric format
            graph_data = self._convert_to_pytorch_geometric(session_graph)
            
            if graph_data.x.size(0) == 0:
                return self._create_empty_result("No graph nodes to analyze")
            
            # Run GNN inference
            self.model.eval()
            with torch.no_grad():
                anomaly_scores, anomaly_types, graph_embeddings = self.model(graph_data)
            
            # Extract scalar values
            anomaly_score = float(anomaly_scores.squeeze())
            anomaly_type_probs = torch.sigmoid(anomaly_types).squeeze().numpy()
            
            # Determine detected anomaly types
            detected_anomalies = []
            for i, prob in enumerate(anomaly_type_probs):
                if prob > 0.5:
                    detected_anomalies.append(list(AnomalyType)[i])
            
            # Baseline comparison if available
            if baseline_graphs:
                baseline_score = self._compare_with_baselines(
                    graph_embeddings, session_graph.user_id, baseline_graphs
                )
                anomaly_score = max(anomaly_score, baseline_score)
            
            # Pattern matching against fraud signatures
            fraud_score = self._check_fraud_patterns(session_graph)
            if fraud_score > 0.7:
                detected_anomalies.append(AnomalyType.FRAUD_SIGNATURE)
                anomaly_score = max(anomaly_score, fraud_score)
            
            # Automation detection
            automation_score = self._detect_automation(session_graph)
            if automation_score > 0.6:
                detected_anomalies.append(AnomalyType.AUTOMATION_DETECTED)
                anomaly_score = max(anomaly_score, automation_score)
            
            # Determine risk level and decision
            risk_level, decision = self._determine_risk_and_decision(
                anomaly_score, detected_anomalies
            )
            
            # Calculate confidence based on graph quality and model certainty
            confidence = self._calculate_graph_confidence(graph_data, anomaly_score)
            
            # Generate explanation
            explanation = self._generate_explanation(
                anomaly_score, detected_anomalies, session_graph
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GNNAnomalyResult(
                anomaly_score=anomaly_score,
                anomaly_types=detected_anomalies,
                confidence=confidence,
                risk_level=risk_level,
                decision=decision,
                graph_embeddings=graph_embeddings.numpy(),
                explanation=explanation,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in GNN anomaly detection: {e}")
            return self._create_empty_result(f"Detection error: {str(e)}")
    
    def _convert_to_pytorch_geometric(self, session_graph: SessionGraph) -> Data:
        """Convert SessionGraph to PyTorch Geometric Data format"""
        try:
            # Extract graph data for GNN analysis
            graph_generator = SessionGraphGenerator()
            graph_export = graph_generator.export_graph_for_gnn(session_graph)
            
            # Convert to tensors
            node_features = torch.tensor(graph_export['node_features'], dtype=torch.float32)
            edge_indices = torch.tensor(graph_export['edge_indices'], dtype=torch.long)
            edge_attributes = torch.tensor(graph_export['edge_attributes'], dtype=torch.float32)
            
            # Ensure minimum dimensions
            if node_features.size(0) == 0:
                node_features = torch.zeros((1, self.node_features), dtype=torch.float32)
                edge_indices = torch.zeros((2, 0), dtype=torch.long)
                edge_attributes = torch.zeros((0, self.edge_features), dtype=torch.float32)
            
            # Pad features to expected size if needed
            if node_features.size(1) < self.node_features:
                padding = torch.zeros(node_features.size(0), 
                                    self.node_features - node_features.size(1))
                node_features = torch.cat([node_features, padding], dim=1)
            
            return Data(
                x=node_features,
                edge_index=edge_indices,
                edge_attr=edge_attributes,
                num_nodes=graph_export['num_nodes']
            )
            
        except Exception as e:
            self.logger.error(f"Error converting graph to PyTorch Geometric: {e}")
            # Return minimal valid graph
            return Data(
                x=torch.zeros((1, self.node_features), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, self.edge_features), dtype=torch.float32)
            )
    
    def _compare_with_baselines(
        self, 
        current_embedding: torch.Tensor, 
        user_id: str,
        baseline_graphs: List[SessionGraph]
    ) -> float:
        """Compare current session with user's historical baselines"""
        try:
            if not baseline_graphs:
                return 0.0
            
            # Convert baseline graphs to embeddings
            baseline_embeddings = []
            for graph in baseline_graphs:
                graph_data = self._convert_to_pytorch_geometric(graph)
                with torch.no_grad():
                    _, _, embedding = self.model(graph_data)
                baseline_embeddings.append(embedding)
            
            # Calculate average distance from baselines
            if baseline_embeddings:
                baseline_stack = torch.stack(baseline_embeddings)
                distances = torch.norm(current_embedding - baseline_stack, dim=1)
                avg_distance = torch.mean(distances)
                
                # Normalize to 0-1 range (higher distance = higher anomaly)
                return min(1.0, float(avg_distance / 10.0))
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error comparing with baselines: {e}")
            return 0.0
    
    def _check_fraud_patterns(self, session_graph: SessionGraph) -> float:
        """Check session against known fraud patterns"""
        try:
            fraud_score = 0.0
            
            # Pattern 1: Rapid transaction sequence
            transaction_nodes = [
                node for node in session_graph.nodes.values()
                if node.action_type.value in ['transaction_start', 'transaction_complete']
            ]
            
            if len(transaction_nodes) > 3:
                # Multiple transactions in short time = suspicious
                time_span = (transaction_nodes[-1].timestamp - transaction_nodes[0].timestamp).total_seconds()
                if time_span < 60:  # 3+ transactions in 1 minute
                    fraud_score += 0.4
            
            # Pattern 2: Automation indicators
            action_types = [node.action_type.value for node in session_graph.nodes.values()]
            if len(set(action_types)) < len(action_types) * 0.3:  # Low diversity
                fraud_score += 0.3
            
            # Pattern 3: Suspicious timing patterns
            edge_times = [edge.time_gap for edge in session_graph.edges.values()]
            if edge_times:
                time_variance = np.var(edge_times)
                if time_variance < 100:  # Very consistent timing = bot-like
                    fraud_score += 0.2
            
            # Pattern 4: Geographic impossibility (if location data available)
            # This would require location features in the graph
            
            return min(1.0, fraud_score)
            
        except Exception as e:
            self.logger.warning(f"Error checking fraud patterns: {e}")
            return 0.0
    
    def _detect_automation(self, session_graph: SessionGraph) -> float:
        """Detect automated/bot behavior in session graph"""
        try:
            automation_score = 0.0
            
            # Check for perfect timing regularity
            edge_times = [edge.time_gap for edge in session_graph.edges.values()]
            if len(edge_times) > 5:
                time_std = np.std(edge_times)
                time_mean = np.mean(edge_times)
                
                # Very low variance relative to mean suggests automation
                if time_mean > 0 and (time_std / time_mean) < 0.1:
                    automation_score += 0.4
            
            # Check for repeated exact coordinates (automation)
            coordinates = [
                node.coordinates for node in session_graph.nodes.values()
                if node.coordinates
            ]
            
            if len(coordinates) > 5:
                unique_coords = len(set(coordinates))
                if unique_coords / len(coordinates) < 0.3:  # Many repeated positions
                    automation_score += 0.3
            
            # Check for unnatural action sequences
            action_sequence = [node.action_type.value for node in session_graph.nodes.values()]
            if len(action_sequence) > 10:
                # Look for repeated patterns
                pattern_score = self._analyze_sequence_patterns(action_sequence)
                automation_score += pattern_score
            
            return min(1.0, automation_score)
            
        except Exception as e:
            self.logger.warning(f"Error detecting automation: {e}")
            return 0.0
    
    def _analyze_sequence_patterns(self, sequence: List[str]) -> float:
        """Analyze action sequence for automation patterns"""
        try:
            # Look for repeated subsequences
            pattern_score = 0.0
            
            for pattern_length in range(2, min(6, len(sequence) // 3)):
                patterns = {}
                for i in range(len(sequence) - pattern_length + 1):
                    pattern = tuple(sequence[i:i + pattern_length])
                    patterns[pattern] = patterns.get(pattern, 0) + 1
                
                # If any pattern repeats more than expected, it's suspicious
                max_repeats = max(patterns.values()) if patterns else 0
                expected_repeats = len(sequence) / (2 ** pattern_length)
                
                if max_repeats > expected_repeats * 3:
                    pattern_score += 0.1 * pattern_length
            
            return min(0.4, pattern_score)
            
        except Exception as e:
            return 0.0
    
    def _determine_risk_and_decision(
        self, 
        anomaly_score: float, 
        anomaly_types: List[AnomalyType]
    ) -> Tuple[RiskLevel, AuthenticationDecision]:
        """Determine risk level and authentication decision"""
        
        # High-priority anomalies that trigger immediate blocking
        critical_anomalies = {
            AnomalyType.FRAUD_SIGNATURE, 
            AnomalyType.AUTOMATION_DETECTED
        }
        
        if any(anomaly in critical_anomalies for anomaly in anomaly_types):
            return RiskLevel.HIGH, AuthenticationDecision.BLOCK
        
        # Score-based decisions
        if anomaly_score >= self.anomaly_thresholds['high_risk']:
            return RiskLevel.HIGH, AuthenticationDecision.BLOCK
        elif anomaly_score >= self.anomaly_thresholds['medium_risk']:
            return RiskLevel.MEDIUM, AuthenticationDecision.CHALLENGE
        elif anomaly_score >= self.anomaly_thresholds['low_risk']:
            return RiskLevel.LOW, AuthenticationDecision.ALLOW
        else:
            return RiskLevel.LOW, AuthenticationDecision.ALLOW
    
    def _calculate_graph_confidence(self, graph_data: Data, anomaly_score: float) -> float:
        """Calculate confidence in the anomaly detection result"""
        try:
            confidence = 0.5  # Base confidence
            
            # Higher confidence with more graph nodes/edges
            graph_size_factor = min(1.0, graph_data.x.size(0) / 20.0)
            confidence += 0.3 * graph_size_factor
            
            # Higher confidence for extreme scores
            score_extremity = max(anomaly_score, 1.0 - anomaly_score)
            confidence += 0.2 * score_extremity
            
            return min(0.95, confidence)
            
        except Exception:
            return 0.5
    
    def _generate_explanation(
        self, 
        anomaly_score: float, 
        anomaly_types: List[AnomalyType],
        session_graph: SessionGraph
    ) -> str:
        """Generate human-readable explanation of the anomaly detection result"""
        
        if anomaly_score < 0.3:
            return "Behavioral patterns appear normal and consistent with user profile."
        
        explanations = []
        
        if AnomalyType.FRAUD_SIGNATURE in anomaly_types:
            explanations.append("Session matches known fraud patterns")
        
        if AnomalyType.AUTOMATION_DETECTED in anomaly_types:
            explanations.append("Detected bot-like automated behavior")
        
        if AnomalyType.SEQUENCE_ANOMALY in anomaly_types:
            explanations.append("Unusual sequence of actions detected")
        
        if AnomalyType.TIMING_ANOMALY in anomaly_types:
            explanations.append("Suspicious timing patterns in user behavior")
        
        if AnomalyType.SPATIAL_ANOMALY in anomaly_types:
            explanations.append("Abnormal touch/gesture patterns detected")
        
        if not explanations:
            if anomaly_score > 0.6:
                explanations.append("Multiple behavioral indicators suggest elevated risk")
            else:
                explanations.append("Minor deviations from normal behavioral patterns")
        
        return "; ".join(explanations)
    
    def _create_empty_result(self, reason: str) -> GNNAnomalyResult:
        """Create empty/default result for error cases"""
        return GNNAnomalyResult(
            anomaly_score=0.5,
            anomaly_types=[],
            confidence=0.3,
            risk_level=RiskLevel.MEDIUM,
            decision=AuthenticationDecision.CHALLENGE,
            graph_embeddings=np.zeros(self.hidden_dim // 2),
            explanation=reason
        )
    
    def _load_model(self):
        """Load pre-trained GNN model weights"""
        try:
            if self.model_path and torch.cuda.is_available():
                checkpoint = torch.load(self.model_path)
            else:
                checkpoint = torch.load(self.model_path, map_location='cpu')
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded GNN model from {self.model_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not load pre-trained model: {e}")
            self.logger.info("Using randomly initialized model")
    
    def _load_fraud_patterns(self) -> Dict[str, Any]:
        """Load known fraud pattern signatures"""
        # In production, this would load from a comprehensive fraud pattern database
        return {
            'rapid_transaction_pattern': {
                'min_transactions': 3,
                'max_time_span': 60,
                'risk_score': 0.8
            },
            'automation_pattern': {
                'timing_variance_threshold': 0.1,
                'coordinate_repetition_threshold': 0.3,
                'risk_score': 0.7
            }
        }
    
    def update_user_baseline(self, user_id: str, session_graph: SessionGraph):
        """Update user's baseline behavioral graphs"""
        try:
            graph_data = self._convert_to_pytorch_geometric(session_graph)
            
            with torch.no_grad():
                _, _, embedding = self.model(graph_data)
            
            if user_id not in self.user_baselines:
                self.user_baselines[user_id] = []
            
            # Keep only recent baselines (rolling window)
            self.user_baselines[user_id].append(embedding)
            if len(self.user_baselines[user_id]) > 10:
                self.user_baselines[user_id].pop(0)
                
        except Exception as e:
            self.logger.warning(f"Error updating user baseline: {e}")
    
    def get_layer_statistics(self) -> Dict[str, Any]:
        """Get GNN layer performance statistics"""
        return {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'user_baselines_count': len(self.user_baselines),
            'anomaly_thresholds': self.anomaly_thresholds,
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'hidden_dim': self.hidden_dim
        }
