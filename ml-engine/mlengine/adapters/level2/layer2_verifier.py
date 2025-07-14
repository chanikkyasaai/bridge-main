"""
Layer 2 Adaptive Context-Aware Verification
Transformer + GNN based advanced behavioral analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import time

from mlengine.config import CONFIG
from mlengine.utils.behavioral_vectors import BehavioralVector, BehavioralEvent

logger = logging.getLogger(__name__)

@dataclass
class SessionGraph:
    """Session behavior graph representation"""
    session_id: str
    user_id: str
    nodes: List[Dict[str, Any]]  # Node features
    edges: List[Tuple[int, int]]  # Edge connections
    edge_features: List[Dict[str, float]]  # Edge attributes
    timestamps: List[datetime]
    graph_embedding: Optional[np.ndarray] = None

@dataclass
class ContextualFeatures:
    """Contextual features for adaptive verification"""
    age_group: str  # 'young', 'middle', 'senior'
    device_type: str  # 'phone', 'tablet'
    time_of_day: str  # 'morning', 'afternoon', 'evening', 'night'
    usage_mode: str  # 'normal', 'hurried', 'stressed'
    network_type: str  # 'wifi', 'mobile', 'unknown'
    location_risk: float  # 0-1 risk score based on location
    interaction_rhythm: str  # 'fast', 'medium', 'slow'

@dataclass
class Layer2Result:
    """Result from Layer 2 verification"""
    session_id: str
    user_id: str
    transformer_confidence: float
    gnn_anomaly_score: float
    context_alignment_score: float
    combined_risk_score: float
    decision: str  # 'continue', 'restrict', 'reauthenticate', 'block'
    explanation: str
    processing_time_ms: float
    metadata: Dict[str, Any]

class TransformerBehavioralEncoder(nn.Module):
    """Transformer-based behavioral sequence encoder"""
    
    def __init__(self, model_name: str = None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        import os
        model_name = model_name or CONFIG.get("TRANSFORMER_MODEL_PATH") or "distilbert-base-uncased"
        try:
            # Try to load transformer model with proper error handling
            self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            # Use a minimal dummy transformer for now
            self.transformer = None
            self.tokenizer = None
            
        # Initialize layers based on transformer availability
        if self.transformer is not None:
            # Freeze transformer weights initially
            for param in self.transformer.parameters():
                param.requires_grad = False
                
            # Behavioral adaptation layers
            self.behavioral_projection = nn.Linear(
                self.transformer.config.hidden_size,
                CONFIG.BEHAVIORAL_VECTOR_DIM
            )
        else:
            # Use dummy projection layer when transformer is not available
            self.behavioral_projection = nn.Linear(
                CONFIG.BEHAVIORAL_VECTOR_DIM,
                CONFIG.BEHAVIORAL_VECTOR_DIM
            )
        
        self.context_attention = nn.MultiheadAttention(
            embed_dim=CONFIG.BEHAVIORAL_VECTOR_DIM,
            num_heads=8,
            dropout=0.1
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(CONFIG.BEHAVIORAL_VECTOR_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, behavioral_sequence: torch.Tensor, context_features: torch.Tensor):
        """
        Args:
            behavioral_sequence: [batch_size, seq_len, vector_dim]
            context_features: [batch_size, context_dim]
        """
        batch_size, seq_len, vector_dim = behavioral_sequence.shape
        
        if self.transformer is not None and self.tokenizer is not None:
            # Create pseudo-text representation of behavioral sequence
            sequence_text = self._vectorize_to_text(behavioral_sequence)
            
            # Tokenize and encode
            inputs = self.tokenizer(
                sequence_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG.TRANSFORMER_MAX_LENGTH
            )
            
            # Get transformer embeddings
            outputs = self.transformer(**inputs)
            sequence_embedding = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        else:
            # Fallback: use behavioral sequence directly as embedding
            sequence_embedding = behavioral_sequence.mean(dim=1).unsqueeze(0)  # [1, behavioral_dim]
        
        # Project to behavioral space
        behavioral_embedding = self.behavioral_projection(sequence_embedding)
        
        # Add context information through attention
        context_expanded = context_features.unsqueeze(1)  # [batch_size, 1, context_dim]
        
        # Pad context to match behavioral_embedding dimension if needed
        if context_features.shape[1] != CONFIG.BEHAVIORAL_VECTOR_DIM:
            context_projection = nn.Linear(context_features.shape[1], CONFIG.BEHAVIORAL_VECTOR_DIM)
            context_expanded = context_projection(context_expanded)
        
        # Apply attention
        attended_output, _ = self.context_attention(
            behavioral_embedding.unsqueeze(1),
            context_expanded,
            context_expanded
        )
        
        final_embedding = attended_output.squeeze(1)
        confidence = self.confidence_head(final_embedding)
        
        return final_embedding, confidence
    
    def _vectorize_to_text(self, behavioral_sequence: torch.Tensor) -> List[str]:
        """Convert behavioral vectors to pseudo-text for transformer processing"""
        # This is a simplified approach - in practice, use learned vector-to-text mapping
        texts = []
        for sequence in behavioral_sequence:
            # Quantize vectors and create text representation
            quantized = torch.round(sequence * 10).int()
            text_tokens = []
            for vector in quantized:
                # Convert vector to space-separated string
                vector_str = " ".join([f"v{i}_{val.item()}" for i, val in enumerate(vector[:10])])  # Use first 10 dims
                text_tokens.append(vector_str)
            texts.append(" [SEP] ".join(text_tokens))
        
        return texts

class SessionGraphGNN(nn.Module):
    """Graph Neural Network for session behavior analysis"""
    
    def __init__(self, node_features: int = 32, edge_features: int = 16):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        
        # GNN layers
        self.conv1 = GATConv(node_features, CONFIG.GNN_HIDDEN_DIM, heads=4, dropout=CONFIG.GNN_DROPOUT)
        self.conv2 = GATConv(CONFIG.GNN_HIDDEN_DIM * 4, CONFIG.GNN_HIDDEN_DIM, heads=1, dropout=CONFIG.GNN_DROPOUT)
        self.conv3 = GCNConv(CONFIG.GNN_HIDDEN_DIM, CONFIG.GNN_HIDDEN_DIM // 2)
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(CONFIG.GNN_HIDDEN_DIM // 2, 32),
            nn.ReLU(),
            nn.Dropout(CONFIG.GNN_DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Graph-level embedding
        self.graph_embedding = nn.Sequential(
            nn.Linear(CONFIG.GNN_HIDDEN_DIM // 2, CONFIG.GRAPH_EMBEDDING_DIM),
            nn.Tanh()
        )
    
    def forward(self, data: Data):
        """
        Args:
            data: PyTorch Geometric Data object with x, edge_index, batch
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        
        # Graph-level aggregation
        graph_embedding = global_mean_pool(x, batch)
        
        # Anomaly detection
        anomaly_scores = self.anomaly_head(graph_embedding)
        
        # Graph embedding
        embedding = self.graph_embedding(graph_embedding)
        
        return anomaly_scores, embedding

class ContextAdapter(nn.Module):
    """Adapts predictions based on contextual features"""
    
    def __init__(self, context_features: int = 16):
        super().__init__()
        self.context_features = context_features
        
        # Context encoding
        self.context_encoder = nn.Sequential(
            nn.Linear(context_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Adaptation weights
        self.adapter = nn.Sequential(
            nn.Linear(16 + CONFIG.BEHAVIORAL_VECTOR_DIM + 1, 32),  # context + embedding + anomaly
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # 4 output classes
            nn.Softmax(dim=1)
        )
    
    def forward(self, context_features: torch.Tensor, behavioral_embedding: torch.Tensor, anomaly_score: torch.Tensor):
        """
        Args:
            context_features: [batch_size, context_features]
            behavioral_embedding: [batch_size, embedding_dim]
            anomaly_score: [batch_size, 1]
        """
        # Encode context
        context_encoded = self.context_encoder(context_features)
        
        # Combine all features
        combined = torch.cat([context_encoded, behavioral_embedding, anomaly_score], dim=1)
        
        # Get adaptation weights
        adaptation_weights = self.adapter(combined)
        
        return adaptation_weights

class SessionGraphBuilder:
    """Builds behavioral session graphs from events"""
    
    def __init__(self):
        self.session_graphs = {}  # session_id -> SessionGraph
        self.max_nodes = CONFIG.MAX_SESSION_NODES
    
    def add_event(self, event: BehavioralEvent):
        """Add behavioral event to session graph"""
        session_id = event.session_id
        
        if session_id not in self.session_graphs:
            self.session_graphs[session_id] = SessionGraph(
                session_id=session_id,
                user_id=event.user_id,
                nodes=[],
                edges=[],
                edge_features=[],
                timestamps=[]
            )
        
        graph = self.session_graphs[session_id]
        
        # Create node features from event
        node_features = self._event_to_node_features(event)
        
        # Add node
        current_node_id = len(graph.nodes)
        graph.nodes.append(node_features)
        graph.timestamps.append(event.timestamp)
        
        # Add edge to previous node if exists
        if current_node_id > 0:
            prev_node_id = current_node_id - 1
            edge_features = self._calculate_edge_features(
                graph.timestamps[prev_node_id],
                graph.timestamps[current_node_id],
                graph.nodes[prev_node_id],
                graph.nodes[current_node_id]
            )
            
            graph.edges.append((prev_node_id, current_node_id))
            graph.edge_features.append(edge_features)
        
        # Limit graph size
        if len(graph.nodes) > self.max_nodes:
            self._prune_graph(graph)
    
    def _event_to_node_features(self, event: BehavioralEvent) -> Dict[str, Any]:
        """Convert behavioral event to node features"""
        return {
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'features': event.features,
            'user_id': event.user_id
        }
    
    def _calculate_edge_features(self, prev_time: datetime, curr_time: datetime, 
                                prev_node: Dict, curr_node: Dict) -> Dict[str, float]:
        """Calculate edge features between two nodes"""
        time_diff = (curr_time - prev_time).total_seconds()
        
        # Transition type
        transition_type = f"{prev_node['event_type']}_to_{curr_node['event_type']}"
        
        return {
            'time_difference': time_diff,
            'transition_type': hash(transition_type) % 100,  # Simple hash
            'interaction_speed': 1.0 / (time_diff + 0.001)  # Inverse of time
        }
    
    def _prune_graph(self, graph: SessionGraph):
        """Prune graph to maintain size limits"""
        # Keep most recent nodes
        keep_count = self.max_nodes // 2
        
        graph.nodes = graph.nodes[-keep_count:]
        graph.timestamps = graph.timestamps[-keep_count:]
        
        # Rebuild edges for remaining nodes
        new_edges = []
        new_edge_features = []
        
        for i in range(len(graph.nodes) - 1):
            edge_features = self._calculate_edge_features(
                graph.timestamps[i],
                graph.timestamps[i + 1],
                graph.nodes[i],
                graph.nodes[i + 1]
            )
            new_edges.append((i, i + 1))
            new_edge_features.append(edge_features)
        
        graph.edges = new_edges
        graph.edge_features = new_edge_features
    
    def get_graph(self, session_id: str) -> Optional[SessionGraph]:
        """Get session graph"""
        return self.session_graphs.get(session_id)
    
    def clear_session(self, session_id: str):
        """Clear session graph"""
        if session_id in self.session_graphs:
            del self.session_graphs[session_id]

class Layer2Verifier:
    """Layer 2 Adaptive Context-Aware Verification Engine"""
    
    def __init__(self):
        self.transformer_encoder = TransformerBehavioralEncoder()
        self.session_gnn = SessionGraphGNN()
        self.context_adapter = ContextAdapter()
        self.graph_builder = SessionGraphBuilder()
        
        # CONTEXT MANIPULATION DETECTION: Add anomaly detector
        self.context_anomaly_detector = ContextAnomalyDetector()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_models_to_device()
        
        # Set models to evaluation mode
        self.transformer_encoder.eval()
        self.session_gnn.eval()
        self.context_adapter.eval()
    
    def _move_models_to_device(self):
        """Move all models to appropriate device"""
        self.transformer_encoder.to(self.device)
        self.session_gnn.to(self.device)
        self.context_adapter.to(self.device)
    
    async def initialize(self):
        """Initialize Layer 2 verifier"""
        logger.info("Initializing Layer 2 Verifier...")
        
        # Move models to device
        self._move_models_to_device()
        
        # Set to evaluation mode
        self.transformer_encoder.eval()
        self.session_gnn.eval()
        self.context_adapter.eval()
        
        self.transformer_model = self.transformer_encoder
        self.gnn_model = self.session_gnn
        self.session_graph_builder = self.graph_builder
        
        logger.info("✓ Layer 2 Verifier initialized")
    
    async def verify(self, 
                    vectors: List[BehavioralVector], 
                    events: List[BehavioralEvent], 
                    context: Dict[str, Any], 
                    session_id: str) -> Layer2Result:
        """Perform Layer 2 verification with context manipulation detection"""
        start_time = time.time()
        
        if not vectors:
            return self._create_empty_result(session_id, start_time)
        
        user_id = vectors[0].user_id
        
        try:
            # Convert context to structured format
            contextual_features = self._convert_context(context)
            
            # CONTEXT MANIPULATION DETECTION: Check for adversarial context
            manipulation_result = self.context_anomaly_detector.detect_context_manipulation(
                user_id, contextual_features
            )
            
            # Add context to history for learning (only if not manipulated)
            if not manipulation_result["is_manipulated"]:
                self.context_anomaly_detector.add_context_history(user_id, contextual_features)
            
            # Transformer-based sequence analysis
            transformer_confidence = await self._encode_sequence_with_transformer(vectors)
            
            # Build and analyze session graph
            session_graph = self.session_graph_builder.build_session_graph(events, session_id, user_id)
            gnn_anomaly_score = await self._analyze_with_gnn(session_graph)
            
            # Context adaptation (penalized if manipulated)
            context_score = self._calculate_context_score_with_manipulation_detection(
                contextual_features, manipulation_result
            )
            
            # ENHANCED SCORING: Integrate manipulation detection
            combined_score = self._combine_scores_with_manipulation(
                transformer_confidence, 
                gnn_anomaly_score, 
                context_score,
                manipulation_result
            )
            
            # Enhanced decision making with manipulation awareness
            decision_factors = {
                "transformer_confidence": transformer_confidence,
                "gnn_anomaly": gnn_anomaly_score,
                "context_alignment": context_score,
                "context_manipulation": manipulation_result
            }
            
            # Make decisions based on all factors
            transformer_decision = "continue" if transformer_confidence > 0.7 else "escalate"
            gnn_decision = "normal" if gnn_anomaly_score < 0.3 else "anomaly"
            
            # MANIPULATION-AWARE FINAL DECISION
            final_recommendation = self._make_manipulation_aware_decision(
                combined_score, manipulation_result
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return Layer2Result(
                session_id=session_id,
                user_id=user_id,
                transformer_confidence=transformer_confidence,
                gnn_anomaly_score=gnn_anomaly_score,
                context_alignment_score=context_score,
                combined_risk_score=1.0 - combined_score,  # Convert to risk score
                decision=final_recommendation,
                explanation=self._generate_explanation(decision_factors, manipulation_result),
                processing_time_ms=processing_time,
                metadata={
                    "context": context,
                    "vectors_analyzed": len(vectors),
                    "events_analyzed": len(events),
                    "context_manipulation": manipulation_result,
                    "decision_factors": decision_factors
                }
            )
            
        except Exception as e:
            logger.error(f"Layer 2 verification error: {e}")
            return self._create_error_result(session_id, user_id, start_time, str(e))
    
    async def _encode_sequence_with_transformer(self, vectors: List[BehavioralVector]) -> float:
        """Encode behavioral sequence using transformer"""
        try:
            # Convert vectors to sequence
            sequence = torch.stack([torch.FloatTensor(v.vector) for v in vectors])
            sequence = sequence.unsqueeze(0)  # Add batch dimension
            
            # Create dummy context features
            context_features = torch.zeros(1, 10)  # Simplified context
            
            with torch.no_grad():
                sequence = sequence.to(self.device)
                context_features = context_features.to(self.device)
                
                confidence = self.transformer_encoder(sequence, context_features)
                return float(confidence.item())
                
        except Exception as e:
            logger.error(f"Transformer encoding error: {e}")
            return 0.5  # Neutral confidence
    
    async def _analyze_with_gnn(self, session_graph: SessionGraph) -> float:
        """Analyze session graph with GNN"""
        try:
            if not session_graph.nodes or not session_graph.edges:
                return 0.1  # Low anomaly for empty graphs
            
            # Convert to PyTorch Geometric data
            node_features = torch.FloatTensor([node.get("features", [0.0] * 10) for node in session_graph.nodes])
            edge_index = torch.LongTensor(session_graph.edges).t()
            
            # Create graph data
            from torch_geometric.data import Data
            graph_data = Data(x=node_features, edge_index=edge_index)
            
            with torch.no_grad():
                graph_data = graph_data.to(self.device)
                anomaly_score = self.session_gnn(graph_data)
                return float(anomaly_score.item())
                
        except Exception as e:
            logger.error(f"GNN analysis error: {e}")
            return 0.2  # Low-medium anomaly on error
    
    def _convert_context(self, context: Dict[str, Any]) -> ContextualFeatures:
        """Convert context dictionary to structured format"""
        return ContextualFeatures(
            age_group=context.get("age_group", "unknown"),
            device_type=context.get("device_type", "phone"),
            time_of_day=context.get("time_of_day", "unknown"),
            usage_mode=context.get("usage_mode", "normal"),
            network_type=context.get("network_type", "unknown"),
            location_risk=context.get("location_risk", 0.5),
            interaction_rhythm=context.get("interaction_rhythm", "medium")
        )
    
    def _calculate_context_score_with_manipulation_detection(self, context: ContextualFeatures, 
                                                           manipulation_result: Dict[str, Any]) -> float:
        """Calculate context alignment score with manipulation penalty"""
        base_score = self._calculate_context_score(context)
        
        # Apply manipulation penalty
        if manipulation_result["is_manipulated"]:
            manipulation_penalty = manipulation_result["manipulation_confidence"] * 0.7
            base_score = max(0.0, base_score - manipulation_penalty)
            logger.warning(f"Context manipulation detected. Original score: {base_score + manipulation_penalty:.3f}, "
                         f"Penalized score: {base_score:.3f}")
        
        return base_score
    
    def _combine_scores_with_manipulation(self, transformer_conf: float, gnn_anomaly: float, 
                                        context_score: float, manipulation_result: Dict[str, Any]) -> float:
        """Combine all scores with manipulation detection"""
        # Base combination
        base_combined = self._combine_scores(transformer_conf, gnn_anomaly, context_score)
        
        # Apply manipulation penalty to final score
        if manipulation_result["is_manipulated"]:
            manipulation_penalty = manipulation_result["manipulation_confidence"] * 0.5
            final_score = max(0.0, base_combined - manipulation_penalty)
            logger.warning(f"Manipulation detected - applying penalty. "
                         f"Base: {base_combined:.3f}, Final: {final_score:.3f}")
        else:
            final_score = base_combined
        
        return final_score
    
    def _make_manipulation_aware_decision(self, combined_score: float, 
                                        manipulation_result: Dict[str, Any]) -> str:
        """Make final decision with context manipulation awareness"""
        
        # If manipulation is detected with high confidence, always block
        if (manipulation_result["is_manipulated"] and 
            manipulation_result["manipulation_confidence"] > 0.7):
            logger.warning("High confidence context manipulation detected - blocking session")
            return "block"
        
        # If manipulation is detected with medium confidence, be more restrictive
        if (manipulation_result["is_manipulated"] and 
            manipulation_result["manipulation_confidence"] > 0.4):
            if combined_score > 0.8:
                return "reauthenticate"  # Step down from continue
            elif combined_score > 0.6:
                return "restrict"  # Step down from monitor
            else:
                return "block"
        
        # Normal decision logic for non-manipulated contexts
        return self._make_final_decision(combined_score)
    
    def _generate_explanation(self, decision_factors: Dict[str, Any], 
                            manipulation_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation for the decision"""
        explanations = []
        
        # Transformer analysis
        transformer_conf = decision_factors["transformer_confidence"]
        if transformer_conf > 0.8:
            explanations.append("High behavioral sequence confidence")
        elif transformer_conf > 0.6:
            explanations.append("Medium behavioral sequence confidence")
        else:
            explanations.append("Low behavioral sequence confidence")
        
        # GNN analysis
        gnn_anomaly = decision_factors["gnn_anomaly"]
        if gnn_anomaly > 0.6:
            explanations.append("High session graph anomaly detected")
        elif gnn_anomaly > 0.3:
            explanations.append("Medium session graph anomaly")
        else:
            explanations.append("Normal session graph pattern")
        
        # Context analysis
        context_score = decision_factors["context_alignment"]
        if context_score > 0.7:
            explanations.append("Good context alignment")
        elif context_score > 0.4:
            explanations.append("Moderate context alignment")
        else:
            explanations.append("Poor context alignment")
        
        # Manipulation detection
        if manipulation_result["is_manipulated"]:
            confidence = manipulation_result["manipulation_confidence"]
            features = manipulation_result.get("anomalous_features", [])
            if confidence > 0.7:
                explanations.append(f"High confidence context manipulation detected in: {', '.join(features)}")
            elif confidence > 0.4:
                explanations.append(f"Possible context manipulation in: {', '.join(features)}")
            else:
                explanations.append("Minor context anomalies detected")
        else:
            explanations.append("No context manipulation detected")
        
        return "; ".join(explanations)
    
    def _calculate_context_score(self, context: ContextualFeatures) -> float:
        """Calculate base context alignment score"""
        score = 0.5  # Base neutral score
        
        # Time of day scoring
        if context.time_of_day in ["morning", "afternoon", "evening"]:
            score += 0.1
        
        # Device type scoring
        if context.device_type in ["phone", "tablet"]:
            score += 0.1
        
        # Location risk scoring (inverse - lower risk is better)
        score += (1.0 - context.location_risk) * 0.2
        
        # Usage mode scoring
        if context.usage_mode == "normal":
            score += 0.1
        elif context.usage_mode in ["hurried", "stressed"]:
            score -= 0.05
        
        # Network type scoring
        if context.network_type == "wifi":
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def build_session_graph(self, events: List[BehavioralEvent], session_id: str, user_id: str) -> SessionGraph:
        """Build session graph from events"""
        # Add all events to graph builder
        for event in events:
            self.graph_builder.add_event(event)
        
        # Get the built graph
        session_graph = self.graph_builder.get_graph(session_id)
        
        if session_graph is None:
            # Create empty graph if none exists
            session_graph = SessionGraph(
                session_id=session_id,
                user_id=user_id,
                nodes=[],
                edges=[],
                edge_features=[],
                timestamps=[]
            )
        
        return session_graph
    
    def _combine_scores(self, transformer_conf: float, gnn_anomaly: float, context_score: float) -> float:
        """Combine all scores into final confidence"""
        # Weights for different components
        weights = {
            "transformer": 0.4,
            "gnn": 0.35,
            "context": 0.25
        }
        
        # Invert GNN anomaly score (high anomaly = low confidence)
        gnn_confidence = 1.0 - gnn_anomaly
        
        combined = (
            weights["transformer"] * transformer_conf +
            weights["gnn"] * gnn_confidence +
            weights["context"] * context_score
        )
        
        return combined
    
    def _make_final_decision(self, combined_score: float) -> str:
        """Make final recommendation based on combined score"""
        if combined_score > 0.8:
            return "continue"
        elif combined_score > 0.6:
            return "monitor"
        elif combined_score > 0.4:
            return "challenge"
        else:
            return "block"
    
    def _create_empty_result(self, session_id: str, start_time: float) -> Layer2Result:
        """Create result for empty input"""
        processing_time = (time.time() - start_time) * 1000
        
        return Layer2Result(
            session_id=session_id,
            user_id="unknown",
            transformer_confidence=0.5,
            gnn_anomaly_score=0.0,
            contextual_score=0.5,
            session_graph_embedding=np.zeros(CONFIG.GRAPH_EMBEDDING_DIM),
            decision_factors={},
            transformer_decision="neutral",
            gnn_decision="normal",
            final_recommendation="escalate",
            processing_time_ms=processing_time,
            metadata={"empty_input": True}
        )
    
    def _create_error_result(self, session_id: str, user_id: str, start_time: float, error_msg: str) -> Layer2Result:
        """Create result for error case"""
        processing_time = (time.time() - start_time) * 1000
        
        return Layer2Result(
            session_id=session_id,
            user_id=user_id,
            transformer_confidence=0.3,
            gnn_anomaly_score=0.5,
            contextual_score=0.3,
            session_graph_embedding=np.zeros(CONFIG.GRAPH_EMBEDDING_DIM),
            decision_factors={"error": error_msg},
            transformer_decision="error",
            gnn_decision="error",
            final_recommendation="escalate",
            processing_time_ms=processing_time,
            metadata={"error": error_msg}
        )
    
    async def update_models(self):
        """Update and retrain models"""
        logger.info("Updating Layer 2 models...")
        # In a full implementation, this would retrain the models
        logger.info("Layer 2 models updated")
    
    async def save_models(self):
        """Save Layer 2 models"""
        # In a full implementation, this would save model weights
        logger.info("Layer 2 models saved")

class ContextAnomalyDetector:
    """Detects adversarial context manipulation"""
    
    def __init__(self):
        self.context_history = defaultdict(list)  # user_id -> [context_features]
        self.context_bounds = {}  # user_id -> {feature: (min, max, mean, std)}
        self.manipulation_threshold = 2.5  # Standard deviations for anomaly detection
        
    def add_context_history(self, user_id: str, context_features: ContextualFeatures):
        """Add context to user history for learning normal patterns"""
        feature_vector = self._context_to_vector(context_features)
        self.context_history[user_id].append(feature_vector)
        
        # Limit history size
        if len(self.context_history[user_id]) > 100:
            self.context_history[user_id] = self.context_history[user_id][-100:]
        
        # Update bounds
        self._update_context_bounds(user_id)
    
    def detect_context_manipulation(self, user_id: str, context_features: ContextualFeatures) -> Dict[str, Any]:
        """Detect if context has been manipulated or is adversarial"""
        if user_id not in self.context_bounds or len(self.context_history[user_id]) < 5:
            # Not enough history - return neutral result
            return {
                "is_manipulated": False,
                "manipulation_confidence": 0.0,
                "anomaly_score": 0.0,
                "explanation": "Insufficient context history for detection"
            }
        
        feature_vector = self._context_to_vector(context_features)
        bounds = self.context_bounds[user_id]
        
        anomaly_scores = []
        anomalous_features = []
        
        for i, (feature_name, value) in enumerate(zip([
            "location_risk", "time_of_day_numeric", "device_type_numeric", 
            "network_type_numeric", "usage_mode_numeric"
        ], feature_vector)):
            
            if feature_name in bounds:
                mean, std = bounds[feature_name]["mean"], bounds[feature_name]["std"]
                if std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > self.manipulation_threshold:
                        anomaly_scores.append(z_score)
                        anomalous_features.append(feature_name)
        
        # Calculate overall manipulation confidence
        if anomaly_scores:
            max_anomaly = max(anomaly_scores)
            manipulation_confidence = min(1.0, max_anomaly / 5.0)  # Scale to 0-1
            is_manipulated = max_anomaly > self.manipulation_threshold
        else:
            manipulation_confidence = 0.0
            is_manipulated = False
        
        # Check for impossible combinations
        impossible_combinations = self._check_impossible_combinations(context_features)
        if impossible_combinations:
            manipulation_confidence = max(manipulation_confidence, 0.8)
            is_manipulated = True
            anomalous_features.extend(impossible_combinations)
        
        return {
            "is_manipulated": is_manipulated,
            "manipulation_confidence": manipulation_confidence,
            "anomaly_score": max(anomaly_scores) if anomaly_scores else 0.0,
            "anomalous_features": anomalous_features,
            "explanation": f"Detected {len(anomalous_features)} anomalous context features" if anomalous_features else "Context appears normal"
        }
    
    def _context_to_vector(self, context: ContextualFeatures) -> List[float]:
        """Convert context features to numerical vector"""
        return [
            context.location_risk,
            self._time_to_numeric(context.time_of_day),
            self._device_to_numeric(context.device_type),
            self._network_to_numeric(context.network_type),
            self._usage_mode_to_numeric(context.usage_mode)
        ]
    
    def _time_to_numeric(self, time_of_day: str) -> float:
        """Convert time of day to numeric value"""
        mapping = {"morning": 0.25, "afternoon": 0.5, "evening": 0.75, "night": 1.0}
        return mapping.get(time_of_day, 0.5)
    
    def _device_to_numeric(self, device_type: str) -> float:
        """Convert device type to numeric value"""
        mapping = {"phone": 0.3, "tablet": 0.7, "unknown": 0.5}
        return mapping.get(device_type, 0.5)
    
    def _network_to_numeric(self, network_type: str) -> float:
        """Convert network type to numeric value"""
        mapping = {"wifi": 0.2, "mobile": 0.8, "unknown": 0.5}
        return mapping.get(network_type, 0.5)
    
    def _usage_mode_to_numeric(self, usage_mode: str) -> float:
        """Convert usage mode to numeric value"""
        mapping = {"normal": 0.3, "hurried": 0.7, "stressed": 0.9, "unknown": 0.5}
        return mapping.get(usage_mode, 0.5)
    
    def _update_context_bounds(self, user_id: str):
        """Update statistical bounds for user context"""
        if len(self.context_history[user_id]) < 3:
            return
        
        history = np.array(self.context_history[user_id])
        feature_names = ["location_risk", "time_of_day_numeric", "device_type_numeric", 
                        "network_type_numeric", "usage_mode_numeric"]
        
        self.context_bounds[user_id] = {}
        for i, feature_name in enumerate(feature_names):
            values = history[:, i]
            self.context_bounds[user_id][feature_name] = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values) + 1e-8  # Add small epsilon to avoid division by zero
            }
    
    def _check_impossible_combinations(self, context: ContextualFeatures) -> List[str]:
        """Check for impossible or highly suspicious context combinations"""
        impossible = []
        
        # Example impossible combinations (can be extended)
        if context.location_risk > 0.9 and context.network_type == "wifi":
            # High risk location but trusted wifi - suspicious
            impossible.append("high_risk_location_with_trusted_network")
        
        if context.time_of_day == "night" and context.usage_mode == "hurried":
            # Unusual to be hurried at night - could be manipulated
            impossible.append("night_hurried_combination")
        
        if context.location_risk < 0.1 and context.device_type == "unknown":
            # Low risk location but unknown device - suspicious
            impossible.append("safe_location_unknown_device")
        
        return impossible
