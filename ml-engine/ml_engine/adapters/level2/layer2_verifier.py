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

from ml_engine.config import CONFIG
from ml_engine.utils.behavioral_vectors import BehavioralVector, BehavioralEvent

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
            # Try local path first
            if os.path.exists(model_name):
                self.transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            else:
                self.transformer = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            try:
                # Fallback: try online if local fails
                self.transformer = AutoModel.from_pretrained("distilbert-base-uncased")
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            except Exception as e2:
                raise RuntimeError(f"Could not load transformer model '{model_name}': {e}\nAlso failed fallback: {e2}\nCheck your internet connection or set TRANSFORMER_MODEL_PATH in config.py to a local model directory.")
            
        # Freeze transformer weights initially
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Behavioral adaptation layers
        self.behavioral_projection = nn.Linear(
            self.transformer.config.hidden_size,
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
        
        # Create pseudo-text representation of behavioral sequence
        # In practice, this would be learned embeddings
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
    
    def build_session_graph(self, events: List[BehavioralEvent], session_id: str, user_id: str) -> SessionGraph:
        """Build session graph from list of events"""
        # Clear existing session
        if session_id in self.session_graphs:
            del self.session_graphs[session_id]
        
        # Add all events
        for event in events:
            self.add_event(event)
        
        # Return the built graph
        return self.get_graph(session_id) or SessionGraph(
            session_id=session_id,
            user_id=user_id,
            nodes=[],
            edges=[],
            edge_features=[],
            timestamps=[]
        )

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
        self.session_graph_builder = SessionGraphBuilder()
        
        # CONTEXT MANIPULATION DETECTION: Add anomaly detector
        self.context_anomaly_detector = ContextAnomalyDetector()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add session tracking for direct access
        self.session_graphs = {}
        
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
            manipulation_result = self.context_anomaly_detector.detect_manipulation(
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
            
            # Convert node dictionaries to numeric features (32 dimensions to match GNN)
            numeric_features = []
            for node in session_graph.nodes:
                features = node.get("features", {})
                # Extract numeric values and pad to 32 dimensions
                feature_vector = [
                    features.get("pressure", 0.5),
                    features.get("velocity", 1.0),
                    features.get("duration", 0.2),
                    features.get("x_coordinate", 0.0) / 1000.0,  # Normalize coordinates
                    features.get("y_coordinate", 0.0) / 1000.0,
                    1.0 if node.get("event_type") == "touch" else 0.0,
                    1.0 if node.get("event_type") == "swipe" else 0.0,
                    1.0 if node.get("event_type") == "type" else 0.0,
                    1.0 if node.get("event_type") == "scroll" else 0.0,
                    1.0 if node.get("event_type") == "tap" else 0.0,
                    # Add padding features to reach 32 dimensions
                    features.get("screen_width", 1080.0) / 2000.0,
                    features.get("screen_height", 1920.0) / 3000.0,
                    len(node.get("user_id", "")) / 100.0,  # User ID length feature
                ] + [0.1] * 19  # Pad with default values to reach 32 features
                
                # Ensure exactly 32 features
                feature_vector = feature_vector[:32]
                while len(feature_vector) < 32:
                    feature_vector.append(0.1)
                    
                numeric_features.append(feature_vector)
            
            # Convert to PyTorch tensors
            node_features = torch.FloatTensor(numeric_features)
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
        """Build session graph from events - delegates to session graph builder"""
        graph = self.session_graph_builder.build_session_graph(events, session_id, user_id)
        # Also keep a reference in our local session_graphs dict
        self.session_graphs[session_id] = graph
        return graph
    
    def get_graph(self, session_id: str) -> Optional[SessionGraph]:
        """Get session graph - check both local and builder"""
        return self.session_graphs.get(session_id) or self.session_graph_builder.get_graph(session_id)
    
    def clear_session(self, session_id: str):
        """Clear session graph from both local and builder"""
        if session_id in self.session_graphs:
            del self.session_graphs[session_id]
        self.session_graph_builder.clear_session(session_id)

    def _create_empty_result(self, session_id: str, start_time: float) -> Layer2Result:
        """Create empty result for sessions with no vectors"""
        import time
        return Layer2Result(
            session_id=session_id,
            user_id="",
            transformer_confidence=0.0,
            gnn_anomaly_score=0.5,
            context_alignment_score=0.0,
            combined_risk_score=1.0,  # High risk by default
            decision="block",
            explanation="No behavioral vectors provided",
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={}
        )

    def _create_error_result(self, session_id: str, user_id: str, start_time: float, error_msg: str) -> Layer2Result:
        """Create error result for failed verification"""
        import time
        return Layer2Result(
            session_id=session_id,
            user_id=user_id,
            transformer_confidence=0.0,
            gnn_anomaly_score=0.5,
            context_alignment_score=0.0,
            combined_risk_score=1.0,  # High risk due to error
            decision="block",
            explanation=f"Verification failed: {error_msg}",
            processing_time_ms=(time.time() - start_time) * 1000,
            metadata={"error": True, "error_message": error_msg}
        )

    def _combine_scores(self, transformer_conf: float, gnn_anomaly: float, context_score: float) -> float:
        """Combine individual scores into final risk score"""
        # Weighted combination
        weights = {
            "transformer": 0.4,
            "gnn": 0.35,
            "context": 0.25
        }
        
        # Higher GNN anomaly means higher risk
        gnn_risk = gnn_anomaly
        # Lower transformer confidence means higher risk
        transformer_risk = 1.0 - transformer_conf
        # Lower context score means higher risk
        context_risk = 1.0 - context_score
        
        combined_risk = (
            weights["transformer"] * transformer_risk +
            weights["gnn"] * gnn_risk +
            weights["context"] * context_risk
        )
        
        return max(0.0, min(1.0, combined_risk))

    def _make_final_decision(self, combined_score: float) -> str:
        """Make final verification decision based on combined score"""
        if combined_score < 0.3:
            return "continue"
        elif combined_score < 0.5:
            return "restrict"
        elif combined_score < 0.7:
            return "reauthenticate"
        else:
            return "block"

class ContextAnomalyDetector(nn.Module):
    """Context manipulation and anomaly detection system"""
    
    def __init__(self, context_features: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.context_features = context_features
        self.hidden_dim = hidden_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 16)
        )
        
        # Anomaly detector
        self.anomaly_detector = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_features: [batch_size, context_features]
        Returns:
            anomaly_score: [batch_size, 1] - probability of context manipulation
        """
        encoded = self.context_encoder(context_features)
        anomaly_score = self.anomaly_detector(encoded)
        return anomaly_score
    
    def detect_manipulation(self, context_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detect context manipulation from context dictionary"""
        try:
            # Convert context to features
            features = self._extract_context_features(context_dict)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                anomaly_score = self.forward(features_tensor)
                score = float(anomaly_score.item())
            
            # Determine if manipulation is detected
            threshold = 0.5
            is_manipulated = score > threshold
            
            # Identify anomalous features
            anomalous_features = self._identify_anomalous_features(context_dict, score)
            
            return {
                "is_manipulated": is_manipulated,
                "manipulation_confidence": score,
                "anomalous_features": anomalous_features,
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Context manipulation detection error: {e}")
            return {
                "is_manipulated": False,
                "manipulation_confidence": 0.0,
                "anomalous_features": [],
                "threshold": 0.5,
                "error": str(e)
            }
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[float]:
        """Extract numeric features from context dictionary"""
        features = []
        
        # Age group encoding
        age_map = {"young": 0.2, "middle": 0.5, "senior": 0.8, "unknown": 0.5}
        features.append(age_map.get(context.get("age_group", "unknown"), 0.5))
        
        # Device type encoding
        device_map = {"phone": 0.3, "tablet": 0.6, "desktop": 0.9, "unknown": 0.5}
        features.append(device_map.get(context.get("device_type", "unknown"), 0.5))
        
        # Time of day encoding
        time_map = {"morning": 0.25, "afternoon": 0.5, "evening": 0.75, "night": 1.0, "unknown": 0.5}
        features.append(time_map.get(context.get("time_of_day", "unknown"), 0.5))
        
        # Usage mode encoding
        usage_map = {"normal": 0.3, "hurried": 0.7, "stressed": 0.9, "unknown": 0.5}
        features.append(usage_map.get(context.get("usage_mode", "unknown"), 0.5))
        
        # Network type encoding
        network_map = {"wifi": 0.2, "mobile": 0.5, "ethernet": 0.1, "unknown": 0.5}
        features.append(network_map.get(context.get("network_type", "unknown"), 0.5))
        
        # Location risk (direct numeric)
        features.append(context.get("location_risk", 0.5))
        
        # Interaction rhythm encoding
        rhythm_map = {"slow": 0.2, "medium": 0.5, "fast": 0.8, "unknown": 0.5}
        features.append(rhythm_map.get(context.get("interaction_rhythm", "unknown"), 0.5))
        
        # Pad to required feature count
        while len(features) < self.context_features:
            features.append(0.1)
        
        return features[:self.context_features]
    
    def _identify_anomalous_features(self, context: Dict[str, Any], anomaly_score: float) -> List[str]:
        """Identify which context features appear anomalous"""
        anomalous = []
        
        # Simple heuristic based on unusual combinations
        if anomaly_score > 0.7:
            # High anomaly - check for suspicious patterns
            
            # Unusual time patterns
            if context.get("time_of_day") == "night" and context.get("usage_mode") == "hurried":
                anomalous.append("time_usage_pattern")
            
            # High location risk
            if context.get("location_risk", 0) > 0.5:
                anomalous.append("location_risk")
            
            # Inconsistent device/usage patterns
            if context.get("device_type") == "desktop" and context.get("interaction_rhythm") == "fast":
                anomalous.append("device_interaction_mismatch")
                
        elif anomaly_score > 0.4:
            # Medium anomaly
            if context.get("location_risk", 0) > 0.3:
                anomalous.append("elevated_location_risk")
                
        return anomalous