"""
Layer G: Session Graph Generator
Constructs real-time behavioral graphs with nodes as actions and edges as transitions.
Critical for national-level behavioral authentication system.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import json

from src.data.models import BehavioralVector, UserProfile

logger = logging.getLogger(__name__)

class ActionType(str, Enum):
    """Types of behavioral actions that become graph nodes"""
    TAP = "tap"
    SCROLL = "scroll"  
    SWIPE = "swipe"
    TYPING = "typing"
    SCREEN_SWITCH = "screen_switch"
    PAUSE = "pause"
    NAVIGATION = "navigation"
    TRANSACTION_START = "transaction_start"
    TRANSACTION_COMPLETE = "transaction_complete"
    MPIN_ENTRY = "mpin_entry"
    BIOMETRIC_AUTH = "biometric_auth"

class TransitionType(str, Enum):
    """Types of transitions between actions (graph edges)"""
    SEQUENTIAL = "sequential"  # One action follows another
    RAPID = "rapid"           # Actions within 500ms
    DELAYED = "delayed"       # Actions after 3s+ pause
    INTERRUPTED = "interrupted" # Action sequence broken
    CONTEXTUAL = "contextual"  # Related to transaction context

@dataclass
class BehavioralNode:
    """Node in the behavioral graph representing an action"""
    node_id: str
    action_type: ActionType
    timestamp: datetime
    duration: float  # Action duration in ms
    coordinates: Optional[Tuple[float, float]] = None
    pressure: Optional[float] = None
    screen_id: Optional[str] = None
    sequence_position: int = 0
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class BehavioralEdge:
    """Edge in the behavioral graph representing a transition"""
    edge_id: str
    source_node: str
    target_node: str
    transition_type: TransitionType
    time_gap: float  # Time between actions in ms
    spatial_distance: Optional[float] = None  # For touch actions
    confidence: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionGraph:
    """Complete behavioral graph for a user session"""
    session_id: str
    user_id: str
    start_time: datetime
    nodes: Dict[str, BehavioralNode] = field(default_factory=dict)
    edges: Dict[str, BehavioralEdge] = field(default_factory=dict)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    def add_node(self, node: BehavioralNode):
        """Add a behavioral node to the graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            action_type=node.action_type.value,
            timestamp=node.timestamp,
            duration=node.duration,
            features=node.features
        )
    
    def add_edge(self, edge: BehavioralEdge):
        """Add a behavioral edge to the graph"""
        self.edges[edge.edge_id] = edge
        self.graph.add_edge(
            edge.source_node,
            edge.target_node,
            transition_type=edge.transition_type.value,
            time_gap=edge.time_gap,
            confidence=edge.confidence,
            features=edge.features
        )

class SessionGraphGenerator:
    """
    Layer G: Session Graph Generator
    
    Constructs real-time behavioral graphs from mobile behavioral data.
    Each user action becomes a node, transitions become edges.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Graph construction parameters
        self.rapid_threshold = 500   # ms - actions within this are "rapid"
        self.delayed_threshold = 3000  # ms - actions after this are "delayed"
        self.spatial_threshold = 100   # pixels - for spatial relationships
        
        # Node sequence tracking
        self.node_counter = 0
        self.edge_counter = 0
        
        self.logger.info("Session Graph Generator (Layer G) initialized")
    
    def generate_session_graph(
        self,
        user_id: str,
        session_id: str,
        behavioral_events: List[Dict[str, Any]]
    ) -> SessionGraph:
        """
        Generate behavioral graph from session events
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            behavioral_events: List of behavioral events from mobile
            
        Returns:
            SessionGraph with nodes and edges constructed
        """
        try:
            # Initialize session graph
            graph = SessionGraph(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now()
            )
            
            if not behavioral_events:
                return graph
            
            # Sort events by timestamp
            sorted_events = sorted(behavioral_events, key=lambda x: x.get('timestamp', 0))
            
            # Convert events to nodes
            nodes = []
            for i, event in enumerate(sorted_events):
                node = self._create_node_from_event(event, i)
                if node:
                    nodes.append(node)
                    graph.add_node(node)
            
            # Create edges between consecutive nodes
            for i in range(len(nodes) - 1):
                edge = self._create_edge_between_nodes(nodes[i], nodes[i + 1])
                if edge:
                    graph.add_edge(edge)
            
            # Add contextual edges (non-sequential relationships)
            contextual_edges = self._find_contextual_edges(nodes)
            for edge in contextual_edges:
                graph.add_edge(edge)
            
            # Calculate graph metrics
            graph_metrics = self._calculate_graph_metrics(graph)
            
            self.logger.info(
                f"Generated session graph for {user_id}: "
                f"{len(graph.nodes)} nodes, {len(graph.edges)} edges"
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Error generating session graph: {e}")
            # Return empty graph on error
            return SessionGraph(
                session_id=session_id,
                user_id=user_id,
                start_time=datetime.now()
            )
    
    def _create_node_from_event(self, event: Dict[str, Any], sequence_pos: int) -> Optional[BehavioralNode]:
        """Convert behavioral event to graph node"""
        try:
            event_type = event.get('event_type', '')
            
            # Map event types to action types
            action_mapping = {
                'touch': ActionType.TAP,
                'scroll': ActionType.SCROLL,
                'swipe': ActionType.SWIPE,
                'typing': ActionType.TYPING,
                'screen_change': ActionType.SCREEN_SWITCH,
                'transaction_start': ActionType.TRANSACTION_START,
                'transaction_complete': ActionType.TRANSACTION_COMPLETE,
                'mpin_entry': ActionType.MPIN_ENTRY,
                'biometric': ActionType.BIOMETRIC_AUTH
            }
            
            action_type = action_mapping.get(event_type)
            if not action_type:
                return None
            
            self.node_counter += 1
            node_id = f"node_{self.node_counter:06d}"
            
            # Extract coordinates if available
            coordinates = None
            if 'x' in event and 'y' in event:
                coordinates = (float(event['x']), float(event['y']))
            
            # Extract features specific to action type
            features = self._extract_node_features(event, action_type)
            
            return BehavioralNode(
                node_id=node_id,
                action_type=action_type,
                timestamp=datetime.fromtimestamp(event.get('timestamp', 0) / 1000),
                duration=float(event.get('duration', 0)),
                coordinates=coordinates,
                pressure=event.get('pressure'),
                screen_id=event.get('screen_id'),
                sequence_position=sequence_pos,
                features=features
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating node from event: {e}")
            return None
    
    def _extract_node_features(self, event: Dict[str, Any], action_type: ActionType) -> Dict[str, Any]:
        """Extract action-specific features for node"""
        features = {}
        
        if action_type == ActionType.TAP:
            features.update({
                'pressure': event.get('pressure', 0),
                'touch_area': event.get('touch_area', 0),
                'is_double_tap': event.get('is_double_tap', False)
            })
        
        elif action_type == ActionType.SCROLL:
            features.update({
                'velocity': event.get('velocity', 0),
                'direction': event.get('direction', 'unknown'),
                'distance': event.get('distance', 0),
                'friction': event.get('friction', 0)
            })
        
        elif action_type == ActionType.TYPING:
            features.update({
                'key_count': event.get('key_count', 0),
                'typing_speed': event.get('typing_speed', 0),
                'dwell_time': event.get('dwell_time', 0),
                'flight_time': event.get('flight_time', 0)
            })
        
        elif action_type == ActionType.TRANSACTION_START:
            features.update({
                'amount': event.get('amount', 0),
                'beneficiary_type': event.get('beneficiary_type', 'unknown'),
                'transaction_type': event.get('transaction_type', 'unknown')
            })
        
        return features
    
    def _create_edge_between_nodes(self, node1: BehavioralNode, node2: BehavioralNode) -> Optional[BehavioralEdge]:
        """Create edge between two consecutive nodes"""
        try:
            self.edge_counter += 1
            edge_id = f"edge_{self.edge_counter:06d}"
            
            # Calculate time gap
            time_gap = (node2.timestamp - node1.timestamp).total_seconds() * 1000
            
            # Determine transition type based on timing
            if time_gap < self.rapid_threshold:
                transition_type = TransitionType.RAPID
            elif time_gap > self.delayed_threshold:
                transition_type = TransitionType.DELAYED
            else:
                transition_type = TransitionType.SEQUENTIAL
            
            # Calculate spatial distance if both nodes have coordinates
            spatial_distance = None
            if node1.coordinates and node2.coordinates:
                dx = node2.coordinates[0] - node1.coordinates[0]
                dy = node2.coordinates[1] - node1.coordinates[1]
                spatial_distance = np.sqrt(dx*dx + dy*dy)
            
            # Extract edge features
            features = {
                'action_sequence': f"{node1.action_type.value}->{node2.action_type.value}",
                'duration_ratio': node2.duration / max(node1.duration, 1),
                'same_screen': node1.screen_id == node2.screen_id if node1.screen_id and node2.screen_id else None
            }
            
            return BehavioralEdge(
                edge_id=edge_id,
                source_node=node1.node_id,
                target_node=node2.node_id,
                transition_type=transition_type,
                time_gap=time_gap,
                spatial_distance=spatial_distance,
                confidence=1.0,
                features=features
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating edge: {e}")
            return None
    
    def _find_contextual_edges(self, nodes: List[BehavioralNode]) -> List[BehavioralEdge]:
        """Find non-sequential contextual relationships between nodes"""
        contextual_edges = []
        
        try:
            # Look for transaction-related patterns
            transaction_nodes = [n for n in nodes if n.action_type in [
                ActionType.TRANSACTION_START, ActionType.TRANSACTION_COMPLETE, 
                ActionType.MPIN_ENTRY, ActionType.BIOMETRIC_AUTH
            ]]
            
            # Connect transaction events with contextual edges
            for i, start_node in enumerate(transaction_nodes[:-1]):
                for end_node in transaction_nodes[i+1:]:
                    # Don't create contextual edge if they're already sequential
                    if abs(start_node.sequence_position - end_node.sequence_position) <= 1:
                        continue
                    
                    self.edge_counter += 1
                    edge = BehavioralEdge(
                        edge_id=f"ctx_edge_{self.edge_counter:06d}",
                        source_node=start_node.node_id,
                        target_node=end_node.node_id,
                        transition_type=TransitionType.CONTEXTUAL,
                        time_gap=(end_node.timestamp - start_node.timestamp).total_seconds() * 1000,
                        confidence=0.8,
                        features={'context': 'transaction_flow'}
                    )
                    contextual_edges.append(edge)
            
            # Look for repeated action patterns
            action_patterns = self._find_repeated_patterns(nodes)
            for pattern in action_patterns:
                if len(pattern) >= 2:
                    for i in range(len(pattern) - 1):
                        if abs(pattern[i].sequence_position - pattern[i+1].sequence_position) > 1:
                            self.edge_counter += 1
                            edge = BehavioralEdge(
                                edge_id=f"pattern_edge_{self.edge_counter:06d}",
                                source_node=pattern[i].node_id,
                                target_node=pattern[i+1].node_id,
                                transition_type=TransitionType.CONTEXTUAL,
                                time_gap=(pattern[i+1].timestamp - pattern[i].timestamp).total_seconds() * 1000,
                                confidence=0.6,
                                features={'context': 'repeated_pattern'}
                            )
                            contextual_edges.append(edge)
            
        except Exception as e:
            self.logger.warning(f"Error finding contextual edges: {e}")
        
        return contextual_edges
    
    def _find_repeated_patterns(self, nodes: List[BehavioralNode]) -> List[List[BehavioralNode]]:
        """Find repeated behavioral patterns in the session"""
        patterns = []
        
        # Group nodes by action type
        action_groups = {}
        for node in nodes:
            if node.action_type not in action_groups:
                action_groups[node.action_type] = []
            action_groups[node.action_type].append(node)
        
        # Find patterns with 3+ repetitions
        for action_type, action_nodes in action_groups.items():
            if len(action_nodes) >= 3:
                patterns.append(action_nodes)
        
        return patterns
    
    def _calculate_graph_metrics(self, graph: SessionGraph) -> Dict[str, Any]:
        """Calculate graph-level metrics for analysis"""
        try:
            nx_graph = graph.graph
            
            if len(nx_graph.nodes) == 0:
                return {}
            
            metrics = {
                'node_count': len(nx_graph.nodes),
                'edge_count': len(nx_graph.edges),
                'density': nx.density(nx_graph),
                'avg_clustering': nx.average_clustering(nx_graph.to_undirected()),
                'is_connected': nx.is_weakly_connected(nx_graph),
                'longest_path': len(nx.dag_longest_path(nx_graph)) if nx.is_directed_acyclic_graph(nx_graph) else 0
            }
            
            # Action type distribution
            action_types = [data['action_type'] for _, data in nx_graph.nodes(data=True)]
            metrics['action_distribution'] = {
                action: action_types.count(action) for action in set(action_types)
            }
            
            # Transition type distribution
            transition_types = [data['transition_type'] for _, _, data in nx_graph.edges(data=True)]
            metrics['transition_distribution'] = {
                transition: transition_types.count(transition) for transition in set(transition_types)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating graph metrics: {e}")
            return {}
    
    def get_graph_features_vector(self, graph: SessionGraph) -> np.ndarray:
        """
        Convert session graph to feature vector for GNN analysis
        
        Returns:
            numpy array of graph-level features for ML analysis
        """
        try:
            features = []
            
            # Basic graph statistics
            features.extend([
                len(graph.nodes),
                len(graph.edges),
                len(graph.nodes) / max(len(graph.edges), 1),  # node/edge ratio
            ])
            
            # Action type frequencies (normalized)
            action_counts = np.zeros(len(ActionType))
            for node in graph.nodes.values():
                action_counts[list(ActionType).index(node.action_type)] += 1
            action_frequencies = action_counts / max(len(graph.nodes), 1)
            features.extend(action_frequencies.tolist())
            
            # Transition type frequencies
            transition_counts = np.zeros(len(TransitionType))
            for edge in graph.edges.values():
                transition_counts[list(TransitionType).index(edge.transition_type)] += 1
            transition_frequencies = transition_counts / max(len(graph.edges), 1)
            features.extend(transition_frequencies.tolist())
            
            # Temporal features
            if graph.nodes:
                durations = [node.duration for node in graph.nodes.values()]
                time_gaps = [edge.time_gap for edge in graph.edges.values()]
                
                features.extend([
                    np.mean(durations),
                    np.std(durations),
                    np.mean(time_gaps) if time_gaps else 0,
                    np.std(time_gaps) if time_gaps else 0
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Graph connectivity features
            if graph.graph.nodes:
                try:
                    features.extend([
                        nx.density(graph.graph),
                        nx.average_clustering(graph.graph.to_undirected()),
                        1 if nx.is_weakly_connected(graph.graph) else 0
                    ])
                except:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error creating graph feature vector: {e}")
            # Return zero vector on error
            vector_size = 3 + len(ActionType) + len(TransitionType) + 7
            return np.zeros(vector_size, dtype=np.float32)
    
    def export_graph_for_gnn(self, graph: SessionGraph) -> Dict[str, Any]:
        """
        Export session graph in format suitable for GNN analysis
        
        Returns:
            Dictionary containing node features, edge indices, and edge attributes
        """
        try:
            # Node features matrix
            node_features = []
            node_mapping = {}
            
            for i, (node_id, node) in enumerate(graph.nodes.items()):
                node_mapping[node_id] = i
                
                # Create node feature vector
                node_feature = [
                    list(ActionType).index(node.action_type),  # Action type
                    node.duration,
                    node.sequence_position,
                    node.coordinates[0] if node.coordinates else 0,
                    node.coordinates[1] if node.coordinates else 0,
                    node.pressure or 0
                ]
                
                # Add action-specific features
                for key in ['velocity', 'pressure', 'touch_area', 'typing_speed']:
                    node_feature.append(node.features.get(key, 0))
                
                node_features.append(node_feature)
            
            # Edge indices and attributes
            edge_indices = []
            edge_attributes = []
            
            for edge in graph.edges.values():
                if edge.source_node in node_mapping and edge.target_node in node_mapping:
                    edge_indices.append([
                        node_mapping[edge.source_node],
                        node_mapping[edge.target_node]
                    ])
                    
                    edge_attr = [
                        list(TransitionType).index(edge.transition_type),
                        edge.time_gap,
                        edge.spatial_distance or 0,
                        edge.confidence
                    ]
                    edge_attributes.append(edge_attr)
            
            return {
                'node_features': np.array(node_features, dtype=np.float32),
                'edge_indices': np.array(edge_indices).T if edge_indices else np.empty((2, 0)),
                'edge_attributes': np.array(edge_attributes, dtype=np.float32),
                'num_nodes': len(node_features),
                'num_edges': len(edge_indices),
                'session_id': graph.session_id,
                'user_id': graph.user_id
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting graph for GNN: {e}")
            return {
                'node_features': np.empty((0, 10)),
                'edge_indices': np.empty((2, 0)),
                'edge_attributes': np.empty((0, 4)),
                'num_nodes': 0,
                'num_edges': 0,
                'session_id': graph.session_id,
                'user_id': graph.user_id
            }
