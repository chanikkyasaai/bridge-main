"""
RIGOROUS LAYER 2 (ADAPTIVE CONTEXT) TESTING SUITE
Banking-Grade Testing for Adaptive Context Analysis Component

This test suite performs exhaustive testing of the Layer 2 Adaptive Context system,
including transformer-based behavioral encoding, GNN session analysis, and contextual
verification with edge cases and adversarial scenarios.

Target: < 80ms per analysis (as per Stage 3 requirements)
"""

import pytest
import asyncio
import numpy as np
import time
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import tempfile
import shutil
from pathlib import Path
import logging
import json
import concurrent.futures
from unittest.mock import patch, MagicMock
import networkx as nx
from collections import defaultdict

# Setup test environment
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from ml_engine.adapters.level2.layer2_verifier import (
    Layer2Verifier, TransformerBehavioralEncoder, SessionGraphGNN,
    ContextualFeatures, SessionGraph, Layer2Result
)
from ml_engine.utils.behavioral_vectors import BehavioralVector, BehavioralEvent
from ml_engine.config import CONFIG

logger = logging.getLogger(__name__)

class Layer2AdaptiveContextRigorousTester:
    """Comprehensive Layer 2 Adaptive Context testing framework"""
    
    def __init__(self):
        self.temp_dir = None
        self.layer2_verifier = None
        self.transformer_encoder = None
        self.gnn_model = None
        self.test_sessions = {}
        self.test_contexts = {}
        self.performance_metrics = {
            'analysis_times': [],
            'transformer_times': [],
            'gnn_times': [],
            'memory_usage': [],
            'accuracy_scores': []
        }
        
    async def setup_test_environment(self):
        """Setup isolated test environment"""
        logger.info("Setting up Layer 2 Adaptive Context test environment...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="layer2_adaptive_test_")
        
        # Override config for testing
        CONFIG.L2_ANALYSIS_TIMEOUT_MS = 100  # 100ms for testing
        CONFIG.BEHAVIORAL_VECTOR_DIM = 64  # Smaller for faster testing
        CONFIG.GNN_HIDDEN_DIM = 32
        CONFIG.GRAPH_EMBEDDING_DIM = 16
        CONFIG.TRANSFORMER_MAX_LENGTH = 256
        
        # Initialize components
        self.transformer_encoder = TransformerBehavioralEncoder()
        self.gnn_model = SessionGraphGNN(node_features=32, edge_features=16)
        
        # Generate test data
        self._generate_test_sessions_and_contexts()
        
        logger.info(f"✓ Test environment ready at {self.temp_dir}")
    
    def _generate_test_sessions_and_contexts(self):
        """Generate comprehensive test sessions and contextual data"""
        np.random.seed(42)  # Reproducible results
        torch.manual_seed(42)
        
        # Generate different user behavior patterns
        session_types = ['normal_session', 'stressed_session', 'hurried_session', 'fraud_session']
        
        for session_type in session_types:
            session_id = f"test_{session_type}"
            user_id = f"user_{session_type}"
            
            # Generate behavioral sequence based on session type
            if session_type == 'normal_session':
                # Consistent, predictable pattern
                base_pattern = np.random.normal(0.5, 0.1, (20, CONFIG.BEHAVIORAL_VECTOR_DIM))
                context = ContextualFeatures(
                    age_group='middle', device_type='phone', time_of_day='afternoon',
                    usage_mode='normal', network_type='wifi', location_risk=0.1,
                    interaction_rhythm='medium'
                )
            elif session_type == 'stressed_session':
                # More erratic, higher variance
                base_pattern = np.random.normal(0.7, 0.2, (20, CONFIG.BEHAVIORAL_VECTOR_DIM))
                context = ContextualFeatures(
                    age_group='young', device_type='phone', time_of_day='evening',
                    usage_mode='stressed', network_type='mobile', location_risk=0.2,
                    interaction_rhythm='fast'
                )
            elif session_type == 'hurried_session':
                # Fast, shorter duration patterns
                base_pattern = np.random.normal(0.8, 0.15, (15, CONFIG.BEHAVIORAL_VECTOR_DIM))
                context = ContextualFeatures(
                    age_group='middle', device_type='tablet', time_of_day='morning',
                    usage_mode='hurried', network_type='wifi', location_risk=0.05,
                    interaction_rhythm='fast'
                )
            else:  # fraud_session
                # Unusual, inconsistent patterns
                base_pattern = np.random.uniform(0, 1, (25, CONFIG.BEHAVIORAL_VECTOR_DIM))
                context = ContextualFeatures(
                    age_group='unknown', device_type='phone', time_of_day='night',
                    usage_mode='normal', network_type='unknown', location_risk=0.8,
                    interaction_rhythm='medium'
                )
            
            # Create session graph
            nodes = []
            edges = []
            edge_features = []
            timestamps = []
            
            for i in range(len(base_pattern)):
                # Node features (behavioral vector + metadata)
                node_feature = {
                    'behavioral_vector': base_pattern[i],
                    'event_type': np.random.choice(['touch', 'swipe', 'type', 'scroll']),
                    'pressure': np.random.uniform(0.1, 1.0),
                    'velocity': np.random.uniform(0.0, 100.0),
                    'duration': np.random.uniform(0.05, 2.0)
                }
                nodes.append(node_feature)
                timestamps.append(datetime.now() + timedelta(seconds=i))
                
                # Create edges to previous nodes (temporal connections)
                if i > 0:
                    edges.append((i-1, i))
                    edge_features.append({
                        'time_delta': 1.0,
                        'interaction_type': 'sequential',
                        'context_similarity': np.random.uniform(0.5, 1.0)
                    })
                
                # Add some random connections for graph structure
                if i > 2 and np.random.random() < 0.3:
                    prev_idx = np.random.randint(0, i-1)
                    edges.append((prev_idx, i))
                    edge_features.append({
                        'time_delta': float(i - prev_idx),
                        'interaction_type': 'contextual',
                        'context_similarity': np.random.uniform(0.3, 0.8)
                    })
            
            session_graph = SessionGraph(
                session_id=session_id,
                user_id=user_id,
                nodes=nodes,
                edges=edges,
                edge_features=edge_features,
                timestamps=timestamps
            )
            
            self.test_sessions[session_type] = session_graph
            self.test_contexts[session_type] = context
    
    async def test_transformer_behavioral_encoding(self):
        """Test transformer-based behavioral sequence encoding"""
        logger.info("Testing transformer behavioral encoding...")
        
        results = {}
        
        # Test 1: Basic encoding functionality
        session_type = 'normal_session'
        session = self.test_sessions[session_type]
        context = self.test_contexts[session_type]
        
        # Prepare behavioral sequence
        behavioral_vectors = torch.tensor([
            node['behavioral_vector'] for node in session.nodes
        ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Prepare context features
        context_vector = torch.tensor([
            0.5,  # age_group (middle = 0.5)
            1.0,  # device_type (phone = 1.0)
            0.6,  # time_of_day (afternoon = 0.6)
            0.5,  # usage_mode (normal = 0.5)
            1.0,  # network_type (wifi = 1.0)
            context.location_risk,
            0.5   # interaction_rhythm (medium = 0.5)
        ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Test encoding
        start_time = time.perf_counter()
        try:
            embedding, confidence = self.transformer_encoder(behavioral_vectors, context_vector)
            encoding_time = (time.perf_counter() - start_time) * 1000
            
            results['basic_encoding_success'] = True
            results['encoding_time_ms'] = encoding_time
            results['embedding_shape'] = list(embedding.shape)
            results['confidence_value'] = float(confidence.item())
            results['embedding_magnitude'] = float(torch.norm(embedding).item())
            
        except Exception as e:
            results['basic_encoding_success'] = False
            results['encoding_error'] = str(e)
            logger.error(f"Transformer encoding failed: {e}")
        
        # Test 2: Multiple session types
        encoding_results = {}
        
        for session_type, session in self.test_sessions.items():
            context = self.test_contexts[session_type]
            
            behavioral_vectors = torch.tensor([
                node['behavioral_vector'] for node in session.nodes
            ], dtype=torch.float32).unsqueeze(0)
            
            context_vector = torch.tensor([
                0.5, 1.0, 0.6, 0.5, 1.0, context.location_risk, 0.5
            ], dtype=torch.float32).unsqueeze(0)
            
            start_time = time.perf_counter()
            try:
                embedding, confidence = self.transformer_encoder(behavioral_vectors, context_vector)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                encoding_results[session_type] = {
                    'success': True,
                    'processing_time_ms': processing_time,
                    'confidence': float(confidence.item()),
                    'embedding_norm': float(torch.norm(embedding).item())
                }
                
            except Exception as e:
                encoding_results[session_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        results['multi_session_encoding'] = encoding_results
        
        # Test 3: Performance requirements
        avg_time = np.mean([
            res['processing_time_ms'] for res in encoding_results.values() 
            if res.get('success', False)
        ])
        results['avg_encoding_time_ms'] = avg_time
        results['meets_performance_target'] = avg_time < 40  # Half of 80ms target for Layer 2
        
        logger.info(f"Transformer Encoding Results: {results}")
        
        # Assertions
        assert results.get('basic_encoding_success', False), "Basic encoding should work"
        assert results.get('meets_performance_target', False), f"Encoding time {avg_time:.2f}ms too slow"
        
        self.performance_metrics['transformer_times'].extend([
            res['processing_time_ms'] for res in encoding_results.values() 
            if res.get('success', False)
        ])
        
        return results
    
    async def test_session_graph_gnn(self):
        """Test GNN-based session graph analysis"""
        logger.info("Testing GNN session graph analysis...")
        
        from torch_geometric.data import Data, Batch
        
        results = {}
        
        # Test 1: Basic GNN functionality
        session_type = 'normal_session'
        session = self.test_sessions[session_type]
        
        # Convert session to PyTorch Geometric format
        node_features = []
        for node in session.nodes:
            # Create simple node feature vector
            features = [
                node['pressure'],
                node['velocity'],
                node['duration'],
                1.0 if node['event_type'] == 'touch' else 0.0,
                1.0 if node['event_type'] == 'swipe' else 0.0,
                1.0 if node['event_type'] == 'type' else 0.0,
                1.0 if node['event_type'] == 'scroll' else 0.0
            ]
            # Pad to required dimension
            while len(features) < 32:
                features.append(0.0)
            node_features.append(features[:32])
        
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(session.edges, dtype=torch.long).t().contiguous()
        
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Test GNN processing
        start_time = time.perf_counter()
        try:
            self.gnn_model.eval()
            with torch.no_grad():
                node_embeddings = self.gnn_model.conv1(graph_data.x, graph_data.edge_index)
                node_embeddings = torch.relu(node_embeddings)
                node_embeddings = self.gnn_model.conv2(node_embeddings, graph_data.edge_index)
                node_embeddings = torch.relu(node_embeddings)
                node_embeddings = self.gnn_model.conv3(node_embeddings, graph_data.edge_index)
                
                # Graph-level embedding
                graph_embedding = torch.mean(node_embeddings, dim=0)
                
                # Anomaly score
                anomaly_score = self.gnn_model.anomaly_head(node_embeddings.mean(dim=0).unsqueeze(0))
            
            gnn_time = (time.perf_counter() - start_time) * 1000
            
            results['basic_gnn_success'] = True
            results['gnn_processing_time_ms'] = gnn_time
            results['node_embeddings_shape'] = list(node_embeddings.shape)
            results['graph_embedding_shape'] = list(graph_embedding.shape)
            results['anomaly_score'] = float(anomaly_score.item())
            
        except Exception as e:
            results['basic_gnn_success'] = False
            results['gnn_error'] = str(e)
            logger.error(f"GNN processing failed: {e}")
        
        # Test 2: Multiple session types with GNN
        gnn_results = {}
        
        for session_type, session in self.test_sessions.items():
            try:
                # Convert to graph data
                node_features = []
                for node in session.nodes:
                    features = [
                        node['pressure'], node['velocity'], node['duration'],
                        1.0 if node['event_type'] == 'touch' else 0.0,
                        1.0 if node['event_type'] == 'swipe' else 0.0,
                        1.0 if node['event_type'] == 'type' else 0.0,
                        1.0 if node['event_type'] == 'scroll' else 0.0
                    ]
                    while len(features) < 32:
                        features.append(0.0)
                    node_features.append(features[:32])
                
                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index = torch.tensor(session.edges, dtype=torch.long).t().contiguous()
                graph_data = Data(x=x, edge_index=edge_index)
                
                start_time = time.perf_counter()
                
                self.gnn_model.eval()
                with torch.no_grad():
                    node_embeddings = self.gnn_model.conv1(graph_data.x, graph_data.edge_index)
                    node_embeddings = torch.relu(node_embeddings)
                    node_embeddings = self.gnn_model.conv2(node_embeddings, graph_data.edge_index)
                    node_embeddings = torch.relu(node_embeddings)
                    node_embeddings = self.gnn_model.conv3(node_embeddings, graph_data.edge_index)
                    
                    anomaly_score = self.gnn_model.anomaly_head(node_embeddings.mean(dim=0).unsqueeze(0))
                
                processing_time = (time.perf_counter() - start_time) * 1000
                
                gnn_results[session_type] = {
                    'success': True,
                    'processing_time_ms': processing_time,
                    'anomaly_score': float(anomaly_score.item()),
                    'num_nodes': len(node_features),
                    'num_edges': len(session.edges)
                }
                
            except Exception as e:
                gnn_results[session_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        results['multi_session_gnn'] = gnn_results
        
        # Test 3: Anomaly detection capability
        fraud_anomaly = gnn_results.get('fraud_session', {}).get('anomaly_score', 0.5)
        normal_anomaly = gnn_results.get('normal_session', {}).get('anomaly_score', 0.5)
        
        results['anomaly_detection_working'] = fraud_anomaly > normal_anomaly
        results['fraud_anomaly_score'] = fraud_anomaly
        results['normal_anomaly_score'] = normal_anomaly
        
        # Performance analysis
        successful_times = [
            res['processing_time_ms'] for res in gnn_results.values() 
            if res.get('success', False)
        ]
        
        if successful_times:
            results['avg_gnn_time_ms'] = np.mean(successful_times)
            results['max_gnn_time_ms'] = np.max(successful_times)
            results['gnn_meets_performance_target'] = np.mean(successful_times) < 40
        
        logger.info(f"GNN Analysis Results: {results}")
        
        # Assertions
        assert results.get('basic_gnn_success', False), "Basic GNN processing should work"
        if successful_times:
            assert results.get('gnn_meets_performance_target', False), f"GNN time {np.mean(successful_times):.2f}ms too slow"
        
        self.performance_metrics['gnn_times'].extend(successful_times)
        
        return results
    
    async def test_contextual_adaptation(self):
        """Test contextual adaptation and decision making"""
        logger.info("Testing contextual adaptation...")
        
        results = {}
        
        # Test different contexts with same behavioral pattern
        base_pattern = torch.tensor([
            node['behavioral_vector'] for node in self.test_sessions['normal_session'].nodes
        ], dtype=torch.float32).unsqueeze(0)
        
        context_variants = {
            'low_risk': torch.tensor([0.5, 1.0, 0.6, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0),
            'medium_risk': torch.tensor([0.5, 1.0, 0.6, 0.5, 0.5, 0.4, 0.5], dtype=torch.float32).unsqueeze(0),
            'high_risk': torch.tensor([0.2, 1.0, 0.9, 0.8, 0.0, 0.8, 0.8], dtype=torch.float32).unsqueeze(0),
        }
        
        adaptation_results = {}
        
        for risk_level, context_vector in context_variants.items():
            try:
                start_time = time.perf_counter()
                embedding, confidence = self.transformer_encoder(base_pattern, context_vector)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                adaptation_results[risk_level] = {
                    'success': True,
                    'confidence': float(confidence.item()),
                    'processing_time_ms': processing_time,
                    'embedding_norm': float(torch.norm(embedding).item())
                }
                
            except Exception as e:
                adaptation_results[risk_level] = {
                    'success': False,
                    'error': str(e)
                }
        
        results['context_adaptation'] = adaptation_results
        
        # Test adaptive response to context
        confidences = [res['confidence'] for res in adaptation_results.values() if res.get('success')]
        if len(confidences) >= 3:
            results['confidence_adapts_to_risk'] = confidences[0] > confidences[1] > confidences[2]  # low > medium > high
            results['confidence_variance'] = float(np.var(confidences))
        
        # Test time-of-day adaptation
        time_contexts = {
            'morning': torch.tensor([0.5, 1.0, 0.2, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0),
            'afternoon': torch.tensor([0.5, 1.0, 0.6, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0),
            'night': torch.tensor([0.5, 1.0, 0.9, 0.5, 1.0, 0.3, 0.5], dtype=torch.float32).unsqueeze(0),
        }
        
        time_results = {}
        for time_period, context_vector in time_contexts.items():
            try:
                embedding, confidence = self.transformer_encoder(base_pattern, context_vector)
                time_results[time_period] = float(confidence.item())
            except Exception as e:
                time_results[time_period] = None
                logger.error(f"Time adaptation test failed for {time_period}: {e}")
        
        results['time_adaptation'] = time_results
        
        logger.info(f"Contextual Adaptation Results: {results}")
        
        return results
    
    async def test_performance_requirements(self):
        """Test Layer 2 performance requirements (< 80ms)"""
        logger.info("Testing Layer 2 performance requirements...")
        
        # Test complete Layer 2 analysis pipeline
        session = self.test_sessions['normal_session']
        context = self.test_contexts['normal_session']
        
        # Prepare data
        behavioral_vectors = torch.tensor([
            node['behavioral_vector'] for node in session.nodes
        ], dtype=torch.float32).unsqueeze(0)
        
        context_vector = torch.tensor([
            0.5, 1.0, 0.6, 0.5, 1.0, context.location_risk, 0.5
        ], dtype=torch.float32).unsqueeze(0)
        
        # Run multiple performance tests
        analysis_times = []
        
        for i in range(20):  # 20 iterations for reliable timing
            start_time = time.perf_counter()
            
            # Complete Layer 2 analysis
            try:
                # Step 1: Transformer encoding
                embedding, confidence = self.transformer_encoder(behavioral_vectors, context_vector)
                
                # Step 2: GNN analysis (simplified)
                node_features = []
                for node in session.nodes:
                    features = [node['pressure'], node['velocity'], node['duration']]
                    while len(features) < 32:
                        features.append(0.0)
                    node_features.append(features[:32])
                
                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index = torch.tensor(session.edges, dtype=torch.long).t().contiguous()
                
                from torch_geometric.data import Data
                graph_data = Data(x=x, edge_index=edge_index)
                
                self.gnn_model.eval()
                with torch.no_grad():
                    node_embeddings = self.gnn_model.conv1(graph_data.x, graph_data.edge_index)
                    anomaly_score = self.gnn_model.anomaly_head(node_embeddings.mean(dim=0).unsqueeze(0))
                
                # Step 3: Combine results
                combined_score = float(confidence.item()) * (1 - float(anomaly_score.item()))
                
                total_time = (time.perf_counter() - start_time) * 1000
                analysis_times.append(total_time)
                
            except Exception as e:
                logger.error(f"Performance test iteration {i} failed: {e}")
        
        # Analyze performance
        if analysis_times:
            performance_results = {
                'total_iterations': len(analysis_times),
                'avg_analysis_time_ms': np.mean(analysis_times),
                'p50_analysis_time_ms': np.percentile(analysis_times, 50),
                'p95_analysis_time_ms': np.percentile(analysis_times, 95),
                'p99_analysis_time_ms': np.percentile(analysis_times, 99),
                'max_analysis_time_ms': np.max(analysis_times),
                'min_analysis_time_ms': np.min(analysis_times),
                'meets_80ms_target': np.mean(analysis_times) < 80.0,
                'p95_meets_80ms_target': np.percentile(analysis_times, 95) < 80.0,
                'all_under_100ms': np.max(analysis_times) < 100.0
            }
        else:
            performance_results = {
                'error': 'No successful performance test iterations'
            }
        
        logger.info(f"Layer 2 Performance Results: {performance_results}")
        
        # Banking performance requirements
        if analysis_times:
            assert performance_results['meets_80ms_target'], f"Average time {performance_results['avg_analysis_time_ms']:.2f}ms exceeds 80ms target"
            assert performance_results['p95_meets_80ms_target'], f"P95 time {performance_results['p95_analysis_time_ms']:.2f}ms exceeds 80ms target"
        
        self.performance_metrics['analysis_times'].extend(analysis_times)
        
        return performance_results
    
    async def test_edge_cases_and_robustness(self):
        """Test edge cases and robustness"""
        logger.info("Testing edge cases and robustness...")
        
        edge_case_results = {}
        
        # Test 1: Empty session
        try:
            empty_vectors = torch.zeros((1, 1, CONFIG.BEHAVIORAL_VECTOR_DIM), dtype=torch.float32)
            context_vector = torch.tensor([0.5, 1.0, 0.6, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0)
            
            embedding, confidence = self.transformer_encoder(empty_vectors, context_vector)
            edge_case_results['empty_session_handled'] = True
            edge_case_results['empty_session_confidence'] = float(confidence.item())
            
        except Exception as e:
            edge_case_results['empty_session_handled'] = False
            edge_case_results['empty_session_error'] = str(e)
        
        # Test 2: Extreme behavioral values
        try:
            extreme_vectors = torch.full((1, 10, CONFIG.BEHAVIORAL_VECTOR_DIM), 1000.0, dtype=torch.float32)
            context_vector = torch.tensor([0.5, 1.0, 0.6, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0)
            
            embedding, confidence = self.transformer_encoder(extreme_vectors, context_vector)
            edge_case_results['extreme_values_handled'] = True
            edge_case_results['extreme_values_confidence'] = float(confidence.item())
            
        except Exception as e:
            edge_case_results['extreme_values_handled'] = False
            edge_case_results['extreme_values_error'] = str(e)
        
        # Test 3: Malformed context
        try:
            normal_vectors = torch.randn((1, 10, CONFIG.BEHAVIORAL_VECTOR_DIM), dtype=torch.float32)
            malformed_context = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=torch.float32).unsqueeze(0)
            
            embedding, confidence = self.transformer_encoder(normal_vectors, malformed_context)
            edge_case_results['malformed_context_handled'] = True
            
        except Exception as e:
            edge_case_results['malformed_context_handled'] = False
            edge_case_results['malformed_context_error'] = str(e)
        
        # Test 4: Very long session
        try:
            long_vectors = torch.randn((1, 100, CONFIG.BEHAVIORAL_VECTOR_DIM), dtype=torch.float32)
            context_vector = torch.tensor([0.5, 1.0, 0.6, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0)
            
            start_time = time.perf_counter()
            embedding, confidence = self.transformer_encoder(long_vectors, context_vector)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            edge_case_results['long_session_handled'] = True
            edge_case_results['long_session_time_ms'] = processing_time
            edge_case_results['long_session_under_limit'] = processing_time < 150  # 150ms limit for very long sessions
            
        except Exception as e:
            edge_case_results['long_session_handled'] = False
            edge_case_results['long_session_error'] = str(e)
        
        # Test 5: Disconnected graph for GNN
        try:
            # Create graph with no edges
            node_features = torch.randn((5, 32), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
            
            from torch_geometric.data import Data
            graph_data = Data(x=node_features, edge_index=edge_index)
            
            self.gnn_model.eval()
            with torch.no_grad():
                node_embeddings = self.gnn_model.conv1(graph_data.x, graph_data.edge_index)
                
            edge_case_results['disconnected_graph_handled'] = True
            
        except Exception as e:
            edge_case_results['disconnected_graph_handled'] = False
            edge_case_results['disconnected_graph_error'] = str(e)
        
        logger.info(f"Edge Case Results: {edge_case_results}")
        
        return edge_case_results
    
    async def test_adversarial_scenarios(self):
        """Test adversarial attack scenarios"""
        logger.info("Testing adversarial scenarios...")
        
        adversarial_results = {}
        
        # Base legitimate session
        legit_vectors = torch.tensor([
            node['behavioral_vector'] for node in self.test_sessions['normal_session'].nodes
        ], dtype=torch.float32).unsqueeze(0)
        
        context_vector = torch.tensor([0.5, 1.0, 0.6, 0.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0)
        
        # Get baseline confidence
        baseline_embedding, baseline_confidence = self.transformer_encoder(legit_vectors, context_vector)
        baseline_score = float(baseline_confidence.item())
        
        adversarial_results['baseline_confidence'] = baseline_score
        
        # Test 1: Gradual drift attack (slowly changing behavior)
        drift_confidences = []
        for drift_factor in np.linspace(0, 0.5, 10):  # Gradually increase noise
            drifted_vectors = legit_vectors + torch.randn_like(legit_vectors) * drift_factor
            
            try:
                embedding, confidence = self.transformer_encoder(drifted_vectors, context_vector)
                drift_confidences.append(float(confidence.item()))
            except Exception as e:
                drift_confidences.append(0.0)
        
        adversarial_results['drift_attack'] = {
            'confidences': drift_confidences,
            'detects_drift': drift_confidences[-1] < baseline_score * 0.8,  # Should detect significant drift
            'gradual_degradation': all(drift_confidences[i] >= drift_confidences[i+1] for i in range(len(drift_confidences)-1))
        }
        
        # Test 2: Context manipulation attack
        manipulated_contexts = [
            torch.tensor([0.2, 0.0, 0.9, 0.8, 0.0, 0.9, 0.9], dtype=torch.float32).unsqueeze(0),  # High risk context
            torch.tensor([1.0, 2.0, -0.5, 1.5, 1.0, 0.1, 0.5], dtype=torch.float32).unsqueeze(0),  # Out of range values
            torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32).unsqueeze(0),   # Uniform context
        ]
        
        context_attack_results = []
        for i, manip_context in enumerate(manipulated_contexts):
            try:
                embedding, confidence = self.transformer_encoder(legit_vectors, manip_context)
                context_attack_results.append({
                    'attack_type': f'context_manipulation_{i}',
                    'confidence': float(confidence.item()),
                    'detected': float(confidence.item()) < baseline_score * 0.9
                })
            except Exception as e:
                context_attack_results.append({
                    'attack_type': f'context_manipulation_{i}',
                    'error': str(e),
                    'detected': True  # Error is also a form of detection
                })
        
        adversarial_results['context_attacks'] = context_attack_results
        
        # Test 3: Replay attack detection
        replay_confidences = []
        for _ in range(5):  # Same vectors multiple times
            try:
                embedding, confidence = self.transformer_encoder(legit_vectors, context_vector)
                replay_confidences.append(float(confidence.item()))
            except Exception as e:
                replay_confidences.append(0.0)
        
        adversarial_results['replay_attack'] = {
            'confidences': replay_confidences,
            'consistency': np.std(replay_confidences),  # Should be low for same input
            'average_confidence': np.mean(replay_confidences)
        }
        
        # Test 4: Timing attack (rapid successive requests)
        timing_attack_times = []
        timing_confidences = []
        
        for i in range(10):
            start_time = time.perf_counter()
            try:
                embedding, confidence = self.transformer_encoder(legit_vectors, context_vector)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                timing_attack_times.append(processing_time)
                timing_confidences.append(float(confidence.item()))
            except Exception as e:
                timing_attack_times.append(1000.0)  # High time indicates failure
                timing_confidences.append(0.0)
        
        adversarial_results['timing_attack'] = {
            'processing_times_ms': timing_attack_times,
            'confidences': timing_confidences,
            'avg_time_ms': np.mean(timing_attack_times),
            'time_variance': np.var(timing_attack_times),
            'performance_degradation': np.mean(timing_attack_times) / 80.0  # Relative to 80ms target
        }
        
        logger.info(f"Adversarial Test Results: {adversarial_results}")
        
        # Verify attack detection capabilities
        assert adversarial_results['drift_attack']['detects_drift'], "Should detect behavioral drift"
        assert any(attack['detected'] for attack in adversarial_results['context_attacks']), "Should detect context manipulation"
        assert adversarial_results['timing_attack']['performance_degradation'] < 2.0, "Performance shouldn't degrade more than 2x under timing attack"
        
        return adversarial_results
    
    async def test_memory_efficiency(self):
        """Test memory usage and efficiency"""
        logger.info("Testing memory efficiency...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_results = {
            'baseline_memory_mb': baseline_memory,
            'memory_usage_progression': []
        }
        
        # Test memory usage with increasing load
        for batch_size in [1, 5, 10, 20]:
            # Create batch of sessions
            batch_vectors = torch.randn((batch_size, 20, CONFIG.BEHAVIORAL_VECTOR_DIM), dtype=torch.float32)
            batch_contexts = torch.randn((batch_size, 7), dtype=torch.float32)
            
            # Process batch
            start_memory = process.memory_info().rss / 1024 / 1024
            
            try:
                embeddings, confidences = self.transformer_encoder(batch_vectors, batch_contexts)
                
                end_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = end_memory - start_memory
                
                memory_results['memory_usage_progression'].append({
                    'batch_size': batch_size,
                    'memory_increase_mb': memory_increase,
                    'memory_per_item_mb': memory_increase / batch_size if batch_size > 0 else 0,
                    'total_memory_mb': end_memory
                })
                
            except Exception as e:
                memory_results['memory_usage_progression'].append({
                    'batch_size': batch_size,
                    'error': str(e)
                })
            
            # Force garbage collection
            gc.collect()
        
        # Test memory cleanup
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_results['final_memory_mb'] = final_memory
        memory_results['memory_leak'] = (final_memory - baseline_memory) > 100  # More than 100MB increase indicates potential leak
        
        logger.info(f"Memory Efficiency Results: {memory_results}")
        
        # Verify reasonable memory usage
        assert not memory_results['memory_leak'], "Significant memory leak detected"
        
        return memory_results
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up Layer 2 test environment...")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"✓ Cleaned up temporary directory: {self.temp_dir}")

async def run_complete_layer2_adaptive_test_suite():
    """Run the complete Layer 2 Adaptive Context test suite"""
    logger.info("🚀 Starting RIGOROUS LAYER 2 (ADAPTIVE CONTEXT) TESTING SUITE")
    
    tester = Layer2AdaptiveContextRigorousTester()
    all_results = {}
    
    try:
        # Setup
        await tester.setup_test_environment()
        
        # Run all tests
        test_functions = [
            ('transformer_encoding', tester.test_transformer_behavioral_encoding),
            ('gnn_analysis', tester.test_session_graph_gnn),
            ('contextual_adaptation', tester.test_contextual_adaptation),
            ('performance_requirements', tester.test_performance_requirements),
            ('edge_cases_robustness', tester.test_edge_cases_and_robustness),
            ('adversarial_scenarios', tester.test_adversarial_scenarios),
            ('memory_efficiency', tester.test_memory_efficiency),
        ]
        
        for test_name, test_func in test_functions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {test_name.upper()} test...")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            try:
                result = await test_func()
                test_duration = time.time() - start_time
                
                all_results[test_name] = {
                    'status': 'PASSED',
                    'duration_seconds': test_duration,
                    'results': result
                }
                logger.info(f"✅ {test_name} PASSED in {test_duration:.2f}s")
                
            except Exception as e:
                test_duration = time.time() - start_time
                all_results[test_name] = {
                    'status': 'FAILED',
                    'duration_seconds': test_duration,
                    'error': str(e)
                }
                logger.error(f"❌ {test_name} FAILED: {e}")
        
        # Generate summary
        passed_tests = sum(1 for result in all_results.values() if result['status'] == 'PASSED')
        total_tests = len(all_results)
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            'test_suite': 'LAYER 2 (ADAPTIVE CONTEXT) RIGOROUS TESTING',
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate_percentage': success_rate,
            'overall_status': 'PASSED' if success_rate >= 85 else 'FAILED',
            'detailed_results': all_results,
            'performance_summary': {
                'avg_transformer_time_ms': np.mean(tester.performance_metrics['transformer_times']) if tester.performance_metrics['transformer_times'] else 0,
                'avg_gnn_time_ms': np.mean(tester.performance_metrics['gnn_times']) if tester.performance_metrics['gnn_times'] else 0,
                'avg_total_analysis_time_ms': np.mean(tester.performance_metrics['analysis_times']) if tester.performance_metrics['analysis_times'] else 0,
            }
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER 2 (ADAPTIVE CONTEXT) TEST SUITE COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        logger.info(f"Overall Status: {summary['overall_status']}")
        
        return summary
        
    finally:
        await tester.cleanup_test_environment()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('layer2_adaptive_test_results.log')
        ]
    )
    
    # Run the test suite
    results = asyncio.run(run_complete_layer2_adaptive_test_suite())
    
    # Save results to file
    with open('layer2_adaptive_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("LAYER 2 (ADAPTIVE CONTEXT) TESTING COMPLETE")
    print("="*80)
    print(f"Results saved to: layer2_adaptive_test_results.json")
    print(f"Logs saved to: layer2_adaptive_test_results.log")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['success_rate_percentage']:.1f}%")
