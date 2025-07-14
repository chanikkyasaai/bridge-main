"""
Comprehensive GNN Testing Suite for BRIDGE ML-Engine
Tests the Graph Neural Network component for correctness and completeness
"""

import sys
import os
import asyncio
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import logging
import time
import json
from typing import List, Dict, Any, Tuple
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.adapters.level2.layer2_verifier import (
    SessionGraphGNN, SessionGraphBuilder, SessionGraph, 
    BehavioralEvent, ContextualFeatures, Layer2Verifier
)
from ml_engine.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNNTester:
    """Comprehensive GNN testing suite"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        self.performance_metrics = {
            'initialization_time': 0,
            'forward_pass_times': [],
            'graph_build_times': [],
            'memory_usage': []
        }
    
    async def run_all_tests(self):
        """Run comprehensive GNN test suite"""
        logger.info("🚀 Starting Comprehensive GNN Test Suite")
        logger.info(f"Device: {self.device}")
        
        tests = [
            ("initialization", self.test_gnn_initialization),
            ("forward_pass", self.test_gnn_forward_pass),
            ("graph_builder", self.test_session_graph_builder),
            ("anomaly_detection", self.test_anomaly_detection),
            ("layer2_integration", self.test_layer2_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running {test_name} tests...")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                logger.info(f"{status}: {test_name}")
                if not result['passed']:
                    logger.error(f"Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"❌ FAILED: {test_name} - {str(e)}")
                self.test_results[test_name] = {'passed': False, 'error': str(e)}
        
        # Generate final report
        await self.generate_test_report()
    
    async def test_gnn_initialization(self):
        """Test GNN model initialization"""
        try:
            start_time = time.time()
            
            # Test basic initialization
            gnn = SessionGraphGNN(node_features=32, edge_features=16)
            gnn.to(self.device)
            
            init_time = time.time() - start_time
            self.performance_metrics['initialization_time'] = init_time
            
            # Verify model structure
            assert hasattr(gnn, 'conv1'), "Missing conv1 layer"
            assert hasattr(gnn, 'conv2'), "Missing conv2 layer"
            assert hasattr(gnn, 'conv3'), "Missing conv3 layer"
            assert hasattr(gnn, 'anomaly_head'), "Missing anomaly_head"
            assert hasattr(gnn, 'graph_embedding'), "Missing graph_embedding"
            
            # Check layer types
            assert isinstance(gnn.conv1, GATConv), "conv1 should be GATConv"
            assert isinstance(gnn.conv2, GATConv), "conv2 should be GATConv"
            assert isinstance(gnn.conv3, GCNConv), "conv3 should be GCNConv"
            
            # Test parameter count
            param_count = sum(p.numel() for p in gnn.parameters())
            logger.info(f"GNN parameter count: {param_count:,}")
            
            # Verify device placement
            for param in gnn.parameters():
                assert param.device == self.device, f"Parameter not on {self.device}"
            
            return {
                'passed': True,
                'initialization_time': init_time,
                'parameter_count': param_count,
                'device': str(self.device)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def test_gnn_forward_pass(self):
        """Test GNN forward pass with various inputs"""
        try:
            gnn = SessionGraphGNN(node_features=32, edge_features=16)
            gnn.to(self.device)
            gnn.eval()
            
            results = []
            
            # Test 1: Simple graph
            node_features = torch.randn(5, 32).to(self.device)
            edge_index = torch.LongTensor([[0, 1, 2, 3], [1, 2, 3, 4]]).to(self.device)
            batch = torch.zeros(5, dtype=torch.long).to(self.device)
            
            data = Data(x=node_features, edge_index=edge_index, batch=batch)
            
            start_time = time.time()
            with torch.no_grad():
                anomaly_scores, embeddings = gnn(data)
            forward_time = time.time() - start_time
            
            self.performance_metrics['forward_pass_times'].append(forward_time)
            
            # Verify output shapes
            assert anomaly_scores.shape == (1, 1), f"Expected anomaly shape (1,1), got {anomaly_scores.shape}"
            assert embeddings.shape == (1, CONFIG.GRAPH_EMBEDDING_DIM), f"Expected embedding shape (1,{CONFIG.GRAPH_EMBEDDING_DIM}), got {embeddings.shape}"
            
            # Verify output ranges
            assert 0 <= anomaly_scores.item() <= 1, f"Anomaly score {anomaly_scores.item()} not in [0,1]"
            assert not torch.isnan(anomaly_scores).any(), "NaN in anomaly scores"
            assert not torch.isnan(embeddings).any(), "NaN in embeddings"
            
            results.append({
                'test': 'simple_graph',
                'forward_time': forward_time,
                'anomaly_score': anomaly_scores.item(),
                'embedding_norm': torch.norm(embeddings).item()
            })
            
            return {
                'passed': True,
                'results': results,
                'avg_forward_time': np.mean(self.performance_metrics['forward_pass_times'])
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def test_session_graph_builder(self):
        """Test session graph building from behavioral events"""
        try:
            builder = SessionGraphBuilder()
            
            # Create test events
            events = []
            base_time = datetime.now()
            
            for i in range(10):
                event = BehavioralEvent(
                    timestamp=base_time + timedelta(seconds=i),
                    event_type=["touch", "swipe", "type", "scroll"][i % 4],
                    features={
                        "pressure": 0.5 + 0.1 * i,
                        "velocity": 1.0 + 0.2 * i,
                        "duration": 0.1 + 0.05 * i
                    },
                    session_id="test_session",
                    user_id="test_user",
                    device_id="test_device",
                    raw_metadata={}
                )
                events.append(event)
            
            # Build graph
            start_time = time.time()
            for event in events:
                builder.add_event(event)
            build_time = time.time() - start_time
            
            self.performance_metrics['graph_build_times'].append(build_time)
            
            # Get built graph
            graph = builder.get_graph("test_session")
            
            # Verify graph structure
            assert graph is not None, "Graph not built"
            assert graph.session_id == "test_session", "Wrong session ID"
            assert graph.user_id == "test_user", "Wrong user ID"
            assert len(graph.nodes) == 10, f"Expected 10 nodes, got {len(graph.nodes)}"
            assert len(graph.edges) == 9, f"Expected 9 edges, got {len(graph.edges)}"
            
            # Verify node features
            for i, node in enumerate(graph.nodes):
                assert "event_type" in node, f"Missing event_type in node {i}"
                assert "features" in node, f"Missing features in node {i}"
                assert "timestamp" in node, f"Missing timestamp in node {i}"
            
            # Verify edge features
            for i, edge_feat in enumerate(graph.edge_features):
                assert "time_difference" in edge_feat, f"Missing time_difference in edge {i}"
                assert "transition_type" in edge_feat, f"Missing transition_type in edge {i}"
                assert "interaction_speed" in edge_feat, f"Missing interaction_speed in edge {i}"
            
            return {
                'passed': True,
                'build_time': build_time,
                'nodes_created': len(graph.nodes),
                'edges_created': len(graph.edges)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def test_anomaly_detection(self):
        """Test anomaly detection capabilities"""
        try:
            gnn = SessionGraphGNN(node_features=32, edge_features=16)
            gnn.to(self.device)
            gnn.eval()
            
            # Normal behavior pattern
            normal_features = torch.randn(10, 32).to(self.device) * 0.5  # Small variance
            normal_edges = torch.LongTensor([[i, i+1] for i in range(9)]).t().to(self.device)
            normal_batch = torch.zeros(10, dtype=torch.long).to(self.device)
            normal_data = Data(x=normal_features, edge_index=normal_edges, batch=normal_batch)
            
            with torch.no_grad():
                normal_score, _ = gnn(normal_data)
            
            # Anomalous behavior pattern
            anomalous_features = torch.randn(10, 32).to(self.device) * 2.0  # Large variance
            anomalous_features[5:] += 3.0  # Sudden change
            anomalous_data = Data(x=anomalous_features, edge_index=normal_edges, batch=normal_batch)
            
            with torch.no_grad():
                anomalous_score, _ = gnn(anomalous_data)
            
            results = {
                'normal_score': normal_score.item(),
                'anomalous_score': anomalous_score.item()
            }
            
            # Basic sanity checks
            all_scores = list(results.values())
            assert all(0 <= score <= 1 for score in all_scores), "Scores not in [0,1] range"
            assert not any(np.isnan(score) for score in all_scores), "NaN scores detected"
            
            return {
                'passed': True,
                'scores': results,
                'score_variance': np.var(all_scores)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def test_layer2_integration(self):
        """Test GNN integration with Layer2 verifier"""
        try:
            # Initialize Layer2 verifier
            layer2 = Layer2Verifier()
            await layer2.initialize()
            
            # Create test behavioral events
            events = []
            base_time = datetime.now()
            
            for i in range(10):
                event = BehavioralEvent(
                    timestamp=base_time + timedelta(seconds=i),
                    event_type=["touch", "swipe", "type"][i % 3],
                    features={
                        "pressure": 0.5 + 0.1 * np.sin(i),
                        "velocity": 1.0 + 0.2 * np.cos(i),
                        "duration": 0.1 + 0.05 * i
                    },
                    session_id="integration_test",
                    user_id="test_user",
                    device_id="test_device",
                    raw_metadata={}
                )
                events.append(event)
            
            # Create session graph
            session_graph = layer2.build_session_graph(events, "integration_test", "test_user")
            
            # Test GNN analysis
            start_time = time.time()
            gnn_score = await layer2._analyze_with_gnn(session_graph)
            analysis_time = time.time() - start_time
            
            # Verify results
            assert isinstance(gnn_score, float), f"Expected float score, got {type(gnn_score)}"
            assert 0 <= gnn_score <= 1, f"Score {gnn_score} not in [0,1] range"
            assert not np.isnan(gnn_score), "NaN score returned"
            
            return {
                'passed': True,
                'gnn_score': gnn_score,
                'analysis_time': analysis_time,
                'events_processed': len(events),
                'graph_nodes': len(session_graph.nodes),
                'graph_edges': len(session_graph.edges)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*80)
        logger.info("🎯 BRIDGE GNN COMPREHENSIVE TEST REPORT")
        logger.info("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('passed', False))
        
        logger.info(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        logger.info(f"🖥️  Device: {self.device}")
        
        if self.performance_metrics['initialization_time']:
            logger.info(f"⚡ Initialization Time: {self.performance_metrics['initialization_time']:.3f}s")
        
        if self.performance_metrics['forward_pass_times']:
            avg_forward = np.mean(self.performance_metrics['forward_pass_times'])
            logger.info(f"🚀 Average Forward Pass: {avg_forward*1000:.2f}ms")
        
        logger.info("\n📋 Test Details:")
        for test_name, result in self.test_results.items():
            status = "✅" if result.get('passed', False) else "❌"
            logger.info(f"{status} {test_name.replace('_', ' ').title()}")
            
            if not result.get('passed', False) and 'error' in result:
                logger.info(f"   Error: {result['error']}")
        
        # Save detailed results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': passed_tests / total_tests,
                'device': str(self.device)
            },
            'performance_metrics': {
                'initialization_time': self.performance_metrics['initialization_time'],
                'avg_forward_pass_time': np.mean(self.performance_metrics['forward_pass_times']) if self.performance_metrics['forward_pass_times'] else 0,
                'avg_graph_build_time': np.mean(self.performance_metrics['graph_build_times']) if self.performance_metrics['graph_build_times'] else 0
            },
            'detailed_results': self.test_results
        }
        
        with open('gnn_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info("\n💾 Detailed report saved to: gnn_test_report.json")
        
        # Summary recommendations
        logger.info("\n🔍 Recommendations:")
        
        if passed_tests == total_tests:
            logger.info("✅ All tests passed! GNN implementation is working correctly.")
        else:
            failed_tests = [name for name, result in self.test_results.items() if not result.get('passed', False)]
            logger.info(f"❌ Failed tests need attention: {', '.join(failed_tests)}")
        
        logger.info("\n" + "="*80)

if __name__ == "__main__":
    async def main():
        tester = GNNTester()
        await tester.run_all_tests()
    
    asyncio.run(main())
