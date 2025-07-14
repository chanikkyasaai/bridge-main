"""
RIGOROUS LAYER 1 (FAISS) TESTING SUITE
Banking-Grade Testing for FAISS Fast Verification Component

This test suite performs exhaustive testing of the FAISS-based Layer 1 verification
system, including performance, security, edge cases, and adversarial scenarios.

Target: < 10ms per verification (as per Stage 2 requirements)
"""

import pytest
import asyncio
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import shutil
from pathlib import Path
import logging
import json
import concurrent.futures
from unittest.mock import patch, MagicMock
import faiss

# Setup test environment
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from mlengine.adapters.faiss.verifier.layer1_verifier import (
    FAISSVerifier, VerificationResult
)
from mlengine.utils.behavioral_vectors import BehavioralVector
from mlengine.config import CONFIG

logger = logging.getLogger(__name__)

class Layer1FAISSRigorousTester:
    """Comprehensive FAISS Layer 1 testing framework"""
    
    def __init__(self):
        self.temp_dir = None
        self.verifier = None
        self.test_vectors = None
        self.test_users = []
        self.performance_metrics = {
            'verification_times': [],
            'index_build_times': [],
            'memory_usage': [],
            'accuracy_scores': []
        }
        
    async def setup_test_environment(self):
        """Setup isolated test environment"""
        logger.info("Setting up FAISS Layer 1 test environment...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="faiss_layer1_test_")
        
        # Override config for testing
        CONFIG.FAISS_INDEX_PATH = os.path.join(self.temp_dir, "test_index.faiss")
        CONFIG.USER_PROFILES_PATH = self.temp_dir
        CONFIG.BEHAVIORAL_VECTOR_DIM = 128  # Smaller for faster testing
        CONFIG.L1_HIGH_CONFIDENCE_THRESHOLD = 0.85
        CONFIG.L1_MEDIUM_CONFIDENCE_THRESHOLD = 0.70
        CONFIG.L1_LOW_CONFIDENCE_THRESHOLD = 0.55
        
        # Initialize verifier
        self.verifier = FAISSVerifier(index_path=CONFIG.FAISS_INDEX_PATH)
        await self.verifier.initialize()
        
        # Generate test data
        self._generate_test_data()
        
        logger.info(f"Test environment ready at {self.temp_dir}")
    
    def _generate_test_data(self):
        """Generate comprehensive test vectors and user profiles"""
        np.random.seed(42)  # Reproducible results
        
        # Generate base behavioral patterns for different user types
        self.test_users = ['user_normal', 'user_power', 'user_senior', 'user_mobile']
        self.test_vectors = {}
        
        for user_id in self.test_users:
            # Generate unique behavioral signature for each user
            if user_id == 'user_normal':
                # Standard office worker pattern
                base_pattern = np.random.normal(0.5, 0.1, CONFIG.BEHAVIORAL_VECTOR_DIM)
            elif user_id == 'user_power':
                # Fast, efficient pattern
                base_pattern = np.random.normal(0.7, 0.05, CONFIG.BEHAVIORAL_VECTOR_DIM)
            elif user_id == 'user_senior':
                # Slower, more deliberate pattern
                base_pattern = np.random.normal(0.3, 0.15, CONFIG.BEHAVIORAL_VECTOR_DIM)
            else:  # mobile user
                # Higher variability due to mobile interface
                base_pattern = np.random.normal(0.5, 0.2, CONFIG.BEHAVIORAL_VECTOR_DIM)
            
            # Generate variations around base pattern
            user_vectors = []
            for i in range(50):  # 50 vectors per user for training
                noise = np.random.normal(0, 0.05, CONFIG.BEHAVIORAL_VECTOR_DIM)
                vector = np.clip(base_pattern + noise, 0, 1)
                user_vectors.append(vector)
            
            self.test_vectors[user_id] = np.array(user_vectors)
    
    async def test_basic_functionality(self):
        """Test basic FAISS functionality"""
        logger.info("Testing basic FAISS functionality...")
        
        results = {}
        
        # Test 1: Index initialization
        start_time = time.time()
        assert self.verifier.is_initialized, "FAISS verifier should be initialized"
        assert self.verifier.index is not None, "FAISS index should exist"
        init_time = (time.time() - start_time) * 1000
        results['initialization_time_ms'] = init_time
        
        # Test 2: Adding user vectors
        user_id = self.test_users[0]
        vectors = self.test_vectors[user_id][:10]  # Use first 10 vectors
        labels = ['normal'] * 10
        
        start_time = time.time()
        self.verifier.add_user_vectors(user_id, vectors, labels)
        add_time = (time.time() - start_time) * 1000
        results['vector_addition_time_ms'] = add_time
        
        # Verify vectors were added
        assert self.verifier.index.ntotal == 10, f"Expected 10 vectors in index, got {self.verifier.index.ntotal}"
        assert user_id in self.verifier.user_bank.profiles, "User profile should be created"
        
        # Test 3: Basic verification
        test_vector = BehavioralVector(
            vector=vectors[0] + np.random.normal(0, 0.01, CONFIG.BEHAVIORAL_VECTOR_DIM),
            timestamp=datetime.now(),
            confidence=1.0,
            source='test'
        )
        
        start_time = time.time()
        result = await self.verifier.verify([test_vector], user_id, 'test_session')
        verify_time = (time.time() - start_time) * 1000
        results['verification_time_ms'] = verify_time
        
        assert isinstance(result, VerificationResult), "Should return VerificationResult"
        assert result.user_id == user_id, "Result should match user ID"
        assert result.similarity_score > 0.5, f"Should have reasonable similarity, got {result.similarity_score}"
        
        results['similarity_score'] = result.similarity_score
        results['decision'] = result.decision
        
        logger.info(f"✓ Basic functionality test passed: {results}")
        return results
    
    async def test_performance_requirements(self):
        """Test performance requirements (< 10ms target)"""
        logger.info("Testing performance requirements...")
        
        # Setup: Add multiple users with substantial profiles
        for user_id in self.test_users:
            vectors = self.test_vectors[user_id]
            labels = ['normal'] * len(vectors)
            self.verifier.add_user_vectors(user_id, vectors, labels)
        
        # Performance test: Multiple verifications
        verification_times = []
        user_id = self.test_users[0]
        
        for i in range(100):  # 100 verification attempts
            # Create slightly modified vector
            base_vector = self.test_vectors[user_id][0]
            test_vector = BehavioralVector(
                vector=base_vector + np.random.normal(0, 0.02, CONFIG.BEHAVIORAL_VECTOR_DIM),
                timestamp=datetime.now(),
                confidence=1.0,
                source='perf_test'
            )
            
            start_time = time.perf_counter()
            result = await self.verifier.verify([test_vector], user_id, f'session_{i}')
            end_time = time.perf_counter()
            
            verification_time_ms = (end_time - start_time) * 1000
            verification_times.append(verification_time_ms)
        
        # Analyze performance
        avg_time = np.mean(verification_times)
        p95_time = np.percentile(verification_times, 95)
        p99_time = np.percentile(verification_times, 99)
        max_time = np.max(verification_times)
        
        performance_results = {
            'avg_verification_time_ms': avg_time,
            'p95_verification_time_ms': p95_time,
            'p99_verification_time_ms': p99_time,
            'max_verification_time_ms': max_time,
            'total_vectors_in_index': self.verifier.index.ntotal,
            'meets_10ms_target': avg_time < 10.0,
            'p95_meets_10ms_target': p95_time < 10.0
        }
        
        logger.info(f"Performance Results: {performance_results}")
        
        # Assertions for banking requirements
        assert avg_time < 10.0, f"Average verification time {avg_time:.2f}ms exceeds 10ms target"
        assert p95_time < 15.0, f"P95 verification time {p95_time:.2f}ms too high for banking"
        
        self.performance_metrics['verification_times'].extend(verification_times)
        
        return performance_results
    
    async def test_similarity_accuracy(self):
        """Test similarity matching accuracy"""
        logger.info("Testing similarity accuracy...")
        
        user_id = self.test_users[0]
        base_vectors = self.test_vectors[user_id]
        
        # Add user profile
        self.verifier.add_user_vectors(user_id, base_vectors[:30], ['normal'] * 30)
        
        accuracy_results = {
            'same_user_matches': 0,
            'different_user_matches': 0,
            'total_same_user_tests': 0,
            'total_different_user_tests': 0
        }
        
        # Test 1: Same user vectors should match with high similarity
        for i in range(10):
            # Use remaining vectors from same user
            test_vector = BehavioralVector(
                vector=base_vectors[30 + i],
                timestamp=datetime.now(),
                confidence=1.0,
                source='accuracy_test'
            )
            
            result = await self.verifier.verify([test_vector], user_id, f'acc_session_{i}')
            
            accuracy_results['total_same_user_tests'] += 1
            if result.similarity_score > 0.7:  # Should be high for same user
                accuracy_results['same_user_matches'] += 1
        
        # Test 2: Different user vectors should have lower similarity
        different_user = self.test_users[1]
        different_vectors = self.test_vectors[different_user]
        
        for i in range(10):
            test_vector = BehavioralVector(
                vector=different_vectors[i],
                timestamp=datetime.now(),
                confidence=1.0,
                source='accuracy_test'
            )
            
            result = await self.verifier.verify([test_vector], user_id, f'diff_session_{i}')
            
            accuracy_results['total_different_user_tests'] += 1
            if result.similarity_score < 0.5:  # Should be low for different user
                accuracy_results['different_user_matches'] += 1
        
        # Calculate accuracy metrics
        same_user_accuracy = accuracy_results['same_user_matches'] / accuracy_results['total_same_user_tests']
        different_user_accuracy = accuracy_results['different_user_matches'] / accuracy_results['total_different_user_tests']
        
        accuracy_results['same_user_accuracy'] = same_user_accuracy
        accuracy_results['different_user_accuracy'] = different_user_accuracy
        accuracy_results['overall_accuracy'] = (same_user_accuracy + different_user_accuracy) / 2
        
        logger.info(f"Accuracy Results: {accuracy_results}")
        
        # Banking-grade accuracy requirements
        assert same_user_accuracy > 0.8, f"Same user accuracy {same_user_accuracy:.2f} too low"
        assert different_user_accuracy > 0.8, f"Different user accuracy {different_user_accuracy:.2f} too low"
        
        return accuracy_results
    
    async def test_adaptive_thresholds(self):
        """Test adaptive threshold learning"""
        logger.info("Testing adaptive threshold learning...")
        
        threshold_learner = AdaptiveThresholdLearner()
        user_id = "test_adaptive_user"
        
        # Initial thresholds (should be defaults)
        initial_thresholds = threshold_learner.get_thresholds(user_id)
        expected_defaults = {
            'high': CONFIG.L1_HIGH_CONFIDENCE_THRESHOLD,
            'medium': CONFIG.L1_MEDIUM_CONFIDENCE_THRESHOLD,
            'low': CONFIG.L1_LOW_CONFIDENCE_THRESHOLD
        }
        
        assert initial_thresholds == expected_defaults, "Should return default thresholds for new user"
        
        # Simulate feedback for legitimate transactions
        legitimate_scores = np.random.normal(0.85, 0.05, 30)  # High similarity for legitimate
        for score in legitimate_scores:
            threshold_learner.update_thresholds(user_id, score, True)
        
        # Simulate feedback for fraudulent transactions
        fraudulent_scores = np.random.normal(0.45, 0.1, 20)  # Low similarity for fraud
        for score in fraudulent_scores:
            threshold_learner.update_thresholds(user_id, score, False)
        
        # Check if thresholds adapted
        adapted_thresholds = threshold_learner.get_thresholds(user_id)
        
        threshold_results = {
            'initial_thresholds': initial_thresholds,
            'adapted_thresholds': adapted_thresholds,
            'threshold_adapted': adapted_thresholds != initial_thresholds,
            'high_threshold_increased': adapted_thresholds['high'] >= initial_thresholds['high'],
            'sufficient_samples_for_adaptation': len(threshold_learner.threshold_history[user_id]) >= 50
        }
        
        logger.info(f"Threshold Adaptation Results: {threshold_results}")
        
        # Verify adaptation logic
        if threshold_results['sufficient_samples_for_adaptation']:
            assert threshold_results['threshold_adapted'], "Thresholds should adapt with sufficient data"
        
        return threshold_results
    
    async def test_concurrent_operations(self):
        """Test thread safety and concurrent operations"""
        logger.info("Testing concurrent operations...")
        
        # Setup: Add base user profile
        user_id = self.test_users[0]
        base_vectors = self.test_vectors[user_id][:20]
        self.verifier.add_user_vectors(user_id, base_vectors, ['normal'] * 20)
        
        # Concurrent verification test
        async def verify_worker(worker_id: int):
            results = []
            for i in range(10):
                test_vector = BehavioralVector(
                    vector=base_vectors[0] + np.random.normal(0, 0.02, CONFIG.BEHAVIORAL_VECTOR_DIM),
                    timestamp=datetime.now(),
                    confidence=1.0,
                    source=f'worker_{worker_id}'
                )
                
                start_time = time.perf_counter()
                result = await self.verifier.verify([test_vector], user_id, f'session_{worker_id}_{i}')
                end_time = time.perf_counter()
                
                results.append({
                    'worker_id': worker_id,
                    'iteration': i,
                    'similarity_score': result.similarity_score,
                    'processing_time_ms': (end_time - start_time) * 1000,
                    'decision': result.decision
                })
            return results
        
        # Run 5 workers concurrently
        tasks = [verify_worker(i) for i in range(5)]
        all_results = await asyncio.gather(*tasks)
        
        # Analyze concurrent performance
        all_times = []
        all_scores = []
        
        for worker_results in all_results:
            for result in worker_results:
                all_times.append(result['processing_time_ms'])
                all_scores.append(result['similarity_score'])
        
        concurrency_results = {
            'total_concurrent_operations': len(all_times),
            'avg_concurrent_time_ms': np.mean(all_times),
            'max_concurrent_time_ms': np.max(all_times),
            'score_consistency': np.std(all_scores),  # Lower is better
            'all_operations_successful': len(all_times) == 50,  # 5 workers * 10 operations
            'concurrent_performance_degradation': np.mean(all_times) / np.mean(self.performance_metrics['verification_times']) if self.performance_metrics['verification_times'] else 1.0
        }
        
        logger.info(f"Concurrency Results: {concurrency_results}")
        
        # Verify thread safety
        assert concurrency_results['all_operations_successful'], "All concurrent operations should succeed"
        assert concurrency_results['concurrent_performance_degradation'] < 2.0, "Concurrent performance shouldn't degrade more than 2x"
        
        return concurrency_results
    
    async def test_edge_cases_and_attacks(self):
        """Test edge cases and potential attack vectors"""
        logger.info("Testing edge cases and attack vectors...")
        
        user_id = self.test_users[0]
        base_vectors = self.test_vectors[user_id][:10]
        self.verifier.add_user_vectors(user_id, base_vectors, ['normal'] * 10)
        
        edge_case_results = {}
        
        # Test 1: Zero vector
        try:
            zero_vector = BehavioralVector(
                vector=np.zeros(CONFIG.BEHAVIORAL_VECTOR_DIM),
                timestamp=datetime.now(),
                confidence=1.0,
                source='zero_test'
            )
            result = await self.verifier.verify([zero_vector], user_id, 'zero_session')
            edge_case_results['zero_vector_handled'] = True
            edge_case_results['zero_vector_similarity'] = result.similarity_score
        except Exception as e:
            edge_case_results['zero_vector_handled'] = False
            edge_case_results['zero_vector_error'] = str(e)
        
        # Test 2: Extreme values
        try:
            extreme_vector = BehavioralVector(
                vector=np.full(CONFIG.BEHAVIORAL_VECTOR_DIM, 1000.0),
                timestamp=datetime.now(),
                confidence=1.0,
                source='extreme_test'
            )
            result = await self.verifier.verify([extreme_vector], user_id, 'extreme_session')
            edge_case_results['extreme_vector_handled'] = True
            edge_case_results['extreme_vector_similarity'] = result.similarity_score
        except Exception as e:
            edge_case_results['extreme_vector_handled'] = False
            edge_case_results['extreme_vector_error'] = str(e)
        
        # Test 3: NaN/Inf values
        try:
            nan_vector = BehavioralVector(
                vector=np.full(CONFIG.BEHAVIORAL_VECTOR_DIM, np.nan),
                timestamp=datetime.now(),
                confidence=1.0,
                source='nan_test'
            )
            result = await self.verifier.verify([nan_vector], user_id, 'nan_session')
            edge_case_results['nan_vector_handled'] = True
        except Exception as e:
            edge_case_results['nan_vector_handled'] = False
            edge_case_results['nan_vector_error'] = str(e)
        
        # Test 4: Replay attack (same vector repeated)
        replay_vector = BehavioralVector(
            vector=base_vectors[0],  # Exact copy
            timestamp=datetime.now(),
            confidence=1.0,
            source='replay_test'
        )
        
        replay_results = []
        for i in range(5):
            result = await self.verifier.verify([replay_vector], user_id, f'replay_session_{i}')
            replay_results.append(result.similarity_score)
        
        edge_case_results['replay_attack_scores'] = replay_results
        edge_case_results['replay_attack_detected'] = any(score < 0.9 for score in replay_results)
        
        # Test 5: Wrong user verification
        try:
            wrong_user_result = await self.verifier.verify([replay_vector], 'nonexistent_user', 'wrong_user_session')
            edge_case_results['wrong_user_handled'] = True
            edge_case_results['wrong_user_decision'] = wrong_user_result.decision
        except Exception as e:
            edge_case_results['wrong_user_handled'] = False
            edge_case_results['wrong_user_error'] = str(e)
        
        logger.info(f"Edge Case Results: {edge_case_results}")
        
        # Verify robustness
        assert edge_case_results.get('zero_vector_handled', False), "Should handle zero vectors gracefully"
        assert edge_case_results.get('extreme_vector_handled', False), "Should handle extreme values"
        
        return edge_case_results
    
    async def test_memory_and_scalability(self):
        """Test memory usage and scalability"""
        logger.info("Testing memory usage and scalability...")
        
        import psutil
        import gc
        
        # Baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        scalability_results = {
            'baseline_memory_mb': baseline_memory,
            'memory_per_user': [],
            'memory_per_vector': [],
            'index_size_progression': []
        }
        
        # Test scaling with users and vectors
        for user_count in [10, 50, 100]:
            # Add users with vectors
            for i in range(user_count):
                user_id = f"scale_user_{i}"
                vectors = np.random.randn(100, CONFIG.BEHAVIORAL_VECTOR_DIM).astype(np.float32)
                labels = ['normal'] * 100
                self.verifier.add_user_vectors(user_id, vectors, labels)
            
            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - baseline_memory
            
            scalability_results['memory_per_user'].append(memory_increase / user_count)
            scalability_results['index_size_progression'].append(self.verifier.index.ntotal)
            
            # Test verification performance with larger index
            test_vector = BehavioralVector(
                vector=np.random.randn(CONFIG.BEHAVIORAL_VECTOR_DIM),
                timestamp=datetime.now(),
                confidence=1.0,
                source='scale_test'
            )
            
            start_time = time.perf_counter()
            result = await self.verifier.verify([test_vector], 'scale_user_0', f'scale_session_{user_count}')
            verification_time = (time.perf_counter() - start_time) * 1000
            
            scalability_results[f'verification_time_at_{user_count}_users_ms'] = verification_time
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024
        scalability_results['final_memory_mb'] = final_memory
        scalability_results['total_memory_increase_mb'] = final_memory - baseline_memory
        
        # Force garbage collection and measure
        gc.collect()
        gc_memory = process.memory_info().rss / 1024 / 1024
        scalability_results['memory_after_gc_mb'] = gc_memory
        
        logger.info(f"Scalability Results: {scalability_results}")
        
        # Verify reasonable memory usage
        assert scalability_results['total_memory_increase_mb'] < 500, "Memory usage should be reasonable"
        
        return scalability_results
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up FAISS Layer 1 test environment...")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

async def run_complete_layer1_faiss_test_suite():
    """Run the complete FAISS Layer 1 test suite"""
    logger.info("Starting RIGOROUS LAYER 1 (FAISS) TESTING SUITE")
    
    tester = Layer1FAISSRigorousTester()
    all_results = {}
    
    try:
        # Setup
        await tester.setup_test_environment()
        
        # Run all tests
        test_functions = [
            ('basic_functionality', tester.test_basic_functionality),
            ('performance_requirements', tester.test_performance_requirements),
            ('similarity_accuracy', tester.test_similarity_accuracy),
            ('adaptive_thresholds', tester.test_adaptive_thresholds),
            ('concurrent_operations', tester.test_concurrent_operations),
            ('edge_cases_and_attacks', tester.test_edge_cases_and_attacks),
            ('memory_and_scalability', tester.test_memory_and_scalability),
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
                logger.info(f"PASSED {test_name} PASSED in {test_duration:.2f}s")
                
            except Exception as e:
                test_duration = time.time() - start_time
                all_results[test_name] = {
                    'status': 'FAILED',
                    'duration_seconds': test_duration,
                    'error': str(e)
                }
                logger.error(f"FAILED {test_name} FAILED: {e}")
        
        # Generate summary
        passed_tests = sum(1 for result in all_results.values() if result['status'] == 'PASSED')
        total_tests = len(all_results)
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            'test_suite': 'LAYER 1 (FAISS) RIGOROUS TESTING',
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate_percentage': success_rate,
            'overall_status': 'PASSED' if success_rate >= 85 else 'FAILED',
            'detailed_results': all_results
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"LAYER 1 (FAISS) TEST SUITE COMPLETE")
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
            logging.FileHandler('layer1_faiss_test_results.log')
        ]
    )
    
    # Run the test suite
    results = asyncio.run(run_complete_layer1_faiss_test_suite())
    
    # Save results to file
    with open('layer1_faiss_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("LAYER 1 (FAISS) TESTING COMPLETE")
    print("="*80)
    print(f"Results saved to: layer1_faiss_test_results.json")
    print(f"Logs saved to: layer1_faiss_test_results.log")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Success Rate: {results['success_rate_percentage']:.1f}%")
