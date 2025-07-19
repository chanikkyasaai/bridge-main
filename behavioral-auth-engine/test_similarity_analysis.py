#!/usr/bin/env python3
"""
FAISS Similarity Analysis and Decision Engine Test
Focused testing of similarity calculations and decision-making logic
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(__file__))

from src.core.enhanced_faiss_engine import EnhancedFAISSEngine
from src.core.ml_database import ml_db
from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor

class SimilarityAnalyzer:
    """Analyzes and improves FAISS similarity calculations"""
    
    def __init__(self):
        self.faiss_engine = None
        self.processor = None
        self.similarity_data = []
        
    async def initialize(self):
        """Initialize components"""
        try:
            await ml_db.initialize()
            self.faiss_engine = EnhancedFAISSEngine(ml_db)
            self.processor = EnhancedBehavioralProcessor()
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def calculate_multiple_similarities(self, vector1: np.ndarray, vector2: np.ndarray) -> Dict[str, float]:
        """Calculate multiple types of similarity metrics"""
        
        similarities = {}
        
        try:
            # Ensure same length
            min_len = min(len(vector1), len(vector2))
            v1 = vector1[:min_len]
            v2 = vector2[:min_len]
            
            # 1. Cosine Similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                similarities['cosine'] = float(dot_product / (norm1 * norm2))
            else:
                similarities['cosine'] = 0.0
            
            # 2. Euclidean Distance (converted to similarity)
            euclidean_dist = np.linalg.norm(v1 - v2)
            similarities['euclidean'] = float(1.0 / (1.0 + euclidean_dist))
            
            # 3. Pearson Correlation
            if len(v1) > 1:
                correlation = np.corrcoef(v1, v2)[0, 1]
                similarities['pearson'] = float(correlation) if not np.isnan(correlation) else 0.0
            else:
                similarities['pearson'] = 0.0
            
            # 4. Manhattan Distance (converted to similarity)
            manhattan_dist = np.sum(np.abs(v1 - v2))
            similarities['manhattan'] = float(1.0 / (1.0 + manhattan_dist))
            
            # 5. Jaccard Similarity (for non-zero elements)
            nonzero1 = set(np.nonzero(v1)[0])
            nonzero2 = set(np.nonzero(v2)[0])
            
            if len(nonzero1) > 0 or len(nonzero2) > 0:
                intersection = len(nonzero1.intersection(nonzero2))
                union = len(nonzero1.union(nonzero2))
                similarities['jaccard'] = float(intersection / union) if union > 0 else 0.0
            else:
                similarities['jaccard'] = 0.0
            
            # 6. Weighted Cosine (emphasizing non-zero elements)
            weights = np.where((v1 != 0) | (v2 != 0), 1.0, 0.1)
            weighted_dot = np.sum(weights * v1 * v2)
            weighted_norm1 = np.sqrt(np.sum(weights * v1 * v1))
            weighted_norm2 = np.sqrt(np.sum(weights * v2 * v2))
            
            if weighted_norm1 > 0 and weighted_norm2 > 0:
                similarities['weighted_cosine'] = float(weighted_dot / (weighted_norm1 * weighted_norm2))
            else:
                similarities['weighted_cosine'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            for key in ['cosine', 'euclidean', 'pearson', 'manhattan', 'jaccard', 'weighted_cosine']:
                similarities[key] = 0.0
        
        return similarities
    
    def make_enhanced_decision(self, similarities: Dict[str, float], user_phase: str = "full_auth") -> Tuple[str, float, Dict[str, Any]]:
        """Make authentication decision using multiple similarity metrics"""
        
        # Weight different similarity metrics
        weights = {
            'cosine': 0.3,
            'weighted_cosine': 0.25,
            'euclidean': 0.15,
            'pearson': 0.15,
            'jaccard': 0.1,
            'manhattan': 0.05
        }
        
        # Calculate weighted similarity score
        weighted_score = sum(similarities[metric] * weight for metric, weight in weights.items())
        
        # Phase-aware thresholds
        if user_phase == "learning":
            thresholds = {"allow": 0.3, "challenge": 0.1}
        elif user_phase == "gradual_risk":
            thresholds = {"allow": 0.6, "challenge": 0.4}
        else:  # full_auth
            thresholds = {"allow": 0.7, "challenge": 0.5}
        
        # Make decision
        if weighted_score >= thresholds["allow"]:
            decision = "allow"
            confidence = min(0.95, weighted_score + 0.1)
        elif weighted_score >= thresholds["challenge"]:
            decision = "challenge"
            confidence = 0.7
        else:
            decision = "block"
            confidence = min(0.95, 1.0 - weighted_score)
        
        # Decision factors
        factors = {
            "weighted_score": weighted_score,
            "individual_similarities": similarities,
            "thresholds_used": thresholds,
            "phase": user_phase,
            "primary_contributors": self._identify_key_contributors(similarities, weights)
        }
        
        return decision, confidence, factors
    
    def _identify_key_contributors(self, similarities: Dict[str, float], weights: Dict[str, float]) -> List[str]:
        """Identify which similarity metrics contributed most to the decision"""
        contributions = [(metric, sim * weight) for metric, sim in similarities.items() for weight in [weights[metric]]]
        contributions.sort(key=lambda x: x[1], reverse=True)
        return [metric for metric, _ in contributions[:3]]
    
    async def test_similarity_calculations(self, user_id: str, num_tests: int = 5):
        """Test similarity calculations for a user with stored vectors"""
        logger.info(f"Testing similarity calculations for user {user_id}")
        
        # Get user's historical vectors
        historical_vectors = await ml_db.get_user_vectors(user_id, limit=20)
        
        if len(historical_vectors) < 2:
            logger.error("Need at least 2 historical vectors for testing")
            return None
        
        test_results = []
        
        for test_num in range(min(num_tests, len(historical_vectors) - 1)):
            # Use one vector as "current" and others as baseline
            current_vector = np.array(historical_vectors[test_num]['vector_data'])
            baseline_vectors = [np.array(v['vector_data']) for v in historical_vectors[test_num+1:test_num+6]]
            
            # Calculate similarities with each baseline vector
            all_similarities = []
            for baseline_vector in baseline_vectors:
                similarities = self.calculate_multiple_similarities(current_vector, baseline_vector)
                all_similarities.append(similarities)
            
            # Average similarities across baseline vectors
            avg_similarities = {}
            for metric in all_similarities[0].keys():
                avg_similarities[metric] = np.mean([sim[metric] for sim in all_similarities])
            
            # Test decisions for different phases
            decisions = {}
            for phase in ["learning", "gradual_risk", "full_auth"]:
                decision, confidence, factors = self.make_enhanced_decision(avg_similarities, phase)
                decisions[phase] = {
                    "decision": decision,
                    "confidence": confidence,
                    "factors": factors
                }
            
            test_result = {
                "test_number": test_num,
                "vector_id": historical_vectors[test_num]['id'],
                "similarities": avg_similarities,
                "decisions": decisions,
                "baseline_count": len(baseline_vectors)
            }
            
            test_results.append(test_result)
            
            logger.info(f"Test {test_num}: Avg cosine similarity: {avg_similarities['cosine']:.3f}, "
                       f"Decisions: {decisions['full_auth']['decision']}")
        
        return test_results
    
    async def run_similarity_optimization_test(self):
        """Run comprehensive similarity testing to optimize decision-making"""
        logger.info("Starting similarity optimization testing")
        
        if not await self.initialize():
            return False
        
        # Get users with sufficient data
        test_users = []
        
        # For now, we'll create test data if none exists
        test_user_id = "test_similarity_user"
        
        # Generate test vectors for analysis
        logger.info("Generating test vectors for similarity analysis...")
        
        # Create baseline behavioral pattern
        base_pattern = np.random.rand(90) * 0.5  # Base pattern
        
        # Generate similar vectors (authentic user sessions)
        authentic_vectors = []
        for i in range(10):
            # Add small variations to base pattern
            noise = np.random.normal(0, 0.1, 90)
            authentic_vector = base_pattern + noise
            authentic_vector = np.clip(authentic_vector, 0, 1)  # Keep in valid range
            authentic_vectors.append(authentic_vector)
        
        # Generate dissimilar vectors (potential attacks)
        attack_vectors = []
        for i in range(5):
            # Create significantly different patterns
            if i % 2 == 0:
                # Pattern shift attack
                attack_vector = base_pattern * 0.3 + np.random.rand(90) * 0.7
            else:
                # Timing attack (different rhythm)
                attack_vector = np.roll(base_pattern, 20) + np.random.normal(0, 0.2, 90)
            
            attack_vector = np.clip(attack_vector, 0, 1)
            attack_vectors.append(attack_vector)
        
        # Store test vectors
        for i, vector in enumerate(authentic_vectors):
            await ml_db.store_behavioral_vector(
                user_id=test_user_id,
                session_id=f"authentic_session_{i}",
                vector_data=vector.tolist(),
                confidence_score=1.0,
                feature_source="similarity_test_authentic"
            )
        
        for i, vector in enumerate(attack_vectors):
            await ml_db.store_behavioral_vector(
                user_id=f"{test_user_id}_attacker",
                session_id=f"attack_session_{i}",
                vector_data=vector.tolist(),
                confidence_score=1.0,
                feature_source="similarity_test_attack"
            )
        
        # Test authentic user similarity
        logger.info("Testing authentic user sessions...")
        authentic_results = await self.test_similarity_calculations(test_user_id, num_tests=5)
        
        # Test attack detection
        logger.info("Testing attack detection...")
        attack_results = []
        
        for i, attack_vector in enumerate(attack_vectors):
            # Compare attack vector against authentic user's baseline
            baseline_vectors = [np.array(v['vector_data']) for v in 
                              (await ml_db.get_user_vectors(test_user_id, limit=5))]
            
            if baseline_vectors:
                similarities = []
                for baseline in baseline_vectors:
                    sim = self.calculate_multiple_similarities(attack_vector, baseline)
                    similarities.append(sim)
                
                # Average similarities
                avg_similarities = {}
                for metric in similarities[0].keys():
                    avg_similarities[metric] = np.mean([sim[metric] for sim in similarities])
                
                # Test decision
                decision, confidence, factors = self.make_enhanced_decision(avg_similarities, "full_auth")
                
                attack_results.append({
                    "attack_type": f"attack_{i}",
                    "similarities": avg_similarities,
                    "decision": decision,
                    "confidence": confidence,
                    "detected": decision in ["challenge", "block"]
                })
                
                logger.info(f"Attack {i}: {decision} (cosine: {avg_similarities['cosine']:.3f}) "
                           f"- {'DETECTED' if decision in ['challenge', 'block'] else 'MISSED'}")
        
        # Analysis
        logger.info("\n" + "="*60)
        logger.info("SIMILARITY ANALYSIS RESULTS")
        logger.info("="*60)
        
        # Authentic user analysis
        if authentic_results:
            authentic_decisions = [result['decisions']['full_auth']['decision'] for result in authentic_results]
            allow_rate = authentic_decisions.count('allow') / len(authentic_decisions) * 100
            
            logger.info(f"\nAUTHENTIC USER PERFORMANCE:")
            logger.info(f"Sessions tested: {len(authentic_results)}")
            logger.info(f"Allow rate: {allow_rate:.1f}%")
            
            avg_cosine = np.mean([result['similarities']['cosine'] for result in authentic_results])
            avg_weighted = np.mean([result['similarities']['weighted_cosine'] for result in authentic_results])
            
            logger.info(f"Average cosine similarity: {avg_cosine:.3f}")
            logger.info(f"Average weighted cosine: {avg_weighted:.3f}")
        
        # Attack detection analysis
        if attack_results:
            detected_attacks = sum(1 for result in attack_results if result['detected'])
            detection_rate = detected_attacks / len(attack_results) * 100
            
            logger.info(f"\nATTACK DETECTION PERFORMANCE:")
            logger.info(f"Attacks tested: {len(attack_results)}")
            logger.info(f"Detection rate: {detection_rate:.1f}%")
            
            avg_attack_cosine = np.mean([result['similarities']['cosine'] for result in attack_results])
            logger.info(f"Average attack cosine similarity: {avg_attack_cosine:.3f}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"similarity_analysis_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "authentic_results": authentic_results,
                "attack_results": attack_results,
                "analysis": {
                    "authentic_allow_rate": allow_rate if authentic_results else 0,
                    "attack_detection_rate": detection_rate if attack_results else 0,
                    "avg_authentic_similarity": avg_cosine if authentic_results else 0,
                    "avg_attack_similarity": avg_attack_cosine if attack_results else 0
                }
            }, f, indent=2, default=str)
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Recommendations
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*60)
        
        if authentic_results and allow_rate < 80:
            logger.info("❌ Authentic user allow rate is below 80%")
            logger.info("   → Lower similarity thresholds")
            logger.info("   → Increase weight of weighted_cosine metric")
        
        if attack_results and detection_rate < 90:
            logger.info("❌ Attack detection rate is below 90%")
            logger.info("   → Implement temporal sequence analysis")
            logger.info("   → Add device fingerprinting features")
        
        if authentic_results and attack_results:
            similarity_gap = avg_cosine - avg_attack_cosine
            if similarity_gap < 0.2:
                logger.info("❌ Similarity gap between authentic and attack vectors is too small")
                logger.info("   → Enhance feature extraction")
                logger.info("   → Add behavioral consistency metrics")
        
        return True

async def main():
    """Main testing function"""
    analyzer = SimilarityAnalyzer()
    success = await analyzer.run_similarity_optimization_test()
    
    if success:
        logger.info("\n🎉 SIMILARITY ANALYSIS COMPLETED!")
    else:
        logger.error("\n❌ Analysis failed.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
