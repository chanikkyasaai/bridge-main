#!/usr/bin/env python3
"""
Persona-Based Authentication Testing System
Tests FAISS similarity calculations and decision-making with realistic user personas
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core components
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.core.enhanced_faiss_engine import EnhancedFAISSEngine
from src.core.ml_database import ml_db
from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
from src.core.vector_store import HDF5VectorStore
from src.data.models import BehavioralVector, LearningPhase
from src.layers.faiss_layer import FAISSLayer
from src.layers.gnn_anomaly_detector import GNNAnomalyDetector
from src.layers.adaptive_layer import AdaptiveLayer
from src.layers.policy_orchestration_engine import PolicyOrchestrationEngine

class UserPersona:
    """Represents a user with specific behavioral patterns"""
    
    def __init__(self, name: str, persona_type: str):
        self.name = name
        self.persona_type = persona_type
        self.user_id = str(uuid.uuid4())
        self.baseline_patterns = {}
        self.session_count = 0
        self.learning_phase = "learning"
        
    def generate_behavioral_data(self, is_authentic: bool = True, attack_type: str = None) -> Dict[str, Any]:
        """Generate behavioral data based on persona characteristics"""
        
        if self.persona_type == "tech_savvy_young":
            return self._generate_tech_savvy_data(is_authentic, attack_type)
        elif self.persona_type == "elderly_careful":
            return self._generate_elderly_data(is_authentic, attack_type)
        elif self.persona_type == "business_professional":
            return self._generate_business_data(is_authentic, attack_type)
        elif self.persona_type == "attacker":
            return self._generate_attack_data(attack_type)
        
    def _generate_tech_savvy_data(self, is_authentic: bool, attack_type: str) -> Dict[str, Any]:
        """Tech-savvy young user: Fast, confident interactions"""
        base_data = {
            "events": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "data": {
                        "touch_events": [
                            {
                                "pressure": 0.7 + random.uniform(-0.1, 0.1),
                                "duration": 80 + random.randint(-20, 20),
                                "x": 250 + random.randint(-50, 50),
                                "y": 400 + random.randint(-50, 50)
                            } for _ in range(5)
                        ],
                        "accelerometer": {
                            "x": 0.05 + random.uniform(-0.02, 0.02),
                            "y": 0.1 + random.uniform(-0.02, 0.02),
                            "z": 9.8 + random.uniform(-0.1, 0.1)
                        },
                        "gyroscope": {
                            "x": 0.01 + random.uniform(-0.005, 0.005),
                            "y": 0.02 + random.uniform(-0.005, 0.005),
                            "z": 0.01 + random.uniform(-0.005, 0.005)
                        },
                        "scroll_events": [
                            {
                                "velocity": 200 + random.randint(-50, 50),
                                "distance": 400 + random.randint(-100, 100),
                                "direction": "down"
                            }
                        ]
                    }
                },
                {
                    "event_type": "typing",
                    "timestamp": int((datetime.now() + timedelta(seconds=2)).timestamp() * 1000),
                    "data": {
                        "keystroke_intervals": [120, 110, 105, 115, 108],  # Fast typing
                        "typing_pressure": [0.6, 0.7, 0.65, 0.7, 0.68],
                        "dwell_times": [80, 85, 78, 82, 79]
                    }
                }
            ]
        }
        
        if not is_authentic:
            # Simulate attack patterns
            if attack_type == "bot_attack":
                # Perfect timing - suspicious
                base_data["events"][1]["data"]["keystroke_intervals"] = [100, 100, 100, 100, 100]
                base_data["events"][0]["data"]["touch_events"] = [
                    {
                        "pressure": 0.5,  # Consistent pressure
                        "duration": 100,  # Exact duration
                        "x": 250, "y": 400  # Exact coordinates
                    } for _ in range(5)
                ]
            elif attack_type == "device_takeover":
                # Different device characteristics
                base_data["events"][0]["data"]["accelerometer"]["z"] = 8.5  # Different gravity reading
                base_data["events"][1]["data"]["keystroke_intervals"] = [200, 180, 190, 195, 185]  # Slower
                
        return base_data
    
    def _generate_elderly_data(self, is_authentic: bool, attack_type: str) -> Dict[str, Any]:
        """Elderly user: Slower, more careful interactions"""
        base_data = {
            "events": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "data": {
                        "touch_events": [
                            {
                                "pressure": 0.9 + random.uniform(-0.1, 0.1),  # Higher pressure
                                "duration": 200 + random.randint(-50, 50),  # Longer duration
                                "x": 250 + random.randint(-30, 30),  # Less variance
                                "y": 400 + random.randint(-30, 30)
                            } for _ in range(3)  # Fewer touches
                        ],
                        "accelerometer": {
                            "x": 0.02 + random.uniform(-0.01, 0.01),  # More stable
                            "y": 0.05 + random.uniform(-0.01, 0.01),
                            "z": 9.8 + random.uniform(-0.05, 0.05)
                        },
                        "scroll_events": [
                            {
                                "velocity": 80 + random.randint(-20, 20),  # Slower scrolling
                                "distance": 150 + random.randint(-50, 50),
                                "direction": "down"
                            }
                        ]
                    }
                },
                {
                    "event_type": "typing",
                    "timestamp": int((datetime.now() + timedelta(seconds=3)).timestamp() * 1000),
                    "data": {
                        "keystroke_intervals": [300, 280, 290, 310, 295],  # Slower typing
                        "typing_pressure": [0.8, 0.9, 0.85, 0.9, 0.88],  # Higher pressure
                        "dwell_times": [150, 160, 145, 155, 152]
                    }
                }
            ]
        }
        
        if not is_authentic and attack_type == "social_engineering":
            # Attacker trying to mimic elderly patterns but failing
            base_data["events"][1]["data"]["keystroke_intervals"] = [250, 320, 180, 350, 200]  # Inconsistent
            base_data["events"][0]["data"]["touch_events"][0]["pressure"] = 0.4  # Too light
            
        return base_data
    
    def _generate_business_data(self, is_authentic: bool, attack_type: str) -> Dict[str, Any]:
        """Business professional: Efficient, consistent patterns"""
        base_data = {
            "events": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "data": {
                        "touch_events": [
                            {
                                "pressure": 0.75 + random.uniform(-0.05, 0.05),  # Consistent pressure
                                "duration": 120 + random.randint(-10, 10),  # Consistent duration
                                "x": 250 + random.randint(-20, 20),
                                "y": 400 + random.randint(-20, 20)
                            } for _ in range(4)
                        ],
                        "accelerometer": {
                            "x": 0.08 + random.uniform(-0.01, 0.01),
                            "y": 0.12 + random.uniform(-0.01, 0.01),
                            "z": 9.8 + random.uniform(-0.05, 0.05)
                        },
                        "scroll_events": [
                            {
                                "velocity": 150 + random.randint(-20, 20),
                                "distance": 300 + random.randint(-50, 50),
                                "direction": "down"
                            }
                        ]
                    }
                },
                {
                    "event_type": "typing",
                    "timestamp": int((datetime.now() + timedelta(seconds=1.5)).timestamp() * 1000),
                    "data": {
                        "keystroke_intervals": [140, 135, 138, 142, 139],  # Professional typing speed
                        "typing_pressure": [0.7, 0.72, 0.69, 0.73, 0.71],
                        "dwell_times": [95, 98, 92, 96, 94]
                    }
                }
            ]
        }
        
        return base_data
    
    def _generate_attack_data(self, attack_type: str) -> Dict[str, Any]:
        """Generate various attack patterns"""
        
        if attack_type == "credential_stuffing":
            return {
                "events": [
                    {
                        "event_type": "rapid_inputs",
                        "timestamp": int(datetime.now().timestamp() * 1000),
                        "data": {
                            "touch_events": [
                                {
                                    "pressure": 0.3,  # Light, automated touches
                                    "duration": 50,   # Very fast
                                    "x": 250, "y": 400  # Exact same coordinates
                                } for _ in range(10)
                            ],
                            "keystroke_intervals": [80, 80, 80, 80, 80],  # Too consistent
                            "typing_pressure": [0.5, 0.5, 0.5, 0.5, 0.5]  # Robotic
                        }
                    }
                ]
            }
        elif attack_type == "account_takeover":
            return {
                "events": [
                    {
                        "event_type": "unusual_device",
                        "timestamp": int(datetime.now().timestamp() * 1000),
                        "data": {
                            "touch_events": [
                                {
                                    "pressure": 1.0,  # Different pressure pattern
                                    "duration": 300,  # Very different timing
                                    "x": 300, "y": 500  # Different screen region
                                } for _ in range(8)
                            ],
                            "accelerometer": {
                                "x": 0.2, "y": 0.3, "z": 9.5  # Different device orientation
                            },
                            "keystroke_intervals": [250, 180, 300, 150, 400],  # Erratic
                        }
                    }
                ]
            }
            
        return {"events": []}

class PersonaBasedTester:
    """Comprehensive testing system with user personas"""
    
    def __init__(self):
        self.faiss_engine = None
        self.behavioral_processor = None
        self.faiss_layer = None
        self.policy_engine = None
        self.test_results = {
            "training_results": {},
            "authentication_tests": {},
            "attack_detection": {},
            "similarity_analysis": {},
            "decision_accuracy": {}
        }
        
    async def initialize_components(self):
        """Initialize all testing components"""
        logger.info("Initializing behavioral authentication components...")
        
        try:
            # Initialize components (ml_db is already initialized as global instance)
            self.behavioral_processor = EnhancedBehavioralProcessor()
            self.faiss_engine = EnhancedFAISSEngine(vector_dimension=90)  # 90D behavioral vectors
            
            # Initialize vector store for FAISS layer
            vector_store = HDF5VectorStore()
            self.faiss_layer = FAISSLayer(vector_store)
            
            # Initialize GNN anomaly detector
            self.gnn_detector = GNNAnomalyDetector()
            
            # Initialize adaptive layer
            self.adaptive_layer = AdaptiveLayer(vector_store)
            
            # Initialize policy orchestration engine with all required components
            self.policy_engine = PolicyOrchestrationEngine(
                faiss_layer=self.faiss_layer,
                gnn_detector=self.gnn_detector,
                adaptive_layer=self.adaptive_layer
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def train_user_persona(self, persona: UserPersona, num_sessions: int = 10):
        """Train the system on a user persona's behavioral patterns"""
        logger.info(f"Training system on persona: {persona.name} ({persona.persona_type})")
        
        training_vectors = []
        
        for session_num in range(num_sessions):
            session_id = f"training_session_{session_num}"
            
            # Generate authentic behavioral data
            behavioral_data = persona.generate_behavioral_data(is_authentic=True)
            
            # Process data into vector
            try:
                # Format data properly for the processor
                formatted_data = {
                    "user_id": persona.user_id,
                    "session_id": session_id,
                    "logs": behavioral_data["events"]
                }
                vector_data = self.behavioral_processor.process_mobile_behavioral_data(
                    formatted_data
                )
                
                if vector_data is not None and len(vector_data) > 0:
                    # Store training vector
                    vector_id = await ml_db.store_behavioral_vector(
                        user_id=persona.user_id,
                        session_id=session_id,
                        vector_data=vector_data.tolist() if hasattr(vector_data, 'tolist') else vector_data,
                        confidence_score=1.0,
                        feature_source=f"training_{persona.persona_type}"
                    )
                    
                    training_vectors.append({
                        "session": session_num,
                        "vector_id": vector_id,
                        "vector_data": vector_data,
                        "non_zero_elements": np.count_nonzero(vector_data)
                    })
                    
                    logger.info(f"Session {session_num}: Stored vector {vector_id}, "
                              f"non-zero elements: {np.count_nonzero(vector_data)}")
                    
                else:
                    logger.warning(f"Session {session_num}: Empty vector generated")
                    
            except Exception as e:
                logger.error(f"Session {session_num}: Error processing data: {e}")
        
        # Update persona state
        persona.session_count = num_sessions
        persona.learning_phase = "gradual_risk" if num_sessions >= 5 else "learning"
        
        # Create baseline vector if enough training data
        if len(training_vectors) >= 5:
            baseline_vector = np.mean([tv["vector_data"] for tv in training_vectors], axis=0)
            
            baseline_id = await ml_db.store_behavioral_vector(
                user_id=persona.user_id,
                session_id="baseline",
                vector_data=baseline_vector.tolist(),
                confidence_score=1.0,
                feature_source="baseline_creation"
            )
            
            logger.info(f"Created baseline vector {baseline_id} for {persona.name}")
        
        self.test_results["training_results"][persona.name] = {
            "sessions_trained": num_sessions,
            "vectors_stored": len(training_vectors),
            "learning_phase": persona.learning_phase,
            "training_vectors": training_vectors
        }
        
        return training_vectors
    
    async def test_authentic_user_session(self, persona: UserPersona):
        """Test authentic user session and analyze similarity"""
        logger.info(f"Testing authentic session for {persona.name}")
        
        session_id = f"auth_test_{int(datetime.now().timestamp())}"
        
        # Generate authentic behavioral data
        behavioral_data = persona.generate_behavioral_data(is_authentic=True)
        
        # Process into vector
        formatted_data = {
            "user_id": persona.user_id,
            "session_id": session_id,
            "logs": behavioral_data["events"]
        }
        current_vector = self.behavioral_processor.process_mobile_behavioral_data(
            formatted_data
        )
        
        if current_vector is None or len(current_vector) == 0:
            logger.error("Failed to generate current vector")
            return None
        
        # Get historical vectors for similarity calculation
        historical_vectors = await ml_db.get_user_vectors(persona.user_id, limit=10)
        
        if not historical_vectors:
            logger.warning(f"No historical vectors found for {persona.user_id}")
            return {
                "decision": "learn",
                "similarity_score": 0.0,
                "reason": "no_baseline_data"
            }
        
        # Calculate FAISS similarity
        similarities = []
        for hist_vector in historical_vectors:
            if 'vector_data' in hist_vector and hist_vector['vector_data']:
                hist_data = np.array(hist_vector['vector_data'])
                similarity = self._calculate_cosine_similarity(current_vector, hist_data)
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        
        # Make authentication decision
        decision, confidence = self._make_similarity_decision(avg_similarity, max_similarity, persona)
        
        # Store test vector
        test_vector_id = await ml_db.store_behavioral_vector(
            user_id=persona.user_id,
            session_id=session_id,
            vector_data=current_vector.tolist(),
            confidence_score=confidence,
            feature_source="authentic_test"
        )
        
        result = {
            "persona": persona.name,
            "session_id": session_id,
            "vector_id": test_vector_id,
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "decision": decision,
            "confidence": confidence,
            "historical_count": len(historical_vectors),
            "similarities": similarities
        }
        
        logger.info(f"Authentic test - {persona.name}: {decision} "
                   f"(avg_sim: {avg_similarity:.3f}, max_sim: {max_similarity:.3f})")
        
        return result
    
    async def test_attack_scenarios(self, persona: UserPersona):
        """Test various attack scenarios against trained persona"""
        logger.info(f"Testing attack scenarios against {persona.name}")
        
        attack_types = [
            "bot_attack",
            "device_takeover", 
            "social_engineering",
            "credential_stuffing",
            "account_takeover"
        ]
        
        attack_results = {}
        
        for attack_type in attack_types:
            logger.info(f"Testing {attack_type} against {persona.name}")
            
            session_id = f"attack_{attack_type}_{int(datetime.now().timestamp())}"
            
            # Generate attack data
            if attack_type in ["credential_stuffing", "account_takeover"]:
                attacker_persona = UserPersona("attacker", "attacker")
                attack_data = attacker_persona.generate_behavioral_data(attack_type=attack_type)
            else:
                attack_data = persona.generate_behavioral_data(is_authentic=False, attack_type=attack_type)
            
            # Process attack vector
            formatted_attack_data = {
                "user_id": persona.user_id,
                "session_id": session_id,
                "logs": attack_data["events"]
            }
            attack_vector = self.behavioral_processor.process_mobile_behavioral_data(
                formatted_attack_data
            )
            
            if attack_vector is None:
                logger.warning(f"Failed to generate vector for {attack_type}")
                continue
            
            # Get user's baseline for comparison
            historical_vectors = await ml_db.get_user_vectors(persona.user_id, limit=10)
            
            if historical_vectors:
                # Calculate similarities
                similarities = []
                for hist_vector in historical_vectors:
                    if 'vector_data' in hist_vector and hist_vector['vector_data']:
                        hist_data = np.array(hist_vector['vector_data'])
                        similarity = self._calculate_cosine_similarity(attack_vector, hist_data)
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities) if similarities else 0.0
                
                # Decision should be BLOCK for low similarity
                decision, confidence = self._make_similarity_decision(avg_similarity, max(similarities), persona)
                
                attack_results[attack_type] = {
                    "avg_similarity": avg_similarity,
                    "max_similarity": max(similarities) if similarities else 0.0,
                    "decision": decision,
                    "confidence": confidence,
                    "detected": decision in ["challenge", "block"],
                    "similarities": similarities
                }
                
                logger.info(f"{attack_type}: {decision} (similarity: {avg_similarity:.3f}) "
                           f"- {'DETECTED' if decision in ['challenge', 'block'] else 'MISSED'}")
        
        return attack_results
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Ensure vectors are same length
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _make_similarity_decision(self, avg_similarity: float, max_similarity: float, persona: UserPersona) -> tuple:
        """Make authentication decision based on similarity scores"""
        
        # Phase-aware thresholds
        if persona.learning_phase == "learning":
            return "learn", 0.5
        elif persona.learning_phase == "gradual_risk":
            if avg_similarity >= 0.7:
                return "allow", 0.8
            elif avg_similarity >= 0.5:
                return "challenge", 0.6
            else:
                return "block", 0.9
        else:  # full_auth
            if avg_similarity >= 0.8:
                return "allow", 0.9
            elif avg_similarity >= 0.6:
                return "challenge", 0.7
            else:
                return "block", 0.95
    
    async def run_comprehensive_test(self):
        """Run comprehensive testing with all personas"""
        logger.info("Starting comprehensive persona-based testing")
        
        # Initialize components
        if not await self.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        # Create test personas
        personas = [
            UserPersona("Alice", "tech_savvy_young"),
            UserPersona("Bob", "elderly_careful"),
            UserPersona("Carol", "business_professional")
        ]
        
        # Train each persona
        for persona in personas:
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING PERSONA: {persona.name.upper()}")
            logger.info(f"{'='*60}")
            
            training_results = await self.train_user_persona(persona, num_sessions=8)
            
            if not training_results:
                logger.error(f"Failed to train {persona.name}")
                continue
        
        # Test authentic sessions
        logger.info(f"\n{'='*60}")
        logger.info("TESTING AUTHENTIC SESSIONS")
        logger.info(f"{'='*60}")
        
        for persona in personas:
            auth_result = await self.test_authentic_user_session(persona)
            if auth_result:
                self.test_results["authentication_tests"][persona.name] = auth_result
        
        # Test attack scenarios
        logger.info(f"\n{'='*60}")
        logger.info("TESTING ATTACK SCENARIOS")
        logger.info(f"{'='*60}")
        
        for persona in personas:
            attack_results = await self.test_attack_scenarios(persona)
            self.test_results["attack_detection"][persona.name] = attack_results
        
        # Generate analysis
        await self.analyze_results()
        
        return True
    
    async def analyze_results(self):
        """Analyze test results and identify improvements needed"""
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE ANALYSIS")
        logger.info(f"{'='*60}")
        
        # Analyze authentication accuracy
        auth_analysis = {
            "successful_authentications": 0,
            "failed_authentications": 0,
            "average_similarity": 0,
            "persona_performance": {}
        }
        
        for persona_name, result in self.test_results["authentication_tests"].items():
            decision = result.get("decision", "unknown")
            similarity = result.get("avg_similarity", 0)
            
            if decision == "allow":
                auth_analysis["successful_authentications"] += 1
            else:
                auth_analysis["failed_authentications"] += 1
            
            auth_analysis["persona_performance"][persona_name] = {
                "decision": decision,
                "similarity": similarity,
                "performance": "good" if decision == "allow" else "needs_improvement"
            }
        
        # Analyze attack detection
        attack_analysis = {
            "total_attacks": 0,
            "detected_attacks": 0,
            "missed_attacks": 0,
            "detection_by_type": {}
        }
        
        for persona_name, attacks in self.test_results["attack_detection"].items():
            for attack_type, attack_result in attacks.items():
                attack_analysis["total_attacks"] += 1
                
                if attack_result.get("detected", False):
                    attack_analysis["detected_attacks"] += 1
                else:
                    attack_analysis["missed_attacks"] += 1
                
                attack_analysis["detection_by_type"][attack_type] = attack_result
        
        # Calculate detection rate
        detection_rate = (attack_analysis["detected_attacks"] / attack_analysis["total_attacks"] * 100) if attack_analysis["total_attacks"] > 0 else 0
        
        logger.info(f"\nAUTHENTICATION ANALYSIS:")
        logger.info(f"Successful authentications: {auth_analysis['successful_authentications']}")
        logger.info(f"Failed authentications: {auth_analysis['failed_authentications']}")
        
        logger.info(f"\nATTACK DETECTION ANALYSIS:")
        logger.info(f"Detection rate: {detection_rate:.1f}%")
        logger.info(f"Attacks detected: {attack_analysis['detected_attacks']}/{attack_analysis['total_attacks']}")
        
        # Detailed persona analysis
        for persona_name, performance in auth_analysis["persona_performance"].items():
            logger.info(f"\n{persona_name.upper()} PERFORMANCE:")
            logger.info(f"  Authentication: {performance['decision']} (similarity: {performance['similarity']:.3f})")
            logger.info(f"  Status: {performance['performance']}")
            
            if persona_name in self.test_results["attack_detection"]:
                persona_attacks = self.test_results["attack_detection"][persona_name]
                detected = sum(1 for attack in persona_attacks.values() if attack.get("detected", False))
                total = len(persona_attacks)
                detection_percentage = (detected/total*100) if total > 0 else 0
                logger.info(f"  Attack detection: {detected}/{total} ({detection_percentage:.1f}%)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"persona_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "test_results": self.test_results,
                "analysis": {
                    "authentication": auth_analysis,
                    "attack_detection": attack_analysis,
                    "detection_rate": detection_rate
                },
                "timestamp": timestamp
            }, f, indent=2, default=str)
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Recommendations
        logger.info(f"\n{'='*60}")
        logger.info("RECOMMENDATIONS FOR IMPROVEMENT")
        logger.info(f"{'='*60}")
        
        if auth_analysis["failed_authentications"] > 0:
            logger.info("❌ Some authentic users are being rejected")
            logger.info("   Recommendation: Lower similarity thresholds for gradual_risk phase")
        
        if detection_rate < 80:
            logger.info("❌ Attack detection rate is below 80%")
            logger.info("   Recommendation: Implement additional behavioral features")
            logger.info("   Recommendation: Add temporal analysis layers")
        
        if detection_rate >= 90:
            logger.info("✅ Excellent attack detection rate!")
        
        return auth_analysis, attack_analysis

async def main():
    """Main testing function"""
    tester = PersonaBasedTester()
    success = await tester.run_comprehensive_test()
    
    if success:
        logger.info("\n🎉 COMPREHENSIVE TESTING COMPLETED SUCCESSFULLY!")
        logger.info("Check the generated results file for detailed analysis.")
    else:
        logger.error("\n❌ Testing failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
