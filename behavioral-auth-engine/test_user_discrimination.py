#!/usr/bin/env python3
"""
Test the discriminative power of FAISS behavioral authentication
This evaluates how well the system can distinguish between different users
"""

import asyncio
import json
import uuid
import sys
import os
import numpy as np
from datetime import datetime
import random
from typing import Dict, List, Tuple

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
from src.core.enhanced_faiss_engine import EnhancedFAISSEngine
from src.core.ml_database import MLSupabaseClient

class UserDiscriminationTester:
    def __init__(self):
        self.db_client = MLSupabaseClient()
        self.processor = EnhancedBehavioralProcessor()
        self.faiss_engine = EnhancedFAISSEngine()
        self.test_users = []
        
    def generate_user_behavioral_pattern(self, user_id: str, pattern_type: str) -> dict:
        """Generate distinct behavioral patterns for different user types"""
        base_timestamp = datetime.utcnow().timestamp() * 1000
        logs = []
        
        # Define distinct behavioral patterns
        patterns = {
            "fast_precise": {
                "touch_pressure": (0.7, 0.9),
                "touch_speed": 80,
                "scroll_velocity": (200, 400),
                "accel_variance": 1.5,
                "gyro_variance": 0.3
            },
            "slow_gentle": {
                "touch_pressure": (0.2, 0.4),
                "touch_speed": 200,
                "scroll_velocity": (50, 150),
                "accel_variance": 0.8,
                "gyro_variance": 0.2
            },
            "aggressive": {
                "touch_pressure": (0.8, 1.0),
                "touch_speed": 50,
                "scroll_velocity": (300, 600),
                "accel_variance": 3.0,
                "gyro_variance": 0.8
            },
            "elderly": {
                "touch_pressure": (0.3, 0.5),
                "touch_speed": 300,
                "scroll_velocity": (30, 80),
                "accel_variance": 0.5,
                "gyro_variance": 0.1
            },
            "gamer": {
                "touch_pressure": (0.6, 0.8),
                "touch_speed": 40,
                "scroll_velocity": (400, 800),
                "accel_variance": 2.5,
                "gyro_variance": 0.6
            }
        }
        
        pattern = patterns.get(pattern_type, patterns["slow_gentle"])
        
        # Generate touch events with pattern characteristics
        for i in range(20):
            pressure_min, pressure_max = pattern["touch_pressure"]
            logs.append({
                "event_type": "touch",
                "timestamp": base_timestamp + i * pattern["touch_speed"],
                "data": {
                    "x": 200 + random.uniform(-80, 80),
                    "y": 400 + random.uniform(-100, 100),
                    "pressure": random.uniform(pressure_min, pressure_max),
                    "action": random.choice(["down", "move", "up"])
                }
            })
        
        # Generate accelerometer events
        for i in range(30):
            variance = pattern["accel_variance"]
            logs.append({
                "event_type": "accelerometer",
                "timestamp": base_timestamp + i * 60,
                "data": {
                    "x": random.uniform(-variance, variance),
                    "y": random.uniform(-variance, variance),
                    "z": 9.8 + random.uniform(-variance/2, variance/2)
                }
            })
        
        # Generate gyroscope events
        for i in range(30):
            variance = pattern["gyro_variance"]
            logs.append({
                "event_type": "gyroscope",
                "timestamp": base_timestamp + i * 60,
                "data": {
                    "x": random.uniform(-variance, variance),
                    "y": random.uniform(-variance, variance),
                    "z": random.uniform(-variance, variance)
                }
            })
        
        # Generate scroll events
        for i in range(12):
            vel_min, vel_max = pattern["scroll_velocity"]
            logs.append({
                "event_type": "scroll",
                "timestamp": base_timestamp + i * 250,
                "data": {
                    "delta_y": random.uniform(-150, 150),
                    "velocity": random.uniform(vel_min, vel_max)
                }
            })
        
        logs.sort(key=lambda x: x['timestamp'])
        
        return {
            "user_id": user_id,
            "session_id": f"session_{user_id}_{random.randint(1000, 9999)}",
            "logs": logs
        }
    
    async def create_user_profiles(self, num_users: int = 5) -> List[Dict]:
        """Create behavioral profiles for multiple distinct users"""
        pattern_types = ["fast_precise", "slow_gentle", "aggressive", "elderly", "gamer"]
        users = []
        
        for i in range(num_users):
            user_id = str(uuid.uuid4())
            pattern_type = pattern_types[i % len(pattern_types)]
            
            print(f"\n🔄 Creating profile for User {i+1} ({pattern_type} pattern)...")
            
            # Create multiple sessions for each user to build a profile
            user_vectors = []
            for session in range(8):  # 8 sessions per user
                behavioral_data = self.generate_user_behavioral_pattern(user_id, pattern_type)
                
                try:
                    # Process behavioral data to get vector
                    result = await self.faiss_engine.process_mobile_behavioral_data(
                        user_id=user_id,
                        session_id=behavioral_data["session_id"],
                        behavioral_data=behavioral_data
                    )
                    
                    if hasattr(result, 'session_vector') and result.session_vector:
                        user_vectors.append(result.session_vector)
                        print(f"  Session {session+1}: Vector generated ({len([v for v in result.session_vector if v != 0])}/90 non-zero)")
                    
                except Exception as e:
                    print(f"  Session {session+1}: Error - {e}")
            
            if user_vectors:
                # Calculate average vector for this user
                avg_vector = np.mean(user_vectors, axis=0)
                users.append({
                    "user_id": user_id,
                    "pattern_type": pattern_type,
                    "vectors": user_vectors,
                    "average_vector": avg_vector.tolist(),
                    "session_count": len(user_vectors)
                })
                print(f"  ✅ Profile created with {len(user_vectors)} sessions")
            else:
                print(f"  ❌ Failed to create profile")
        
        return users
    
    def calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def analyze_user_discrimination(self, users: List[Dict]) -> Dict:
        """Analyze how well the system can discriminate between users"""
        print(f"\n🔍 Analyzing User Discrimination...")
        
        # Calculate intra-user similarities (same user, different sessions)
        intra_user_similarities = []
        for user in users:
            vectors = user["vectors"]
            if len(vectors) > 1:
                similarities = []
                for i in range(len(vectors)):
                    for j in range(i+1, len(vectors)):
                        sim = self.calculate_vector_similarity(vectors[i], vectors[j])
                        similarities.append(sim)
                
                if similarities:
                    avg_intra_sim = np.mean(similarities)
                    intra_user_similarities.append(avg_intra_sim)
                    print(f"  {user['pattern_type']}: Intra-user similarity = {avg_intra_sim:.4f}")
        
        # Calculate inter-user similarities (different users)
        inter_user_similarities = []
        for i in range(len(users)):
            for j in range(i+1, len(users)):
                sim = self.calculate_vector_similarity(
                    users[i]["average_vector"], 
                    users[j]["average_vector"]
                )
                inter_user_similarities.append(sim)
                print(f"  {users[i]['pattern_type']} vs {users[j]['pattern_type']}: {sim:.4f}")
        
        # Calculate discrimination metrics
        avg_intra_sim = np.mean(intra_user_similarities) if intra_user_similarities else 0
        avg_inter_sim = np.mean(inter_user_similarities) if inter_user_similarities else 0
        discrimination_ratio = avg_intra_sim / avg_inter_sim if avg_inter_sim > 0 else float('inf')
        
        return {
            "intra_user_similarities": intra_user_similarities,
            "inter_user_similarities": inter_user_similarities,
            "avg_intra_similarity": avg_intra_sim,
            "avg_inter_similarity": avg_inter_sim,
            "discrimination_ratio": discrimination_ratio,
            "separation_gap": avg_intra_sim - avg_inter_sim
        }
    
    async def test_authentication_accuracy(self, users: List[Dict]) -> Dict:
        """Test authentication accuracy using FAISS similarity search"""
        print(f"\n🎯 Testing Authentication Accuracy...")
        
        results = {
            "correct_authentications": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "total_tests": 0,
            "detailed_results": []
        }
        
        for test_user in users:
            print(f"\nTesting user: {test_user['pattern_type']}")
            
            # Generate new test session for this user
            test_data = self.generate_user_behavioral_pattern(
                test_user["user_id"], 
                test_user["pattern_type"]
            )
            
            try:
                # Process the test session
                test_result = await self.faiss_engine.process_mobile_behavioral_data(
                    user_id=test_user["user_id"],
                    session_id=test_data["session_id"],
                    behavioral_data=test_data
                )
                
                if hasattr(test_result, 'session_vector') and test_result.session_vector:
                    test_vector = test_result.session_vector
                    
                    # Calculate similarities to all user profiles
                    similarities = []
                    for profile_user in users:
                        sim = self.calculate_vector_similarity(
                            test_vector, 
                            profile_user["average_vector"]
                        )
                        similarities.append({
                            "user_id": profile_user["user_id"],
                            "pattern_type": profile_user["pattern_type"],
                            "similarity": sim
                        })
                    
                    # Sort by similarity (highest first)
                    similarities.sort(key=lambda x: x["similarity"], reverse=True)
                    
                    # Check if the top match is correct
                    top_match = similarities[0]
                    is_correct = top_match["user_id"] == test_user["user_id"]
                    
                    if is_correct:
                        results["correct_authentications"] += 1
                        print(f"  ✅ Correct match: {top_match['similarity']:.4f}")
                    else:
                        results["false_positives"] += 1
                        print(f"  ❌ Wrong match: {top_match['pattern_type']} ({top_match['similarity']:.4f})")
                        print(f"     Correct similarity: {similarities[next(i for i, s in enumerate(similarities) if s['user_id'] == test_user['user_id'])]['similarity']:.4f}")
                    
                    results["total_tests"] += 1
                    results["detailed_results"].append({
                        "test_user": test_user["pattern_type"],
                        "correct": is_correct,
                        "similarities": similarities[:3]  # Top 3 matches
                    })
            
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
        
        # Calculate accuracy metrics
        if results["total_tests"] > 0:
            results["accuracy"] = results["correct_authentications"] / results["total_tests"]
            results["false_positive_rate"] = results["false_positives"] / results["total_tests"]
        
        return results
    
    async def cleanup_test_data(self):
        """Clean up test data"""
        print(f"\n🧹 Cleaning up test data...")
        try:
            for user in self.test_users:
                # Delete test vectors
                self.db_client.supabase.table('enhanced_behavioral_vectors')\
                    .delete()\
                    .eq('user_id', user["user_id"])\
                    .execute()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def main():
    """Run user discrimination analysis"""
    tester = UserDiscriminationTester()
    
    try:
        print("🧪 BEHAVIORAL AUTHENTICATION DISCRIMINATION TEST")
        print("=" * 60)
        
        # Create user profiles with distinct behavioral patterns
        users = await tester.create_user_profiles(num_users=5)
        tester.test_users = users
        
        if len(users) < 2:
            print("❌ Not enough user profiles created for discrimination analysis")
            return
        
        # Analyze discrimination capability
        discrimination_analysis = tester.analyze_user_discrimination(users)
        
        # Test authentication accuracy
        auth_accuracy = await tester.test_authentication_accuracy(users)
        
        # Print comprehensive results
        print(f"\n📊 DISCRIMINATION ANALYSIS RESULTS")
        print("=" * 40)
        print(f"Average Intra-user Similarity: {discrimination_analysis['avg_intra_similarity']:.4f}")
        print(f"Average Inter-user Similarity: {discrimination_analysis['avg_inter_similarity']:.4f}")
        print(f"Discrimination Ratio: {discrimination_analysis['discrimination_ratio']:.2f}")
        print(f"Separation Gap: {discrimination_analysis['separation_gap']:.4f}")
        
        print(f"\n🎯 AUTHENTICATION ACCURACY")
        print("=" * 30)
        print(f"Correct Authentications: {auth_accuracy['correct_authentications']}/{auth_accuracy['total_tests']}")
        print(f"Accuracy: {auth_accuracy.get('accuracy', 0):.2%}")
        print(f"False Positive Rate: {auth_accuracy.get('false_positive_rate', 0):.2%}")
        
        # Interpretation
        print(f"\n🤔 SYSTEM EVALUATION")
        print("=" * 25)
        
        discrimination_ratio = discrimination_analysis['discrimination_ratio']
        accuracy = auth_accuracy.get('accuracy', 0)
        
        if discrimination_ratio > 2.0 and accuracy > 0.8:
            print("🎉 EXCELLENT: System shows strong user discrimination")
            print("   - High intra-user consistency")
            print("   - Good inter-user separation")
            print("   - High authentication accuracy")
        elif discrimination_ratio > 1.5 and accuracy > 0.6:
            print("✅ GOOD: System shows reasonable user discrimination")
            print("   - Moderate behavioral consistency")
            print("   - Acceptable user separation")
        elif discrimination_ratio > 1.0:
            print("⚠️ MODERATE: System shows limited discrimination")
            print("   - Some behavioral patterns detectable")
            print("   - May need more training data or feature engineering")
        else:
            print("❌ POOR: System struggles with user discrimination")
            print("   - Behavioral patterns not sufficiently distinct")
            print("   - Needs significant improvement")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"discrimination_analysis_{timestamp}.json"
        
        full_results = {
            "discrimination_analysis": discrimination_analysis,
            "authentication_accuracy": auth_accuracy,
            "user_profiles": [
                {
                    "pattern_type": u["pattern_type"],
                    "session_count": u["session_count"],
                    "user_id": u["user_id"]
                } for u in users
            ],
            "test_timestamp": timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.cleanup_test_data()

if __name__ == "__main__":
    asyncio.run(main())
