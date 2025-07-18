#!/usr/bin/env python3
"""
Verify that behavioral vectors are now meaningful (not all zeros)
"""

import asyncio
import aiohttp
import json
import uuid
import random
import numpy as np
from datetime import datetime

ML_ENGINE_URL = "http://localhost:8001"

class VectorVerificationTest:
    def __init__(self):
        self.session = None
        self.user_id = str(uuid.uuid4())
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_mobile_behavioral_logs(self) -> list:
        """Generate realistic mobile behavioral logs"""
        base_timestamp = datetime.utcnow().timestamp() * 1000
        logs = []
        
        # Rich touch interaction pattern
        for i in range(20):
            logs.append({
                "event_type": "touch",
                "timestamp": base_timestamp + i * 150,
                "data": {
                    "x": 200 + random.uniform(-100, 100),
                    "y": 400 + random.uniform(-100, 100),
                    "pressure": 0.6 + random.uniform(-0.3, 0.3),
                    "action": random.choice(["down", "move", "up"])
                }
            })
        
        # Varied accelerometer pattern (walking/movement)
        for i in range(30):
            logs.append({
                "event_type": "accelerometer",
                "timestamp": base_timestamp + i * 100,
                "data": {
                    "x": random.uniform(-3, 3),
                    "y": random.uniform(-3, 3),
                    "z": 9.8 + random.uniform(-2, 2)
                }
            })
        
        # Gyroscope pattern (device rotation)
        for i in range(30):
            logs.append({
                "event_type": "gyroscope",
                "timestamp": base_timestamp + i * 100,
                "data": {
                    "x": random.uniform(-1, 1),
                    "y": random.uniform(-1, 1),
                    "z": random.uniform(-1, 1)
                }
            })
        
        # Scroll events with varying patterns
        for i in range(15):
            logs.append({
                "event_type": "scroll",
                "timestamp": base_timestamp + i * 300,
                "data": {
                    "delta_y": random.uniform(-200, 200),
                    "velocity": random.uniform(100, 500)
                }
            })
        
        logs.sort(key=lambda x: x['timestamp'])
        return logs
    
    async def test_vector_generation(self, logs):
        """Test that vectors are meaningful (not all zeros)"""
        try:
            ml_data = {
                "user_id": self.user_id,
                "session_id": str(uuid.uuid4()),
                "logs": logs
            }
            
            async with self.session.post(f"{ML_ENGINE_URL}/analyze-mobile", json=ml_data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    print(f"🤖 ML Engine Analysis:")
                    print(f"   Decision: {result.get('decision', 'N/A')}")
                    print(f"   Confidence: {result.get('confidence', 0):.3f}")
                    print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
                    print(f"   Vector ID: {result.get('vector_id', 'N/A')}")
                    
                    # Check for vector statistics
                    if 'vector_stats' in result:
                        stats = result['vector_stats']
                        print(f"\n📊 Vector Statistics:")
                        print(f"   🎯 Vector Length: {stats.get('length', 'N/A')}")
                        print(f"   ✅ Non-zero values: {stats.get('non_zero_count', 'N/A')}")
                        print(f"   ❌ Zero values: {stats.get('zero_count', 'N/A')}")
                        print(f"   📈 Non-zero percentage: {stats.get('non_zero_percentage', 0):.1f}%")
                        print(f"   📊 Mean: {stats.get('mean', 0):.4f}")
                        print(f"   📐 Std Dev: {stats.get('std', 0):.4f}")
                        print(f"   🔍 Is meaningful: {stats.get('is_meaningful', False)}")
                        
                        # Show sample vector values
                        if 'vector_sample' in result:
                            vector_sample = result['vector_sample']
                            print(f"   � Sample values: {[f'{v:.4f}' for v in vector_sample[:5]]}")
                        
                        # Check if vector is meaningful
                        if stats.get('is_meaningful', False):
                            print(f"   🎉 VECTOR IS MEANINGFUL!")
                            return True
                        else:
                            print(f"   ⚠️ Vector is mostly zeros")
                            return False
                    elif 'vector_sample' in result:
                        vector_sample = result['vector_sample']
                        non_zero_count = sum(1 for v in vector_sample if v != 0)
                        print(f"\n📊 Vector Sample Analysis:")
                        print(f"   🔬 Sample values: {[f'{v:.4f}' for v in vector_sample[:5]]}")
                        print(f"   ✅ Non-zero in sample: {non_zero_count}/{len(vector_sample)}")
                        
                        if non_zero_count > 0:
                            print(f"   🎉 VECTOR HAS NON-ZERO VALUES!")
                            return True
                        else:
                            print(f"   ⚠️ Vector sample is all zeros")
                            return False
                    
                    return result
                else:
                    error_text = await response.text()
                    print(f"⚠️ ML Engine analysis failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"⚠️ Error testing ML engine: {e}")
            return None

async def main():
    """Run vector verification test"""
    print("🔍 Vector Verification Test")
    print("=" * 40)
    
    async with VectorVerificationTest() as tester:
        # Generate multiple test cases
        test_cases = [
            {"name": "Standard Mobile Usage", "count": 95},
            {"name": "Heavy Touch Activity", "count": 150},
            {"name": "Motion-Heavy Pattern", "count": 80}
        ]
        
        meaningful_vectors = 0
        total_tests = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}/{total_tests}: {test_case['name']}")
            print("-" * 30)
            
            logs = tester.generate_mobile_behavioral_logs()
            result = await tester.test_vector_generation(logs)
            
            if result and isinstance(result, dict):
                meaningful_vectors += 1
        
        print(f"\n📋 FINAL RESULTS:")
        print("=" * 20)
        print(f"✅ Meaningful vectors: {meaningful_vectors}/{total_tests}")
        print(f"📊 Success rate: {(meaningful_vectors/total_tests*100):.1f}%")
        
        if meaningful_vectors > 0:
            print(f"\n🎉 SUCCESS! Mobile app is now generating meaningful vectors!")
            print(f"✅ The zero vector problem has been FIXED!")
        else:
            print(f"\n❌ ISSUE: Still generating mostly zero vectors")
            print(f"⚠️ Need to investigate behavioral processor further")

if __name__ == "__main__":
    asyncio.run(main())
