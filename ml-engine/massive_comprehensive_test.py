#!/usr/bin/env python3
"""
🌟 MASSIVE COMPREHENSIVE BEHAVIORAL AUTHENTICATION TEST
=======================================================
Testing with vast data, multiple scenarios, edge cases, and extensive possibilities
"""

import requests
import json
import random
import time
import numpy as np
from typing import Dict, List, Any
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics

class MassiveTestSuite:
    def __init__(self):
        self.ml_engine_url = "http://localhost:8001/analyze-mobile"
        self.results = []
        self.test_counter = 0
        
    def generate_realistic_touch_events(self, user_type: str, count: int = 50) -> List[Dict]:
        """Generate realistic touch events based on user type"""
        touches = []
        
        if user_type == "steady_professional":
            base_pressure = 0.5
            pressure_var = 0.1
            duration_range = (100, 150)
            coord_variance = 10
        elif user_type == "elderly_cautious":
            base_pressure = 0.3
            pressure_var = 0.15
            duration_range = (150, 250)
            coord_variance = 20
        elif user_type == "young_gamer":
            base_pressure = 0.7
            pressure_var = 0.2
            duration_range = (50, 100)
            coord_variance = 5
        elif user_type == "nervous_tremor":
            base_pressure = 0.4
            pressure_var = 0.3
            duration_range = (80, 200)
            coord_variance = 30
        elif user_type == "automation_bot":
            base_pressure = 0.5
            pressure_var = 0.01  # Very consistent
            duration_range = (100, 105)  # Nearly identical
            coord_variance = 1  # Perfect precision
        elif user_type == "scraping_bot":
            base_pressure = 0.6
            pressure_var = 0.0  # Identical
            duration_range = (50, 50)  # Exact same
            coord_variance = 0  # Perfect identical
        else:
            # Random human
            base_pressure = random.uniform(0.2, 0.8)
            pressure_var = random.uniform(0.05, 0.25)
            duration_range = (random.randint(60, 80), random.randint(120, 200))
            coord_variance = random.randint(8, 25)
        
        base_x, base_y = random.randint(100, 400), random.randint(200, 600)
        
        for i in range(count):
            if user_type in ["automation_bot", "scraping_bot"]:
                # Bots have identical or near-identical patterns
                x = base_x + (i % 3)  # Minimal variation
                y = base_y + (i % 2)
                pressure = base_pressure
                duration = duration_range[0]
            else:
                # Humans have natural variation
                x = base_x + random.randint(-coord_variance, coord_variance)
                y = base_y + random.randint(-coord_variance, coord_variance)
                pressure = max(0.1, min(1.0, np.random.normal(base_pressure, pressure_var)))
                duration = random.randint(*duration_range)
            
            touches.append({
                "x": x,
                "y": y,
                "pressure": round(pressure, 3),
                "duration": duration,
                "timestamp": f"2024-01-01T10:00:{i:02d}"
            })
        
        return touches
    
    def generate_keystroke_patterns(self, user_type: str, count: int = 30) -> Dict:
        """Generate realistic keystroke patterns"""
        keys = "abcdefghijklmnopqrstuvwxyz0123456789"
        keystrokes = []
        typing_rhythm = []
        intervals = []
        
        if user_type == "steady_professional":
            base_dwell = 120
            dwell_var = 20
            rhythm_base = 95
            rhythm_var = 15
        elif user_type == "elderly_cautious":
            base_dwell = 180
            dwell_var = 40
            rhythm_base = 150
            rhythm_var = 30
        elif user_type == "young_gamer":
            base_dwell = 80
            dwell_var = 15
            rhythm_base = 65
            rhythm_var = 10
        elif user_type == "nervous_tremor":
            base_dwell = 140
            dwell_var = 60
            rhythm_base = 120
            rhythm_var = 50
        elif user_type in ["automation_bot", "scraping_bot"]:
            base_dwell = 100
            dwell_var = 1  # Nearly identical
            rhythm_base = 85
            rhythm_var = 2  # Very consistent
        else:
            base_dwell = random.randint(90, 160)
            dwell_var = random.randint(15, 40)
            rhythm_base = random.randint(70, 130)
            rhythm_var = random.randint(10, 35)
        
        for i in range(count):
            key = random.choice(keys)
            
            if user_type in ["automation_bot", "scraping_bot"]:
                dwell_time = base_dwell
                pressure = 0.6  # Consistent
                rhythm = rhythm_base
                interval = 0.1  # Consistent
            else:
                dwell_time = max(50, int(np.random.normal(base_dwell, dwell_var)))
                pressure = random.uniform(0.3, 0.8)
                rhythm = max(30, int(np.random.normal(rhythm_base, rhythm_var)))
                interval = random.uniform(0.08, 0.25)
            
            keystrokes.append({
                "key": key,
                "dwell_time": dwell_time,
                "pressure": round(pressure, 3)
            })
            typing_rhythm.append(rhythm)
            intervals.append(round(interval, 3))
        
        return {
            "keystrokes": keystrokes,
            "typing_rhythm": typing_rhythm,
            "inter_key_intervals": intervals
        }
    
    def generate_sensor_data(self, user_type: str) -> Dict:
        """Generate realistic sensor data"""
        if user_type == "steady_professional":
            accel_base = [0.02, 0.15, 9.78]
            gyro_base = [0.001, 0.002, 0.0015]
            variance = 0.05
        elif user_type == "elderly_cautious":
            accel_base = [0.05, 0.12, 9.75]
            gyro_base = [0.003, 0.004, 0.003]
            variance = 0.1
        elif user_type == "young_gamer":
            accel_base = [0.1, 0.25, 9.82]
            gyro_base = [0.008, 0.012, 0.01]
            variance = 0.15
        elif user_type == "nervous_tremor":
            accel_base = [0.08, 0.3, 9.7]
            gyro_base = [0.015, 0.02, 0.018]
            variance = 0.25
        elif user_type in ["automation_bot", "scraping_bot"]:
            accel_base = [0.02, 0.15, 9.81]
            gyro_base = [0.001, 0.002, 0.0015]
            variance = 0.001  # Nearly identical
        else:
            accel_base = [random.uniform(-0.1, 0.1), random.uniform(0.1, 0.3), random.uniform(9.7, 9.85)]
            gyro_base = [random.uniform(0, 0.02), random.uniform(0, 0.025), random.uniform(0, 0.02)]
            variance = random.uniform(0.03, 0.2)
        
        return {
            "accelerometer": {
                "x": round(np.random.normal(accel_base[0], variance), 4),
                "y": round(np.random.normal(accel_base[1], variance), 4),
                "z": round(np.random.normal(accel_base[2], variance), 4)
            },
            "gyroscope": {
                "x": round(np.random.normal(gyro_base[0], variance), 4),
                "y": round(np.random.normal(gyro_base[1], variance), 4),
                "z": round(np.random.normal(gyro_base[2], variance), 4)
            }
        }
    
    def generate_massive_behavioral_data(self, user_type: str, session_count: int = 5) -> List[Dict]:
        """Generate massive behavioral data for a user type"""
        all_logs = []
        
        for session in range(session_count):
            # Generate 20-40 touch sequences per session
            touch_sequences = random.randint(20, 40)
            for i in range(touch_sequences):
                touch_events = self.generate_realistic_touch_events(user_type, random.randint(5, 15))
                sensor_data = self.generate_sensor_data(user_type)
                
                all_logs.append({
                    "timestamp": f"2024-01-01T{10+session}:{i:02d}:00",
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": touch_events,
                        **sensor_data
                    }
                })
            
            # Generate 10-20 keystroke sequences per session
            keystroke_sequences = random.randint(10, 20)
            for i in range(keystroke_sequences):
                keystroke_data = self.generate_keystroke_patterns(user_type, random.randint(10, 25))
                
                all_logs.append({
                    "timestamp": f"2024-01-01T{10+session}:{30+i:02d}:00",
                    "event_type": "keystroke_sequence",
                    "data": keystroke_data
                })
        
        return all_logs
    
    def test_user_scenario(self, user_id: str, user_type: str, sessions: int = 5) -> Dict:
        """Test a complete user scenario with multiple sessions"""
        print(f"\n🧪 TESTING USER: {user_id} ({user_type})")
        print("=" * 60)
        
        session_results = []
        
        for session_num in range(1, sessions + 1):
            session_id = f"{user_id}_session_{session_num}"
            print(f"📊 Session {session_num}/{sessions}: {session_id}")
            
            # Generate massive behavioral data
            behavioral_logs = self.generate_massive_behavioral_data(user_type, 1)
            
            test_data = {
                "user_id": user_id,
                "session_id": session_id,
                "logs": behavioral_logs[:100]  # Limit to prevent timeout
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    self.ml_engine_url,
                    json=test_data,
                    timeout=60
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    session_result = {
                        "session_id": session_id,
                        "user_type": user_type,
                        "status": "success",
                        "decision": result.get("decision"),
                        "risk_score": result.get("risk_score"),
                        "confidence": result.get("confidence"),
                        "vector_stats": result.get("vector_stats", {}),
                        "gnn_analysis": result.get("gnn_analysis", {}),
                        "processing_time": response_time,
                        "behavioral_data_count": len(behavioral_logs)
                    }
                    
                    print(f"✅ Decision: {result.get('decision')} | Risk: {result.get('risk_score'):.4f} | Confidence: {result.get('confidence'):.2f}")
                    print(f"📈 Vector Quality: {result.get('vector_stats', {}).get('non_zero_count', 0)}/90 | Processing: {response_time:.2f}s")
                    
                else:
                    session_result = {
                        "session_id": session_id,
                        "user_type": user_type,
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {response.text[:200]}"
                    }
                    print(f"❌ Error: HTTP {response.status_code}")
                
                session_results.append(session_result)
                
            except Exception as e:
                session_result = {
                    "session_id": session_id,
                    "user_type": user_type,
                    "status": "exception",
                    "error": str(e)
                }
                session_results.append(session_result)
                print(f"❌ Exception: {e}")
        
        return {
            "user_id": user_id,
            "user_type": user_type,
            "sessions": session_results
        }
    
    def run_stress_test(self, concurrent_users: int = 5):
        """Run stress test with concurrent users"""
        print(f"\n🚀 STRESS TEST: {concurrent_users} Concurrent Users")
        print("=" * 60)
        
        user_scenarios = [
            ("stress_user_1", "steady_professional"),
            ("stress_user_2", "elderly_cautious"),
            ("stress_user_3", "young_gamer"),
            ("stress_user_4", "automation_bot"),
            ("stress_user_5", "scraping_bot"),
        ]
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for user_id, user_type in user_scenarios:
                future = executor.submit(self.test_user_scenario, user_id, user_type, 3)
                futures.append(future)
            
            stress_results = []
            for future in futures:
                try:
                    result = future.result(timeout=180)  # 3 minute timeout
                    stress_results.append(result)
                except Exception as e:
                    print(f"❌ Stress test error: {e}")
        
        return stress_results
    
    def analyze_comprehensive_results(self, all_results: List[Dict]):
        """Analyze all test results comprehensively"""
        print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 80)
        
        # Categorize results
        human_results = []
        bot_results = []
        
        for user_result in all_results:
            user_type = user_result.get("user_type", "")
            if "bot" in user_type:
                bot_results.extend(user_result.get("sessions", []))
            else:
                human_results.extend(user_result.get("sessions", []))
        
        # Analyze human performance
        if human_results:
            human_risks = [s.get("risk_score", 0) for s in human_results if s.get("status") == "success"]
            human_confidences = [s.get("confidence", 0) for s in human_results if s.get("status") == "success"]
            human_vectors = [s.get("vector_stats", {}).get("non_zero_count", 0) for s in human_results if s.get("status") == "success"]
            
            print(f"👥 HUMAN ANALYSIS ({len(human_results)} sessions):")
            print(f"   Risk Scores: min={min(human_risks):.4f}, max={max(human_risks):.4f}, avg={statistics.mean(human_risks):.4f}")
            print(f"   Confidence: min={min(human_confidences):.2f}, max={max(human_confidences):.2f}, avg={statistics.mean(human_confidences):.2f}")
            print(f"   Vector Quality: min={min(human_vectors)}, max={max(human_vectors)}, avg={statistics.mean(human_vectors):.1f}")
        
        # Analyze bot performance
        if bot_results:
            bot_risks = [s.get("risk_score", 0) for s in bot_results if s.get("status") == "success"]
            bot_confidences = [s.get("confidence", 0) for s in bot_results if s.get("status") == "success"]
            bot_vectors = [s.get("vector_stats", {}).get("non_zero_count", 0) for s in bot_results if s.get("status") == "success"]
            
            print(f"🤖 BOT ANALYSIS ({len(bot_results)} sessions):")
            print(f"   Risk Scores: min={min(bot_risks):.4f}, max={max(bot_risks):.4f}, avg={statistics.mean(bot_risks):.4f}")
            print(f"   Confidence: min={min(bot_confidences):.2f}, max={max(bot_confidences):.2f}, avg={statistics.mean(bot_confidences):.2f}")
            print(f"   Vector Quality: min={min(bot_vectors)}, max={max(bot_vectors)}, avg={statistics.mean(bot_vectors):.1f}")
        
        # Performance analysis
        all_sessions = []
        for user_result in all_results:
            all_sessions.extend(user_result.get("sessions", []))
        
        processing_times = [s.get("processing_time", 0) for s in all_sessions if s.get("status") == "success"]
        if processing_times:
            print(f"\n⚡ PERFORMANCE ANALYSIS:")
            print(f"   Processing Times: min={min(processing_times):.2f}s, max={max(processing_times):.2f}s, avg={statistics.mean(processing_times):.2f}s")
            print(f"   Success Rate: {len([s for s in all_sessions if s.get('status') == 'success'])}/{len(all_sessions)} ({100*len([s for s in all_sessions if s.get('status') == 'success'])/len(all_sessions):.1f}%)")
        
        return {
            "human_sessions": len(human_results),
            "bot_sessions": len(bot_results),
            "total_sessions": len(all_sessions),
            "success_rate": len([s for s in all_sessions if s.get('status') == 'success']) / len(all_sessions) if all_sessions else 0
        }
    
    def run_massive_test(self):
        """Run the complete massive test suite"""
        print("🌟 MASSIVE COMPREHENSIVE BEHAVIORAL AUTHENTICATION TEST")
        print("=" * 80)
        print("Testing with vast data, multiple scenarios, edge cases, and extensive possibilities")
        print("=" * 80)
        
        # Define comprehensive test scenarios
        test_scenarios = [
            # Professional Users
            ("corporate_executive", "steady_professional"),
            ("finance_analyst", "steady_professional"),
            ("project_manager", "steady_professional"),
            
            # Elderly/Cautious Users
            ("senior_citizen_1", "elderly_cautious"),
            ("senior_citizen_2", "elderly_cautious"),
            ("cautious_user", "elderly_cautious"),
            
            # Young/Active Users
            ("mobile_gamer_1", "young_gamer"),
            ("mobile_gamer_2", "young_gamer"),
            ("social_media_user", "young_gamer"),
            
            # Special Cases
            ("tremor_user", "nervous_tremor"),
            ("accessibility_user", "nervous_tremor"),
            
            # Bot Scenarios
            ("automation_script_1", "automation_bot"),
            ("automation_script_2", "automation_bot"),
            ("scraping_bot_1", "scraping_bot"),
            ("scraping_bot_2", "scraping_bot"),
            ("advanced_bot", "automation_bot"),
        ]
        
        all_results = []
        
        print(f"\n🎯 PHASE 1: COMPREHENSIVE USER TESTING ({len(test_scenarios)} users)")
        print("=" * 60)
        
        for user_id, user_type in test_scenarios:
            try:
                result = self.test_user_scenario(user_id, user_type, sessions=7)  # 7 sessions per user
                all_results.append(result)
                time.sleep(0.5)  # Small delay between users
            except Exception as e:
                print(f"❌ Failed to test {user_id}: {e}")
        
        print(f"\n🚀 PHASE 2: STRESS TESTING")
        print("=" * 60)
        
        try:
            stress_results = self.run_stress_test(5)
            all_results.extend(stress_results)
        except Exception as e:
            print(f"❌ Stress test failed: {e}")
        
        print(f"\n📈 PHASE 3: COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        analysis = self.analyze_comprehensive_results(all_results)
        
        print(f"\n🏆 MASSIVE TEST COMPLETE!")
        print("=" * 60)
        print(f"Total Users Tested: {len(test_scenarios)}")
        print(f"Total Sessions: {analysis.get('total_sessions', 0)}")
        print(f"Human Sessions: {analysis.get('human_sessions', 0)}")
        print(f"Bot Sessions: {analysis.get('bot_sessions', 0)}")
        print(f"Success Rate: {analysis.get('success_rate', 0):.2%}")
        
        # Save results
        with open("massive_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"📁 Results saved to: massive_test_results.json")
        
        return all_results

def main():
    """Run the massive comprehensive test"""
    test_suite = MassiveTestSuite()
    results = test_suite.run_massive_test()
    
    print(f"\n✅ MASSIVE TEST COMPLETED SUCCESSFULLY!")
    print(f"📊 Total results collected: {len(results)}")

if __name__ == "__main__":
    main()
