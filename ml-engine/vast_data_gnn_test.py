#!/usr/bin/env python3
"""
🔬 VAST DATA GNN STRESS TEST
============================
Comprehensive testing of GNN with extensive datasets to ensure proper bot detection
"""

import requests
import json
import time
import random

def create_vast_human_data(user_id, session_id, log_count=100):
    """Generate vast human behavioral data with natural variations"""
    logs = []
    
    for i in range(log_count):
        # Human-like touch patterns with natural variation
        if i % 3 == 0:  # Touch sequences
            touch_events = []
            for j in range(random.randint(3, 8)):  # Variable touch count
                touch_events.append({
                    "x": random.randint(50, 400) + random.gauss(0, 15),  # Natural jitter
                    "y": random.randint(100, 800) + random.gauss(0, 20),
                    "pressure": 0.3 + random.random() * 0.4 + random.gauss(0, 0.05),  # 0.3-0.7 + noise
                    "duration": 80 + random.randint(-30, 50) + random.gauss(0, 10)  # 50-130ms + noise
                })
            
            logs.append({
                "timestamp": f"2024-01-01T10:{i:02d}:{random.randint(0,59):02d}",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": touch_events,
                    "accelerometer": {
                        "x": random.gauss(0.02, 0.01),
                        "y": random.gauss(0.15, 0.05), 
                        "z": random.gauss(9.78, 0.1)
                    },
                    "gyroscope": {
                        "x": random.gauss(0.001, 0.0005),
                        "y": random.gauss(0.002, 0.001),
                        "z": random.gauss(0.0015, 0.0008)
                    }
                }
            })
        
        elif i % 3 == 1:  # Keystroke sequences
            keystroke_count = random.randint(3, 12)
            keystrokes = []
            typing_rhythm = []
            intervals = []
            
            for k in range(keystroke_count):
                key_char = chr(97 + random.randint(0, 25))  # a-z
                dwell = 70 + random.randint(-20, 40) + random.gauss(0, 8)
                pressure = 0.4 + random.random() * 0.3 + random.gauss(0, 0.03)
                
                keystrokes.append({
                    "key": key_char,
                    "dwell_time": max(30, dwell),  # Minimum 30ms
                    "pressure": max(0.1, min(0.9, pressure))  # 0.1-0.9 range
                })
                
                typing_rhythm.append(60 + random.randint(-15, 30))
                intervals.append(0.08 + random.random() * 0.12 + random.gauss(0, 0.02))
            
            logs.append({
                "timestamp": f"2024-01-01T10:{i:02d}:{random.randint(0,59):02d}",
                "event_type": "keystroke_sequence", 
                "data": {
                    "keystrokes": keystrokes,
                    "typing_rhythm": typing_rhythm,
                    "inter_key_intervals": intervals
                }
            })
        
        else:  # Mouse movements
            movement_count = random.randint(2, 6)
            movements = []
            
            for m in range(movement_count):
                movements.append({
                    "x": random.randint(0, 1920) + random.gauss(0, 5),
                    "y": random.randint(0, 1080) + random.gauss(0, 5), 
                    "velocity": random.uniform(50, 300) + random.gauss(0, 20),
                    "acceleration": random.uniform(-50, 50) + random.gauss(0, 10)
                })
            
            logs.append({
                "timestamp": f"2024-01-01T10:{i:02d}:{random.randint(0,59):02d}",
                "event_type": "mouse_movement",
                "data": {
                    "movements": movements,
                    "scroll_events": [
                        {"direction": random.choice(["up", "down"]), "amount": random.randint(1, 5)}
                        for _ in range(random.randint(0, 3))
                    ]
                }
            })
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "logs": logs
    }

def create_vast_bot_data(user_id, session_id, log_count=100):
    """Generate vast bot behavioral data with suspicious identical patterns"""
    logs = []
    
    # Bot patterns - very consistent/identical
    base_x, base_y = 200, 300
    base_pressure = 0.5
    base_duration = 120
    
    for i in range(log_count):
        if i % 2 == 0:  # Identical touch sequences (major bot indicator)
            identical_touches = []
            for j in range(5):  # Always 5 touches, all identical
                identical_touches.append({
                    "x": base_x,  # EXACTLY the same
                    "y": base_y,  # EXACTLY the same
                    "pressure": base_pressure,  # EXACTLY the same
                    "duration": base_duration   # EXACTLY the same
                })
            
            logs.append({
                "timestamp": f"2024-01-01T11:{i:02d}:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": identical_touches,
                    "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},  # IDENTICAL
                    "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}   # IDENTICAL
                }
            })
        
        else:  # Identical keystroke sequences
            identical_keystrokes = []
            for k in range(4):  # Always 4 keys, all identical timing
                identical_keystrokes.append({
                    "key": "a",  # Always same key
                    "dwell_time": 100,  # EXACTLY the same
                    "pressure": 0.55    # EXACTLY the same
                })
            
            logs.append({
                "timestamp": f"2024-01-01T11:{i:02d}:00",
                "event_type": "keystroke_sequence",
                "data": {
                    "keystrokes": identical_keystrokes,
                    "typing_rhythm": [95, 95, 95, 95],  # IDENTICAL rhythm
                    "inter_key_intervals": [0.12, 0.12, 0.12, 0.12]  # IDENTICAL intervals
                }
            })
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "logs": logs
    }

def create_mixed_bot_data(user_id, session_id, log_count=80):
    """Generate mixed data - mostly human with some bot patterns"""
    logs = []
    
    for i in range(log_count):
        # 70% human-like, 30% bot-like patterns
        is_bot_pattern = (i % 10) < 3  # Every 10 logs, 3 are bot-like
        
        if is_bot_pattern:
            # Bot pattern
            logs.append({
                "timestamp": f"2024-01-01T12:{i:02d}:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 150, "y": 250, "pressure": 0.6, "duration": 110},  # IDENTICAL
                        {"x": 150, "y": 250, "pressure": 0.6, "duration": 110},  # IDENTICAL
                        {"x": 150, "y": 250, "pressure": 0.6, "duration": 110}   # IDENTICAL
                    ],
                    "accelerometer": {"x": 0.01, "y": 0.12, "z": 9.79},
                    "gyroscope": {"x": 0.0008, "y": 0.0018, "z": 0.0012}
                }
            })
        else:
            # Human pattern
            touch_events = []
            for j in range(random.randint(2, 6)):
                touch_events.append({
                    "x": random.randint(100, 300) + random.gauss(0, 10),
                    "y": random.randint(200, 400) + random.gauss(0, 15),
                    "pressure": 0.4 + random.random() * 0.3,
                    "duration": 90 + random.randint(-20, 40)
                })
            
            logs.append({
                "timestamp": f"2024-01-01T12:{i:02d}:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": touch_events,
                    "accelerometer": {
                        "x": random.gauss(0.015, 0.008),
                        "y": random.gauss(0.12, 0.03),
                        "z": random.gauss(9.79, 0.08)
                    },
                    "gyroscope": {
                        "x": random.gauss(0.0008, 0.0004),
                        "y": random.gauss(0.0018, 0.0006),
                        "z": random.gauss(0.0012, 0.0005)
                    }
                }
            })
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "logs": logs
    }

def test_vast_data_scenarios():
    """Test GNN with vast datasets"""
    print("🔬 GNN VAST DATA STRESS TEST")
    print("="*80)
    print("🚀 Testing GNN bot detection with extensive behavioral datasets")
    print("="*80)
    
    # Test scenarios with increasing data sizes
    test_scenarios = [
        {
            "name": "🧑 VAST HUMAN DATA (100 logs)",
            "data_func": lambda: create_vast_human_data("vast_human_001", "human_session_100", 100),
            "expected": "Human-like behavior, low anomaly score"
        },
        {
            "name": "🤖 VAST BOT DATA (100 logs)", 
            "data_func": lambda: create_vast_bot_data("vast_bot_001", "bot_session_100", 100),
            "expected": "Bot behavior, HIGH anomaly score"
        },
        {
            "name": "🔀 MIXED DATA (80 logs - 70% human, 30% bot)",
            "data_func": lambda: create_mixed_bot_data("mixed_user_001", "mixed_session_80", 80),
            "expected": "Mixed behavior, moderate anomaly score"
        },
        {
            "name": "🧑 MEGA HUMAN DATA (200 logs)",
            "data_func": lambda: create_vast_human_data("mega_human_001", "human_session_200", 200),
            "expected": "Human-like behavior, consistent low anomaly"
        },
        {
            "name": "🤖 MEGA BOT DATA (200 logs)",
            "data_func": lambda: create_vast_bot_data("mega_bot_001", "bot_session_200", 200), 
            "expected": "Bot behavior, MAXIMUM anomaly score"
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("="*70)
        print(f"📋 Expected: {scenario['expected']}")
        
        try:
            # Generate test data
            print("📊 Generating test data...")
            test_data = scenario['data_func']()
            log_count = len(test_data['logs'])
            
            # Analyze first few logs for pattern verification
            sample_logs = test_data['logs'][:3]
            print(f"📈 Generated {log_count} behavioral logs")
            print(f"📝 Sample events: {[log['event_type'] for log in sample_logs]}")
            
            # Send to ML Engine
            print(f"📤 Sending vast dataset to ML Engine...")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8001/analyze-mobile",
                json=test_data,
                timeout=120  # Extended timeout for large datasets
            )
            
            processing_time = time.time() - start_time
            print(f"📥 Response Status: {response.status_code}")
            print(f"⏱️  Processing Time: {processing_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key metrics
                decision = result.get('decision', 'N/A')
                risk_score = result.get('risk_score', 0)
                confidence = result.get('confidence', 0)
                
                # GNN Analysis
                gnn_analysis = result.get('gnn_analysis', {})
                anomaly_score = gnn_analysis.get('anomaly_score', 0)
                gnn_confidence = gnn_analysis.get('gnn_confidence', 0)
                anomaly_types = gnn_analysis.get('anomaly_types', [])
                risk_adjustment = gnn_analysis.get('risk_adjustment', 0)
                
                # Vector Analysis
                vector_stats = result.get('vector_stats', {})
                data_richness = vector_stats.get('non_zero_percentage', 0)
                vector_meaningful = vector_stats.get('is_meaningful', False)
                
                print(f"\n🎯 VAST DATA ANALYSIS RESULTS:")
                print(f"   Final Decision: {decision}")
                print(f"   Risk Score: {risk_score}")
                print(f"   Overall Confidence: {confidence}")
                print(f"   Processing Time: {result.get('processing_time_ms', 'N/A')}ms")
                
                print(f"\n🧠 GNN ANALYSIS (KEY METRICS):")
                print(f"   🔍 Anomaly Score: {anomaly_score}")
                print(f"   🎯 GNN Confidence: {gnn_confidence}")  
                print(f"   ⚠️  Risk Adjustment: {risk_adjustment}")
                print(f"   🏷️  Anomaly Types: {anomaly_types}")
                
                print(f"\n📊 BEHAVIORAL VECTOR QUALITY:")
                print(f"   Data Richness: {data_richness}%")
                print(f"   Vector Meaningful: {vector_meaningful}")
                print(f"   Total Features: {vector_stats.get('length', 90)}")
                print(f"   Non-Zero Features: {vector_stats.get('non_zero_count', 0)}")
                
                # Analyze results
                if "bot" in scenario['name'].lower():
                    if anomaly_score > 0.5:
                        print(f"✅ SUCCESS: Bot correctly detected! (Score: {anomaly_score})")
                        result_status = "✅ PASS"
                    elif anomaly_score > 0.0:
                        print(f"⚠️  PARTIAL: Some anomaly detected but low score: {anomaly_score}")
                        result_status = "⚠️ PARTIAL"
                    else:
                        print(f"❌ FAILED: Bot not detected! (Score: {anomaly_score})")
                        result_status = "❌ FAIL"
                elif "human" in scenario['name'].lower():
                    if anomaly_score < 0.3:
                        print(f"✅ SUCCESS: Human behavior correctly identified! (Score: {anomaly_score})")
                        result_status = "✅ PASS"
                    else:
                        print(f"⚠️  WARNING: Human flagged as anomalous: {anomaly_score}")
                        result_status = "⚠️ PARTIAL"
                else:  # Mixed
                    if 0.2 < anomaly_score < 0.8:
                        print(f"✅ SUCCESS: Mixed behavior detected appropriately! (Score: {anomaly_score})")
                        result_status = "✅ PASS"
                    else:
                        print(f"⚠️  PARTIAL: Mixed behavior scoring: {anomaly_score}")
                        result_status = "⚠️ PARTIAL"
                
                # Store results
                results.append({
                    'scenario': scenario['name'],
                    'logs': log_count,
                    'processing_time': processing_time,
                    'anomaly_score': anomaly_score,
                    'gnn_confidence': gnn_confidence,
                    'decision': decision,
                    'status': result_status
                })
                
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                results.append({
                    'scenario': scenario['name'],
                    'logs': log_count,
                    'status': "❌ ERROR",
                    'error': response.text
                })
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                'scenario': scenario['name'],
                'status': "❌ EXCEPTION",
                'error': str(e)
            })
        
        print("\n" + "⏸️ " * 30)  # Separator
        time.sleep(2)  # Brief pause between tests
    
    # Summary Report
    print(f"\n🏆 VAST DATA GNN STRESS TEST SUMMARY")
    print("="*70)
    
    for result in results:
        logs_info = f"({result['logs']} logs)" if 'logs' in result else ""
        processing_info = f" | {result['processing_time']:.2f}s" if 'processing_time' in result else ""
        anomaly_info = f" | Anomaly: {result['anomaly_score']}" if 'anomaly_score' in result else ""
        
        print(f"{result['status']} {result['scenario']} {logs_info}{processing_info}{anomaly_info}")
    
    # Overall Assessment
    passes = sum(1 for r in results if r['status'] == "✅ PASS")
    partials = sum(1 for r in results if r['status'] == "⚠️ PARTIAL")
    failures = sum(1 for r in results if "❌" in r['status'])
    
    print(f"\n📈 OVERALL GNN PERFORMANCE:")
    print(f"   ✅ Passes: {passes}/{len(results)}")
    print(f"   ⚠️  Partials: {partials}/{len(results)}")  
    print(f"   ❌ Failures: {failures}/{len(results)}")
    
    if failures == 0 and passes >= len(results) * 0.7:
        print(f"🎉 GNN VAST DATA TEST: EXCELLENT PERFORMANCE!")
    elif failures <= 1:
        print(f"✅ GNN VAST DATA TEST: GOOD PERFORMANCE")
    else:
        print(f"⚠️  GNN VAST DATA TEST: NEEDS IMPROVEMENT")
    
    return results

if __name__ == "__main__":
    print("🔬 Starting GNN Vast Data Stress Test...")
    print("🎯 This will test GNN bot detection with extensive behavioral datasets")
    print("⏱️  Expected total time: 2-5 minutes depending on data size")
    print()
    
    results = test_vast_data_scenarios()
    
    print(f"\n✅ GNN Vast Data Stress Test Complete!")
    print("📋 Check the results above to verify GNN detection capabilities")
    print("🔍 Look specifically for bot detection scores > 0.5 and human scores < 0.3")
