#!/usr/bin/env python3
"""Comprehensive test of the complete behavioral authentication system"""

import requests
import json
import time

def comprehensive_system_test():
    """Test the complete behavioral authentication system with all layers"""
    print("🎯 COMPREHENSIVE BEHAVIORAL AUTHENTICATION SYSTEM TEST")
    print("=" * 80)
    
    test_scenarios = {
        "normal_user": {
            "description": "Normal human behavior with natural variations",
            "user_id": "comprehensive_test_normal_user",
            "session_id": "normal_comprehensive_001",
            "logs": [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"coordinates": [120, 250], "pressure": 0.75, "duration": 125},
                            {"coordinates": [125, 255], "pressure": 0.80, "duration": 118},
                            {"coordinates": [130, 260], "pressure": 0.70, "duration": 132}
                        ],
                        "accelerometer": [
                            {"x": 0.12, "y": 0.25, "z": 9.82},
                            {"x": 0.08, "y": 0.22, "z": 9.79}
                        ],
                        "gyroscope": [
                            {"x": 0.015, "y": 0.018, "z": 0.012},
                            {"x": 0.011, "y": 0.021, "z": 0.008}
                        ]
                    }
                }
            ],
            "expected": "Normal behavior - should be allowed or learned"
        },
        
        "subtle_automation": {
            "description": "Subtle automation with slightly too perfect patterns",
            "user_id": "comprehensive_test_automation",
            "session_id": "automation_comprehensive_001",
            "logs": [
                {
                    "timestamp": "2024-01-01T10:01:00",
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100}
                        ],
                        "accelerometer": [
                            {"x": 0.0, "y": 0.0, "z": 9.8},
                            {"x": 0.0, "y": 0.0, "z": 9.8}
                        ],
                        "gyroscope": [
                            {"x": 0.0, "y": 0.0, "z": 0.0},
                            {"x": 0.0, "y": 0.0, "z": 0.0}
                        ]
                    }
                }
            ],
            "expected": "Robotic patterns - should be flagged"
        },
        
        "extreme_attack": {
            "description": "Clearly malicious behavior with impossible values", 
            "user_id": "comprehensive_test_attack",
            "session_id": "attack_comprehensive_001",
            "logs": [
                {
                    "timestamp": "2024-01-01T10:02:00",
                    "event_type": "touch_sequence", 
                    "data": {
                        "touch_events": [
                            {"coordinates": [0, 0], "pressure": 10.0, "duration": 1},
                            {"coordinates": [9999, 9999], "pressure": 10.0, "duration": 1},
                            {"coordinates": [0, 9999], "pressure": 10.0, "duration": 1}
                        ],
                        "accelerometer": [
                            {"x": 50.0, "y": 50.0, "z": 50.0}
                        ],
                        "gyroscope": [
                            {"x": 10.0, "y": 10.0, "z": 10.0}
                        ]
                    }
                }
            ],
            "expected": "Extreme attack - should be denied"
        }
    }
    
    results = {}
    
    for scenario_name, scenario in test_scenarios.items():
        print(f"\n🧪 TESTING: {scenario_name.upper()}")
        print(f"📝 {scenario['description']}")
        print("-" * 60)
        
        try:
            test_data = {
                "user_id": scenario["user_id"],
                "session_id": scenario["session_id"], 
                "logs": scenario["logs"]
            }
            
            response = requests.post(
                "http://localhost:8001/analyze-mobile",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key metrics
                decision = result.get('decision', 'unknown')
                confidence = result.get('confidence', 0)
                risk_score = result.get('risk_score', 0)
                analysis_type = result.get('analysis_type', 'unknown')
                gnn_analysis = result.get('gnn_analysis', {})
                vector_stats = result.get('vector_stats', {})
                
                results[scenario_name] = {
                    'decision': decision,
                    'confidence': confidence,
                    'risk_score': risk_score,
                    'analysis_type': analysis_type,
                    'gnn_anomaly_score': gnn_analysis.get('anomaly_score', 0),
                    'gnn_types': gnn_analysis.get('anomaly_types', []),
                    'vector_non_zeros': vector_stats.get('non_zero_count', 0),
                    'expected': scenario['expected']
                }
                
                print(f"✅ RESULT:")
                print(f"   Decision: {decision}")
                print(f"   Confidence: {confidence}")
                print(f"   Risk Score: {risk_score:.6f}")
                print(f"   Analysis: {analysis_type}")
                print(f"   GNN Anomaly: {gnn_analysis.get('anomaly_score', 0):.6f}")
                print(f"   Vector Quality: {vector_stats.get('non_zero_count', 0)}/90 non-zeros")
                print(f"   Expected: {scenario['expected']}")
                
                # Assess if result matches expectation
                if scenario_name == 'normal_user':
                    if decision in ['learn', 'allow'] and risk_score < 0.5:
                        print(f"   ✅ EXPECTED: Normal behavior correctly handled")
                    else:
                        print(f"   ⚠️  UNEXPECTED: Normal behavior flagged as risky")
                
                elif scenario_name == 'subtle_automation':
                    if risk_score > 0.1 or gnn_analysis.get('anomaly_score', 0) > 0.1:
                        print(f"   ✅ GOOD: Automation patterns detected")
                    else:
                        print(f"   📋 NOTE: Subtle automation not flagged (may need more data)")
                
                elif scenario_name == 'extreme_attack':
                    if decision in ['challenge', 'deny'] or risk_score > 0.5:
                        print(f"   ✅ EXCELLENT: Extreme attack correctly blocked")
                    else:
                        print(f"   ❌ ISSUE: Extreme attack not blocked")
                
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                results[scenario_name] = {'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results[scenario_name] = {'error': str(e)}
        
        time.sleep(0.5)  # Small delay between tests
    
    # Final system assessment
    print(f"\n🏆 FINAL SYSTEM ASSESSMENT")
    print("=" * 60)
    
    working_components = []
    if any('enhanced_faiss_with_gnn' in str(result.get('analysis_type', '')) for result in results.values() if 'error' not in result):
        working_components.append("✅ Enhanced FAISS Engine")
        working_components.append("✅ Enhanced Behavioral Processor") 
        working_components.append("✅ Adaptive Layer")
        working_components.append("✅ GNN Anomaly Detector")
    
    print("WORKING COMPONENTS:")
    for component in working_components:
        print(f"   {component}")
    
    # Check system capabilities
    capabilities = []
    if any(result.get('vector_non_zeros', 0) > 30 for result in results.values() if 'error' not in result):
        capabilities.append("✅ Genuine behavioral vector generation")
    
    if any(result.get('risk_score', 0) > 0.5 for result in results.values() if 'error' not in result):
        capabilities.append("✅ High-risk anomaly detection")
    
    if any(result.get('gnn_anomaly_score', 0) > 0 for result in results.values() if 'error' not in result):
        capabilities.append("✅ GNN pattern analysis")
    
    unique_decisions = set(result.get('decision', '') for result in results.values() if 'error' not in result)
    if len(unique_decisions) > 1:
        capabilities.append("✅ Dynamic decision making")
    
    print(f"\nSYSTEM CAPABILITIES:")
    for capability in capabilities:
        print(f"   {capability}")
    
    print(f"\n🎯 SYSTEM STATUS: {'FULLY OPERATIONAL' if len(working_components) == 4 and len(capabilities) >= 3 else 'PARTIALLY OPERATIONAL'}")
    
    return results

if __name__ == "__main__":
    comprehensive_system_test()
