#!/usr/bin/env python3
"""
REAL WORLD BEHAVIORAL AUTHENTICATION TEST
Tests the complete backend + ML Engine pipeline with realistic user personas
"""

import requests
import json
import time
import random
import math
from datetime import datetime, timedelta

# Backend and ML Engine endpoints
BACKEND_URL = "http://localhost:8000"
ML_ENGINE_URL = "http://localhost:8001"

def generate_realistic_human_behavior(user_profile, session_count=1):
    """Generate realistic human behavioral data based on user profile"""
    
    behaviors = []
    base_time = datetime.now()
    
    for session in range(session_count):
        # Realistic human variations
        pressure_base = user_profile.get('pressure_preference', 0.7)
        pressure_variance = user_profile.get('pressure_variance', 0.15)
        
        touch_duration_base = user_profile.get('touch_duration', 120)
        touch_variance = user_profile.get('touch_variance', 20)
        
        # Generate natural touch sequence with human-like variations
        touch_events = []
        for i in range(random.randint(3, 8)):
            # Natural coordinate progression with slight tremor
            x = user_profile.get('start_x', 150) + i * 10 + random.uniform(-5, 5)
            y = user_profile.get('start_y', 200) + i * 8 + random.uniform(-3, 7)
            
            # Human pressure variation
            pressure = max(0.1, min(2.0, pressure_base + random.gauss(0, pressure_variance)))
            
            # Duration with human variation
            duration = max(50, touch_duration_base + random.gauss(0, touch_variance))
            
            touch_events.append({
                "coordinates": [round(x, 1), round(y, 1)],
                "pressure": round(pressure, 3),
                "duration": round(duration)
            })
        
        # Realistic accelerometer data (phone in hand)
        accel_data = []
        for i in range(random.randint(2, 5)):
            # Natural hand tremor and movement
            x_accel = random.gauss(0.1, 0.05)  # Slight hand shake
            y_accel = random.gauss(0.2, 0.08)  # Natural tilt
            z_accel = 9.8 + random.gauss(0, 0.02)  # Gravity with minor variations
            
            accel_data.append({
                "x": round(x_accel, 4),
                "y": round(y_accel, 4), 
                "z": round(z_accel, 4)
            })
        
        # Realistic gyroscope data (minor rotations)
        gyro_data = []
        for i in range(len(accel_data)):
            gyro_data.append({
                "x": round(random.gauss(0.01, 0.005), 5),
                "y": round(random.gauss(0.02, 0.008), 5),
                "z": round(random.gauss(0.01, 0.003), 5)
            })
        
        behavior = {
            "user_id": user_profile["user_id"],
            "session_id": f"{user_profile['user_id']}_session_{session + 1}",
            "logs": [
                {
                    "timestamp": (base_time + timedelta(seconds=session*30)).isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": touch_events,
                        "accelerometer": accel_data,
                        "gyroscope": gyro_data
                    }
                }
            ]
        }
        behaviors.append(behavior)
    
    return behaviors

def generate_robotic_behavior(bot_profile, session_count=1):
    """Generate robotic/automated behavioral patterns"""
    
    behaviors = []
    base_time = datetime.now()
    
    for session in range(session_count):
        # Perfect robotic patterns - no variation
        touch_events = []
        for i in range(bot_profile.get('touch_count', 5)):
            # Identical coordinates (dead giveaway)
            x = bot_profile.get('target_x', 100)
            y = bot_profile.get('target_y', 200)
            
            # Perfect pressure (unnatural)
            pressure = bot_profile.get('pressure', 1.0)
            
            # Identical duration (robotic)
            duration = bot_profile.get('duration', 100)
            
            touch_events.append({
                "coordinates": [x, y],
                "pressure": pressure,
                "duration": duration
            })
        
        # Perfect accelerometer (phone perfectly still - impossible for human)
        accel_data = []
        for i in range(3):  # Consistent count
            accel_data.append({
                "x": 0.0,  # Perfect stillness
                "y": 0.0,  # Perfect stillness
                "z": 9.8   # Perfect gravity
            })
        
        # Perfect gyroscope (no rotation at all)
        gyro_data = []
        for i in range(3):
            gyro_data.append({
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            })
        
        behavior = {
            "user_id": bot_profile["user_id"],
            "session_id": f"{bot_profile['user_id']}_bot_session_{session + 1}",
            "logs": [
                {
                    "timestamp": (base_time + timedelta(seconds=session*30)).isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": touch_events,
                        "accelerometer": accel_data,
                        "gyroscope": gyro_data
                    }
                }
            ]
        }
        behaviors.append(behavior)
    
    return behaviors

def test_ml_engine_direct(user_data, description):
    """Test ML Engine directly (bypassing backend for now)"""
    
    print(f"\n🧪 TESTING: {description}")
    print(f"👤 User: {user_data['user_id']}")
    print(f"🔗 Session: {user_data['session_id']}")
    print("-" * 60)
    
    try:
        # Send behavioral data directly to ML Engine
        print("📤 1. Sending behavioral data to ML ENGINE...")
        
        ml_response = requests.post(
            f"{ML_ENGINE_URL}/analyze-mobile",
            json=user_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📥 ML Engine Status: {ml_response.status_code}")
        
        if ml_response.status_code == 200:
            ml_result = ml_response.json()
            print(f"✅ ML Engine Decision: {ml_result.get('decision', 'unknown')}")
            print(f"📊 Risk Score: {ml_result.get('risk_score', 0):.6f}")
            print(f"🔍 Analysis: {ml_result.get('analysis_type', 'unknown')}")
            print(f"🧠 Confidence: {ml_result.get('confidence', 0):.3f}")
            
            # Check if GNN was involved
            if 'gnn_analysis' in ml_result:
                gnn_data = ml_result['gnn_analysis']
                print(f"� GNN Anomaly Score: {gnn_data.get('anomaly_score', 0):.6f}")
                if gnn_data.get('anomaly_types'):
                    print(f"🚨 GNN Detected: {', '.join(gnn_data.get('anomaly_types', []))}")
            
            # Check vector quality
            if 'vector_stats' in ml_result:
                vector_stats = ml_result['vector_stats']
                print(f"📈 Vector Quality: {vector_stats.get('non_zero_count', 0)}/90 non-zeros")
                print(f"📊 Vector Mean: {vector_stats.get('mean', 0):.6f}")
            
            return ml_result
        else:
            print(f"❌ ML Engine Error: {ml_response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def run_comprehensive_real_world_test():
    """Run comprehensive real-world test scenarios"""
    
    print("🌟 COMPREHENSIVE REAL-WORLD BEHAVIORAL AUTHENTICATION TEST")
    print("=" * 80)
    print("Testing ML Engine directly with realistic user personas and bot detection")
    print("=" * 80)
    
    # Define realistic user personas
    user_personas = {
        "sarah_marketing": {
            "user_id": "sarah_marketing_exec",
            "description": "Marketing Executive - Confident touch, steady hand",
            "pressure_preference": 0.8,
            "pressure_variance": 0.12,
            "touch_duration": 130,
            "touch_variance": 25,
            "start_x": 160,
            "start_y": 220
        },
        
        "elderly_user": {
            "user_id": "robert_senior_user", 
            "description": "Senior User - Lighter touch, more variation",
            "pressure_preference": 0.5,
            "pressure_variance": 0.20,
            "touch_duration": 180,
            "touch_variance": 40,
            "start_x": 120,
            "start_y": 250
        },
        
        "young_gamer": {
            "user_id": "alex_mobile_gamer",
            "description": "Mobile Gamer - Fast, precise movements",
            "pressure_preference": 1.1,
            "pressure_variance": 0.08,
            "touch_duration": 90,
            "touch_variance": 15,
            "start_x": 140,
            "start_y": 180
        }
    }
    
    # Define bot profiles
    bot_profiles = {
        "automation_bot": {
            "user_id": "suspicious_automation_bot",
            "description": "Automation Bot - Perfect identical movements",
            "pressure": 1.0,
            "duration": 100,
            "target_x": 100,
            "target_y": 200,
            "touch_count": 4
        },
        
        "scraping_bot": {
            "user_id": "scraping_attack_bot",
            "description": "Scraping Bot - Rapid identical interactions", 
            "pressure": 1.0,
            "duration": 50,
            "target_x": 0,
            "target_y": 0,
            "touch_count": 6
        }
    }
    
    results = {}
    
    # Test Phase 1: Establish normal user baselines
    print(f"\n🏗️  PHASE 1: ESTABLISHING USER BASELINES")
    print("=" * 50)
    
    for persona_name, persona in user_personas.items():
        print(f"\n👤 Building baseline for: {persona['description']}")
        
        # Generate 3 sessions to establish baseline
        behaviors = generate_realistic_human_behavior(persona, session_count=3)
        
        persona_results = []
        for i, behavior in enumerate(behaviors):
            result = test_ml_engine_direct(
                behavior, 
                f"Normal User Baseline {i+1}/3"
            )
            if result:
                persona_results.append(result)
            time.sleep(1)  # Small delay
        
        results[persona_name] = persona_results
    
    # Test Phase 2: Bot detection
    print(f"\n🤖 PHASE 2: BOT DETECTION TESTING")
    print("=" * 50)
    
    for bot_name, bot_profile in bot_profiles.items():
        print(f"\n🚨 Testing: {bot_profile['description']}")
        
        # Generate bot behaviors
        bot_behaviors = generate_robotic_behavior(bot_profile, session_count=2)
        
        bot_results = []
        for i, behavior in enumerate(bot_behaviors):
            result = test_ml_engine_direct(
                behavior,
                f"Bot Attack {i+1}/2"
            )
            if result:
                bot_results.append(result)
            time.sleep(1)
        
        results[bot_name] = bot_results
    
    # Analysis Phase
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # Analyze user patterns
    print(f"\n👥 HUMAN USER ANALYSIS:")
    for persona_name, persona_results in results.items():
        if persona_name in user_personas:
            if persona_results:
                decisions = [r.get('decision', 'unknown') for r in persona_results]
                risk_scores = [r.get('risk_score', 0) for r in persona_results]
                
                print(f"   {persona_name}:")
                print(f"     Decisions: {decisions}")
                print(f"     Risk Range: {min(risk_scores):.3f} - {max(risk_scores):.3f}")
                print(f"     Avg Risk: {sum(risk_scores)/len(risk_scores):.3f}")
    
    # Analyze bot detection
    print(f"\n🤖 BOT DETECTION ANALYSIS:")
    for bot_name, bot_results in results.items():
        if bot_name in bot_profiles:
            if bot_results:
                decisions = [r.get('decision', 'unknown') for r in bot_results]
                risk_scores = [r.get('risk_score', 0) for r in bot_results]
                
                print(f"   {bot_name}:")
                print(f"     Decisions: {decisions}")
                print(f"     Risk Range: {min(risk_scores):.3f} - {max(risk_scores):.3f}")
                print(f"     Avg Risk: {sum(risk_scores)/len(risk_scores):.3f}")
    
    # Overall system assessment
    print(f"\n🏆 SYSTEM PERFORMANCE ASSESSMENT")
    print("=" * 50)
    
    # Check if system is properly differentiating
    all_human_risks = []
    all_bot_risks = []
    
    for persona_name, persona_results in results.items():
        if persona_results:
            risks = [r.get('risk_score', 0) for r in persona_results]
            if persona_name in user_personas:
                all_human_risks.extend(risks)
            elif persona_name in bot_profiles:
                all_bot_risks.extend(risks)
    
    if all_human_risks and all_bot_risks:
        avg_human_risk = sum(all_human_risks) / len(all_human_risks)
        avg_bot_risk = sum(all_bot_risks) / len(all_bot_risks)
        
        print(f"📈 Average Human Risk: {avg_human_risk:.4f}")
        print(f"📈 Average Bot Risk: {avg_bot_risk:.4f}")
        print(f"📊 Risk Separation: {abs(avg_bot_risk - avg_human_risk):.4f}")
        
        if avg_bot_risk > avg_human_risk + 0.1:
            print(f"✅ EXCELLENT: System successfully differentiates bots from humans!")
        elif avg_bot_risk > avg_human_risk:
            print(f"✅ GOOD: System shows some bot detection capability")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: Bot detection may need tuning")
    
    print(f"\n🎯 TEST COMPLETED - Check backend and ML Engine logs for detailed traces")
    
    return results

if __name__ == "__main__":
    run_comprehensive_real_world_test()
