#!/usr/bin/env python3
"""
Real-world behavioral authentication test with realistic user personas
Tests the complete backend + ML Engine pipeline with comprehensive logging
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

class UserPersona:
    """Represents a realistic user with specific behavioral patterns"""
    
    def __init__(self, name: str, user_type: str, characteristics: Dict[str, Any]):
        self.name = name
        self.user_type = user_type  # "human" or "bot"
        self.characteristics = characteristics
        self.session_count = 0
    
    def generate_behavioral_logs(self, session_id: str, num_events: int = 5) -> List[Dict[str, Any]]:
        """Generate realistic behavioral logs based on persona"""
        logs = []
        
        if self.user_type == "human":
            logs = self._generate_human_behavior(session_id, num_events)
        else:
            logs = self._generate_bot_behavior(session_id, num_events)
        
        return logs
    
    def _generate_human_behavior(self, session_id: str, num_events: int) -> List[Dict[str, Any]]:
        """Generate natural human behavioral patterns"""
        logs = []
        base_time = datetime.now()
        
        for i in range(num_events):
            # Natural variations in touch patterns
            base_x = self.characteristics["preferred_touch_area"]["x"]
            base_y = self.characteristics["preferred_touch_area"]["y"]
            
            # Add human-like variations
            x_variation = random.normal(0, self.characteristics["touch_precision"]["x_std"])
            y_variation = random.normal(0, self.characteristics["touch_precision"]["y_std"])
            
            # Touch pressure varies naturally
            base_pressure = self.characteristics["touch_pressure"]["base"]
            pressure_variation = random.normal(0, self.characteristics["touch_pressure"]["std"])
            pressure = max(0.1, min(2.0, base_pressure + pressure_variation))
            
            # Duration varies with user mood/fatigue
            base_duration = self.characteristics["touch_duration"]["base"]
            duration_variation = random.normal(0, self.characteristics["touch_duration"]["std"])
            duration = max(50, int(base_duration + duration_variation))
            
            # Natural device movement
            accel_x = random.normal(self.characteristics["device_stability"]["accel_base"], 0.1)
            accel_y = random.normal(0.1, 0.05)
            accel_z = random.normal(9.8, 0.1)
            
            gyro_x = random.normal(0, self.characteristics["device_stability"]["gyro_std"])
            gyro_y = random.normal(0, self.characteristics["device_stability"]["gyro_std"])
            gyro_z = random.normal(0, self.characteristics["device_stability"]["gyro_std"])
            
            event_time = base_time + timedelta(seconds=i * random.uniform(1, 3))
            
            logs.append({
                "timestamp": event_time.isoformat() + "Z",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {
                            "coordinates": [int(base_x + x_variation), int(base_y + y_variation)],
                            "pressure": round(pressure, 3),
                            "duration": duration
                        }
                    ],
                    "accelerometer": [
                        {"x": round(accel_x, 3), "y": round(accel_y, 3), "z": round(accel_z, 3)}
                    ],
                    "gyroscope": [
                        {"x": round(gyro_x, 4), "y": round(gyro_y, 4), "z": round(gyro_z, 4)}
                    ]
                }
            })
        
        return logs
    
    def _generate_bot_behavior(self, session_id: str, num_events: int) -> List[Dict[str, Any]]:
        """Generate robotic/automated behavioral patterns"""
        logs = []
        base_time = datetime.now()
        
        for i in range(num_events):
            # Perfect, identical touch patterns (robotic)
            x = self.characteristics["fixed_coordinates"]["x"]
            y = self.characteristics["fixed_coordinates"]["y"]
            
            # Perfect pressure and timing
            pressure = self.characteristics["fixed_pressure"]
            duration = self.characteristics["fixed_duration"]
            
            # No device movement (automated)
            accel_x = self.characteristics["perfect_stability"]["accel_x"]
            accel_y = self.characteristics["perfect_stability"]["accel_y"]
            accel_z = self.characteristics["perfect_stability"]["accel_z"]
            
            gyro_x = self.characteristics["perfect_stability"]["gyro_x"]
            gyro_y = self.characteristics["perfect_stability"]["gyro_y"]
            gyro_z = self.characteristics["perfect_stability"]["gyro_z"]
            
            # Perfect timing intervals
            event_time = base_time + timedelta(seconds=i * self.characteristics["fixed_interval"])
            
            logs.append({
                "timestamp": event_time.isoformat() + "Z",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {
                            "coordinates": [x, y],
                            "pressure": pressure,
                            "duration": duration
                        }
                    ],
                    "accelerometer": [
                        {"x": accel_x, "y": accel_y, "z": accel_z}
                    ],
                    "gyroscope": [
                        {"x": gyro_x, "y": gyro_y, "z": gyro_z}
                    ]
                }
            })
        
        return logs

def create_user_personas() -> Dict[str, UserPersona]:
    """Create realistic user personas for testing"""
    
    personas = {
        "rajesh_normal": UserPersona(
            name="Rajesh Kumar",
            user_type="human",
            characteristics={
                "preferred_touch_area": {"x": 200, "y": 300},
                "touch_precision": {"x_std": 15, "y_std": 12},
                "touch_pressure": {"base": 0.7, "std": 0.15},
                "touch_duration": {"base": 120, "std": 20},
                "device_stability": {"accel_base": 0.1, "gyro_std": 0.02}
            }
        ),
        
        "priya_careful": UserPersona(
            name="Priya Sharma",
            user_type="human",
            characteristics={
                "preferred_touch_area": {"x": 180, "y": 250},
                "touch_precision": {"x_std": 8, "y_std": 6},  # More precise
                "touch_pressure": {"base": 0.5, "std": 0.1},  # Lighter touch
                "touch_duration": {"base": 150, "std": 25},   # Longer duration
                "device_stability": {"accel_base": 0.05, "gyro_std": 0.01}  # More stable
            }
        ),
        
        "automation_bot": UserPersona(
            name="Automation Bot",
            user_type="bot",
            characteristics={
                "fixed_coordinates": {"x": 100, "y": 200},
                "fixed_pressure": 1.0,
                "fixed_duration": 100,
                "fixed_interval": 1.0,  # Perfect 1-second intervals
                "perfect_stability": {
                    "accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.8,
                    "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0
                }
            }
        ),
        
        "malicious_bot": UserPersona(
            name="Malicious Bot",
            user_type="bot",
            characteristics={
                "fixed_coordinates": {"x": 0, "y": 0},  # Suspicious coordinates
                "fixed_pressure": 5.0,  # Impossible pressure
                "fixed_duration": 1,    # Impossibly fast
                "fixed_interval": 0.1,  # Super fast intervals
                "perfect_stability": {
                    "accel_x": 50.0, "accel_y": 50.0, "accel_z": 50.0,  # Impossible values
                    "gyro_x": 10.0, "gyro_y": 10.0, "gyro_z": 10.0
                }
            }
        )
    }
    
    return personas

def test_authentication_flow(persona: UserPersona, backend_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test complete authentication flow with a persona"""
    
    print(f"🧪 TESTING USER PERSONA: {persona.name} ({persona.user_type})")
    print("-" * 60)
    
    results = {
        "persona_name": persona.name,
        "persona_type": persona.user_type,
        "steps": []
    }
    
    # Step 1: Register user
    register_data = {
        "user_id": f"test_{persona.name.lower().replace(' ', '_')}",
        "phone_number": f"+91{random.randint(1000000000, 9999999999)}",
        "name": persona.name,
        "mpin": "1234"
    }
    
    try:
        print("📝 Step 1: User Registration")
        response = requests.post(f"{backend_url}/api/v1/auth/register", json=register_data, timeout=10)
        register_result = {
            "step": "register",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text
        }
        results["steps"].append(register_result)
        print(f"   Status: {response.status_code} - {'✅ Success' if response.status_code == 200 else '❌ Failed'}")
        
        if response.status_code != 200:
            print(f"   Error: {response.text}")
            return results
            
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        results["steps"].append({"step": "register", "error": str(e)})
        return results
    
    # Step 2: Login with behavioral data
    user_id = register_data["user_id"]
    session_id = f"session_{user_id}_{int(time.time())}"
    
    # Generate behavioral logs for this persona
    behavioral_logs = persona.generate_behavioral_logs(session_id, num_events=3)
    
    login_data = {
        "user_id": user_id,
        "phone_number": register_data["phone_number"],
        "mpin": register_data["mpin"],
        "session_id": session_id,
        "behavioral_data": {
            "user_id": user_id,
            "session_id": session_id,
            "logs": behavioral_logs
        }
    }
    
    try:
        print("🔐 Step 2: Login with Behavioral Analysis")
        response = requests.post(f"{backend_url}/api/v1/auth/login", json=login_data, timeout=30)
        login_result = {
            "step": "login",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text,
            "behavioral_logs_count": len(behavioral_logs)
        }
        results["steps"].append(login_result)
        print(f"   Status: {response.status_code} - {'✅ Success' if response.status_code == 200 else '❌ Failed'}")
        
        if response.status_code == 200:
            login_response = response.json()
            print(f"   ML Analysis: {login_response.get('ml_analysis', {}).get('analysis_type', 'N/A')}")
            print(f"   Decision: {login_response.get('ml_analysis', {}).get('decision', 'N/A')}")
            print(f"   Risk Score: {login_response.get('ml_analysis', {}).get('risk_score', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Login failed: {e}")
        results["steps"].append({"step": "login", "error": str(e)})
        return results
    
    print(f"✅ Test completed for {persona.name}")
    return results

def run_comprehensive_test():
    """Run comprehensive test with all personas"""
    print("🎯 COMPREHENSIVE BEHAVIORAL AUTHENTICATION TEST")
    print("Testing complete backend + ML Engine pipeline with realistic user personas")
    print("=" * 80)
    
    # Create user personas
    personas = create_user_personas()
    all_results = []
    
    # Test each persona
    for persona_id, persona in personas.items():
        try:
            result = test_authentication_flow(persona)
            all_results.append(result)
            time.sleep(2)  # Delay between tests
        except Exception as e:
            print(f"❌ Persona test failed for {persona.name}: {e}")
            all_results.append({
                "persona_name": persona.name,
                "persona_type": persona.user_type,
                "error": str(e)
            })
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 40)
    
    human_users = [r for r in all_results if r.get("persona_type") == "human"]
    bot_users = [r for r in all_results if r.get("persona_type") == "bot"]
    
    print(f"Human Users Tested: {len(human_users)}")
    print(f"Bot Users Tested: {len(bot_users)}")
    
    # Check if system can differentiate
    successful_logins = 0
    failed_logins = 0
    
    for result in all_results:
        if "steps" in result:
            login_step = next((s for s in result["steps"] if s["step"] == "login"), None)
            if login_step and login_step.get("success"):
                successful_logins += 1
            else:
                failed_logins += 1
    
    print(f"\nAuthentication Results:")
    print(f"  Successful: {successful_logins}")
    print(f"  Failed: {failed_logins}")
    print(f"\nDetailed results saved to test log files")
    
    # Save detailed results
    with open(f"comprehensive_test_results_{int(time.time())}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results

if __name__ == "__main__":
    run_comprehensive_test()
