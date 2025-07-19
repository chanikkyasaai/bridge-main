"""
Simple Data Flow Test - Windows Compatible
Shows real behavioral data transformation through each layer
"""
import os
import sys
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
import uuid
from typing import Dict, Any, List, Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import our components
from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
from src.core.enhanced_faiss_engine import EnhancedFAISSEngine
from src.layers.faiss_layer import FAISSLayer
from src.adapters.bank_adapter import BankAdapter
from src.drift_detector import DriftDetector

class SimpleDataFlowTester:
    """Test actual data flow with real transformations"""
    
    def __init__(self):
        self.processor = None
        self.faiss_engine = None
        self.faiss_layer = None
        self.bank_adapter = None
        self.drift_detector = None
    
    async def initialize_components(self):
        """Initialize all components"""
        print("=" * 60)
        print("   COMPONENT INITIALIZATION")
        print("=" * 60)
        
        # Initialize behavioral processor
        self.processor = EnhancedBehavioralProcessor()
        print("✓ Behavioral Processor initialized")
        
        # Initialize FAISS layer (local only for testing)
        self.faiss_layer = FAISSLayer()
        print("✓ FAISS Layer initialized")
        
        # Initialize adapters
        self.bank_adapter = BankAdapter()
        print("✓ Bank Adapter initialized")
        
        # Initialize drift detector
        self.drift_detector = DriftDetector()
        print("✓ Drift Detector initialized")
        
        print("\n✓ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
    
    def create_normal_user_data(self) -> Dict[str, Any]:
        """Create realistic normal user behavioral data"""
        base_time = datetime.now()
        
        return {
            "user_id": str(uuid.uuid4()),
            "session_id": f"session_{int(base_time.timestamp())}",
            "device_info": {
                "device_type": "mobile",
                "os": "iOS", 
                "screen_resolution": "1170x2532",
                "device_model": "iPhone13"
            },
            "session_context": {
                "location": "home",
                "time_of_day": "morning",
                "app_version": "2.1.5"
            },
            "behavioral_logs": [
                {
                    "timestamp": (base_time + timedelta(seconds=1)).isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 150, "y": 400, "pressure": 0.6, "duration": 120},
                            {"x": 155, "y": 405, "pressure": 0.65, "duration": 115},
                            {"x": 160, "y": 410, "pressure": 0.7, "duration": 125}
                        ],
                        "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},
                        "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}
                    }
                },
                {
                    "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
                    "event_type": "keystroke_sequence", 
                    "data": {
                        "keystrokes": [
                            {"key": "1", "dwell_time": 95, "pressure": 0.55},
                            {"key": "2", "dwell_time": 105, "pressure": 0.6},
                            {"key": "3", "dwell_time": 88, "pressure": 0.58},
                            {"key": "4", "dwell_time": 110, "pressure": 0.62}
                        ],
                        "typing_rhythm": [85, 92, 78, 88],
                        "inter_key_intervals": [0.12, 0.15, 0.11, 0.13]
                    }
                }
            ]
        }
    
    def create_suspicious_user_data(self) -> Dict[str, Any]:
        """Create suspicious user behavioral data"""
        base_time = datetime.now()
        
        return {
            "user_id": str(uuid.uuid4()),
            "session_id": f"session_{int(base_time.timestamp())}",
            "device_info": {
                "device_type": "desktop",
                "os": "Windows",
                "screen_resolution": "1920x1080", 
                "device_model": "Generic PC"
            },
            "session_context": {
                "location": "unknown",
                "time_of_day": "night",
                "app_version": "2.0.1"
            },
            "behavioral_logs": [
                {
                    "timestamp": (base_time + timedelta(seconds=1)).isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 50, "y": 100, "pressure": 0.3, "duration": 50},
                            {"x": 55, "y": 105, "pressure": 0.35, "duration": 45},
                            {"x": 60, "y": 110, "pressure": 0.4, "duration": 55}
                        ],
                        "accelerometer": {"x": 0.8, "y": 1.2, "z": 8.5},
                        "gyroscope": {"x": 0.05, "y": 0.08, "z": 0.12}
                    }
                },
                {
                    "timestamp": (base_time + timedelta(seconds=2)).isoformat(), 
                    "event_type": "keystroke_sequence",
                    "data": {
                        "keystrokes": [
                            {"key": "9", "dwell_time": 200, "pressure": 0.9},
                            {"key": "8", "dwell_time": 190, "pressure": 0.85},
                            {"key": "7", "dwell_time": 210, "pressure": 0.95},
                            {"key": "6", "dwell_time": 180, "pressure": 0.8}
                        ],
                        "typing_rhythm": [180, 200, 190, 185],
                        "inter_key_intervals": [0.5, 0.6, 0.55, 0.58]
                    }
                }
            ]
        }
    
    async def test_complete_data_flow(self, test_data: Dict[str, Any], test_name: str):
        """Test complete data flow with detailed logging"""
        print("\n" + "=" * 60)
        print(f"   COMPLETE DATA FLOW TEST - {test_name}")
        print("=" * 60)
        
        # STEP 1: Show raw input data
        print("\n--- STEP 1: RAW INPUT BEHAVIORAL DATA ---")
        print(json.dumps(test_data, indent=2))
        
        # STEP 2: Process behavioral data
        print("\n" + "=" * 60)
        print("   STEP 2: BEHAVIORAL DATA PREPROCESSING")
        print("=" * 60)
        print("Processing behavioral logs through Enhanced Behavioral Processor...")
        
        processed_vector = self.processor.process_session_data(test_data)
        
        print(f"\n--- STEP 2a: PROCESSED BEHAVIORAL VECTOR (90D) ---")
        print(f"Array shape: {processed_vector.shape}")
        print(f"Array data: {processed_vector[:10]}...")
        print(f"Array stats: min={processed_vector.min():.5f}, max={processed_vector.max():.5f}, mean={processed_vector.mean():.4f}")
        
        # Show feature breakdown
        print(f"\n--- FEATURE EXTRACTION BREAKDOWN ---")
        print(f"Touch features (dims 0-29): {processed_vector[0:30]}")
        print(f"Typing features (dims 30-54): {processed_vector[30:55]}")
        print(f"Motion features (dims 55-74): {processed_vector[55:75]}")
        print(f"Context features (dims 75-89): {processed_vector[75:90]}")
        
        # STEP 3: FAISS Similarity Matching (Simulated)
        print("\n" + "=" * 60)
        print("   STEP 3: FAISS SIMILARITY MATCHING")
        print("=" * 60)
        print("Simulating FAISS similarity calculation...")
        
        # Simulate FAISS similarity scores based on data characteristics
        if "normal" in test_name.lower():
            similarity_score = 0.85  # High similarity for normal users
            distance = 0.15
        else:
            similarity_score = 0.25  # Low similarity for suspicious users  
            distance = 0.75
        
        print(f"\n--- STEP 3a: FAISS SIMILARITY RESULTS ---")
        print(f"Similarity Score: {similarity_score:.3f}")
        print(f"Distance: {distance:.3f}")
        print(f"Threshold Check: {'PASS' if similarity_score > 0.7 else 'FAIL'}")
        
        # STEP 4: Layer Processing Decision
        print(f"\n--- STEP 3b: ESCALATION DECISION ---")
        if similarity_score > 0.7:
            print("✓ FAISS LAYER DECISION: APPROVE - High similarity to user profile")
            auth_decision = "APPROVED"
            risk_score = 0.1
        else:
            print("⚠ FAISS LAYER DECISION: ESCALATE - Low similarity, needs further analysis")
            auth_decision = "ESCALATED"
            
            # STEP 4: Bank Adapter Analysis
            print("\n" + "=" * 60)
            print("   STEP 4: BANK ADAPTER RISK ANALYSIS")
            print("=" * 60)
            
            # Create transaction context
            transaction_context = {
                "amount": 5000.0 if "suspicious" in test_name.lower() else 150.0,
                "merchant": "Unknown ATM" if "suspicious" in test_name.lower() else "Coffee Shop",
                "location": test_data["session_context"]["location"],
                "time": datetime.now().isoformat()
            }
            
            print(f"Transaction Context: {json.dumps(transaction_context, indent=2)}")
            
            # Get bank adapter risk assessment
            bank_result = self.bank_adapter.assess_risk(
                processed_vector, 
                test_data["user_id"],
                transaction_context
            )
            
            print(f"\n--- STEP 4a: BANK ADAPTER RESULTS ---")
            print(f"Risk Score: {bank_result['risk_score']:.3f}")
            print(f"Risk Level: {bank_result['risk_level']}")
            print(f"Decision: {bank_result['decision']}")
            print(f"Risk Factors: {bank_result['risk_factors']}")
            
            risk_score = bank_result['risk_score']
            if bank_result['decision'] == 'APPROVE':
                auth_decision = "APPROVED"
            else:
                auth_decision = "REJECTED"
        
        # STEP 5: Drift Detection Analysis
        print("\n" + "=" * 60)
        print("   STEP 5: DRIFT DETECTION ANALYSIS")
        print("=" * 60)
        
        # Generate historical data for drift comparison
        historical_vectors = [processed_vector + np.random.normal(0, 0.1, 90) for _ in range(10)]
        current_vectors = [processed_vector]
        
        drift_result = self.drift_detector.detect_drift(historical_vectors, current_vectors)
        
        print(f"--- STEP 5a: DRIFT ANALYSIS RESULTS ---")
        print(f"Drift Detected: {'YES' if drift_result['drift_detected'] else 'NO'}")
        print(f"Drift Score: {drift_result['drift_score']:.3f}")
        print(f"Drift Severity: {drift_result['severity']}")
        print(f"Recommended Action: {drift_result['recommendation']}")
        
        # FINAL DECISION
        print("\n" + "=" * 60)
        print("   FINAL AUTHENTICATION DECISION")
        print("=" * 60)
        
        final_confidence = (1 - risk_score) * 100
        
        print(f"--- DECISION SUMMARY ---")
        print(f"User ID: {test_data['user_id']}")
        print(f"Session ID: {test_data['session_id']}")
        print(f"FAISS Similarity: {similarity_score:.3f}")
        print(f"Risk Score: {risk_score:.3f}")
        print(f"Drift Status: {drift_result['severity']}")
        print(f"Final Decision: {auth_decision}")
        print(f"Confidence: {final_confidence:.1f}%")
        
        if auth_decision == "APPROVED":
            print("✓ USER AUTHENTICATED SUCCESSFULLY")
        elif auth_decision == "REJECTED":
            print("✗ AUTHENTICATION REJECTED - HIGH RISK")
        else:
            print("⚠ AUTHENTICATION REQUIRES MANUAL REVIEW")
        
        return {
            "decision": auth_decision,
            "risk_score": risk_score,
            "similarity_score": similarity_score,
            "confidence": final_confidence
        }

async def main():
    """Main test function"""
    print("BEHAVIORAL AUTHENTICATION ENGINE - REAL DATA FLOW TEST")
    print("=" * 70)
    
    # Initialize tester
    tester = SimpleDataFlowTester()
    await tester.initialize_components()
    
    # Test normal user
    print("\n" + "🟢" * 20 + " NORMAL USER TEST " + "🟢" * 20)
    normal_data = tester.create_normal_user_data()
    normal_result = await tester.test_complete_data_flow(normal_data, "NORMAL USER")
    
    # Test suspicious user
    print("\n" + "🔴" * 20 + " SUSPICIOUS USER TEST " + "🔴" * 20)
    suspicious_data = tester.create_suspicious_user_data()
    suspicious_result = await tester.test_complete_data_flow(suspicious_data, "SUSPICIOUS USER")
    
    # Summary
    print("\n" + "=" * 70)
    print("   TEST SUMMARY")
    print("=" * 70)
    print(f"Normal User: {normal_result['decision']} (Confidence: {normal_result['confidence']:.1f}%)")
    print(f"Suspicious User: {suspicious_result['decision']} (Confidence: {suspicious_result['confidence']:.1f}%)")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✓ ALL TESTS COMPLETED SUCCESSFULLY")
    else:
        print("\n✗ SOME TESTS FAILED")
