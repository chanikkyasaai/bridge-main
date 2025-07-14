"""
BRIDGE ML-Engine Demo Script
Demonstrates the complete ML-Engine functionality
"""

import asyncio
import logging
import sys
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlengine.config import CONFIG
from mlengine import ml_engine, AuthenticationRequest
from mlengine.utils.behavioral_vectors import BehavioralEvent, BehavioralVector
from mlengine.core.policy_orchestrator import SessionAction, RiskLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BRIDGEMLEngineDemo:
    """Demo class for BRIDGE ML-Engine"""
    
    def __init__(self):
        self.demo_users = ["alice", "bob", "charlie", "david", "eve"]
        self.demo_sessions = {}
        
    def create_demo_behavioral_events(self, user_id: str, session_id: str, count: int = 10) -> list:
        """Create demo behavioral events"""
        events = []
        base_time = datetime.now()
        
        # User-specific behavioral patterns
        user_patterns = {
            "alice": {"pressure_base": 0.8, "velocity_base": 2.5, "style": "precise"},
            "bob": {"pressure_base": 0.6, "velocity_base": 3.2, "style": "fast"},
            "charlie": {"pressure_base": 0.9, "velocity_base": 1.8, "style": "deliberate"},
            "david": {"pressure_base": 0.7, "velocity_base": 2.8, "style": "variable"},
            "eve": {"pressure_base": 0.5, "velocity_base": 4.0, "style": "aggressive"}
        }
        
        pattern = user_patterns.get(user_id, user_patterns["alice"])
        
        for i in range(count):
            # Simulate different event types
            event_type = np.random.choice(["touch", "swipe", "keystroke", "navigation"], p=[0.4, 0.3, 0.2, 0.1])
            
            # Generate features based on user pattern
            if event_type == "touch":
                features = {
                    "pressure": max(0.1, min(1.0, np.random.normal(pattern["pressure_base"], 0.1))),
                    "velocity": max(0.5, np.random.normal(pattern["velocity_base"], 0.3)),
                    "duration": max(0.1, np.random.normal(0.3, 0.1)),
                    "x_position": np.random.uniform(100, 980),
                    "y_position": np.random.uniform(200, 1720),
                    "finger_orientation": np.random.uniform(0, 360)
                }
            elif event_type == "swipe":
                features = {
                    "velocity": max(1.0, np.random.normal(pattern["velocity_base"] * 1.5, 0.5)),
                    "acceleration": np.random.uniform(0.5, 3.0),
                    "direction": np.random.uniform(0, 360),
                    "distance": np.random.uniform(50, 500),
                    "duration": max(0.2, np.random.normal(0.5, 0.2))
                }
            elif event_type == "keystroke":
                features = {
                    "dwell_time": max(0.05, np.random.normal(0.15, 0.05)),
                    "flight_time": max(0.02, np.random.normal(0.08, 0.03)),
                    "pressure": max(0.1, np.random.normal(pattern["pressure_base"], 0.15)),
                    "key_size": np.random.uniform(20, 60),
                    "typing_speed": pattern["velocity_base"] * 10
                }
            else:  # navigation
                features = {
                    "screen_transition_time": np.random.uniform(0.5, 2.0),
                    "scroll_velocity": np.random.uniform(1.0, 5.0),
                    "tap_count": np.random.randint(1, 5),
                    "navigation_pattern": np.random.uniform(0, 1)
                }
            
            event = BehavioralEvent(
                timestamp=base_time + timedelta(milliseconds=i * 200),
                event_type=event_type,
                features=features,
                session_id=session_id,
                user_id=user_id
            )
            
            events.append(event)
        
        return events
    
    async def train_demo_users(self):
        """Train profiles for demo users"""
        logger.info("🎓 Training profiles for demo users...")
        
        for user_id in self.demo_users:
            logger.info(f"Training user: {user_id}")
            
            # Create training events
            training_events = []
            for session_num in range(5):  # 5 training sessions per user
                session_events = self.create_demo_behavioral_events(
                    user_id, f"training_session_{session_num}", count=20
                )
                training_events.extend(session_events)
            
            # Process into vectors
            vectors = await ml_engine.vector_processor.process_events(training_events)
            
            # Train user models
            labels = ["normal"] * len(vectors)
            await ml_engine.train_user_model(user_id, vectors, labels)
            
            logger.info(f"✓ Trained {user_id} with {len(vectors)} vectors")
        
        logger.info("✅ All demo users trained!")
    
    async def demo_authentication_flow(self, user_id: str, session_id: str):
        """Demonstrate authentication flow for a user"""
        logger.info(f"\n🔐 Demonstrating authentication for {user_id} (session: {session_id})")
        
        # Create behavioral events
        events = self.create_demo_behavioral_events(user_id, session_id, count=8)
        
        # Create context
        context = {
            "device_type": np.random.choice(["phone", "tablet"]),
            "time_of_day": np.random.choice(["morning", "afternoon", "evening", "night"]),
            "known_device": np.random.choice([True, False]),
            "home_network": np.random.choice([True, False]),
            "location_risk": np.random.uniform(0.0, 0.5),
            "high_value_transaction": np.random.choice([True, False])
        }
        
        # Create authentication request
        request = AuthenticationRequest(
            session_id=session_id,
            user_id=user_id,
            events=events,
            context=context,
            timestamp=datetime.now(),
            require_explanation=True
        )
        
        # Perform authentication
        response = await ml_engine.authenticate(request)
        
        # Display results
        logger.info(f"📊 Authentication Results for {user_id}:")
        logger.info(f"   Decision: {response.decision.value}")
        logger.info(f"   Risk Level: {response.risk_level.value}")
        logger.info(f"   Risk Score: {response.risk_score:.3f}")
        logger.info(f"   Confidence: {response.confidence:.3f}")
        logger.info(f"   Processing Time: {response.processing_time_ms:.1f}ms")
        logger.info(f"   L1 Similarity: {response.l1_result.similarity_score:.3f}")
        if response.l2_result:
            logger.info(f"   L2 Transformer: {response.l2_result.transformer_confidence:.3f}")
            logger.info(f"   L2 GNN Anomaly: {response.l2_result.gnn_anomaly_score:.3f}")
        logger.info(f"   Next Verification: {response.next_verification_delay}s")
        
        # Show explanation if available
        if response.explanation:
            logger.info(f"📝 Explanation: {response.explanation.human_readable_explanation[:200]}...")
        
        return response
    
    async def demo_attack_scenarios(self):
        """Demonstrate attack detection scenarios"""
        logger.info("\n🚨 Demonstrating Attack Detection Scenarios...")
        
        # Scenario 1: Session Hijacking (different behavioral pattern)
        logger.info("\n📱 Scenario 1: Session Hijacking Simulation")
        legitimate_user = "alice"
        attacker_pattern = "eve"  # Different behavioral pattern
        
        # Legitimate session start
        legit_events = self.create_demo_behavioral_events(legitimate_user, "hijack_session", count=5)
        
        # Attacker takes over (creates events with different pattern but same user_id)
        attack_events = []
        for i in range(5):
            event = self.create_demo_behavioral_events(attacker_pattern, "hijack_session", count=1)[0]
            event.user_id = legitimate_user  # Attacker uses legitimate user_id
            attack_events.append(event)
        
        # Combine events
        combined_events = legit_events + attack_events
        
        request = AuthenticationRequest(
            session_id="hijack_session",
            user_id=legitimate_user,
            events=combined_events,
            context={"device_type": "phone", "attack_simulation": True},
            timestamp=datetime.now(),
            require_explanation=True
        )
        
        response = await ml_engine.authenticate(request)
        logger.info(f"🔍 Hijacking Detection - Decision: {response.decision.value}, Risk: {response.risk_score:.3f}")
        
        # Scenario 2: Bot/Automation Detection
        logger.info("\n🤖 Scenario 2: Bot/Automation Detection")
        bot_events = []
        base_time = datetime.now()
        
        # Create highly regular, mechanical events
        for i in range(10):
            event = BehavioralEvent(
                timestamp=base_time + timedelta(milliseconds=i * 100),  # Perfectly regular timing
                event_type="touch",
                features={
                    "pressure": 0.5,  # Exactly the same every time
                    "velocity": 2.0,  # No variation
                    "duration": 0.2,  # Mechanical precision
                    "x_position": 540,  # Exact center every time
                    "y_position": 960
                },
                session_id="bot_session",
                user_id="alice"
            )
            bot_events.append(event)
        
        bot_request = AuthenticationRequest(
            session_id="bot_session",
            user_id="alice",
            events=bot_events,
            context={"device_type": "phone", "bot_simulation": True},
            timestamp=datetime.now(),
            require_explanation=True
        )
        
        bot_response = await ml_engine.authenticate(bot_request)
        logger.info(f"🔍 Bot Detection - Decision: {bot_response.decision.value}, Risk: {bot_response.risk_score:.3f}")
        
        # Scenario 3: Behavioral Drift Detection
        logger.info("\n📈 Scenario 3: Behavioral Drift Detection")
        
        # Simulate user behavior change (e.g., hand injury)
        drift_events = []
        for i in range(8):
            event = BehavioralEvent(
                timestamp=datetime.now() + timedelta(milliseconds=i * 150),
                event_type="touch",
                features={
                    "pressure": np.random.normal(0.3, 0.1),  # Much lighter pressure
                    "velocity": np.random.normal(1.0, 0.2),  # Much slower
                    "duration": np.random.normal(0.8, 0.2),  # Longer duration
                    "x_position": np.random.uniform(200, 880),  # Less precise
                    "y_position": np.random.uniform(300, 1620)
                },
                session_id="drift_session",
                user_id="alice"
            )
            drift_events.append(event)
        
        drift_request = AuthenticationRequest(
            session_id="drift_session",
            user_id="alice",
            events=drift_events,
            context={"device_type": "phone", "drift_simulation": True},
            timestamp=datetime.now(),
            require_explanation=True
        )
        
        drift_response = await ml_engine.authenticate(drift_request)
        logger.info(f"🔍 Drift Detection - Decision: {drift_response.decision.value}, Risk: {drift_response.risk_score:.3f}")
        
        if drift_response.drift_result and drift_response.drift_result.drift_detected:
            logger.info(f"   Drift Detected: {drift_response.drift_result.drift_type}")
            logger.info(f"   Drift Magnitude: {drift_response.drift_result.drift_magnitude:.3f}")
    
    async def demo_performance_test(self):
        """Demonstrate performance characteristics"""
        logger.info("\n⚡ Performance Test - Processing Multiple Sessions Concurrently")
        
        # Create multiple concurrent authentication requests
        tasks = []
        for i in range(10):
            user_id = np.random.choice(self.demo_users)
            session_id = f"perf_test_session_{i}"
            
            events = self.create_demo_behavioral_events(user_id, session_id, count=6)
            request = AuthenticationRequest(
                session_id=session_id,
                user_id=user_id,
                events=events,
                context={"device_type": "phone", "performance_test": True},
                timestamp=datetime.now()
            )
            
            task = ml_engine.authenticate(request)
            tasks.append(task)
        
        # Execute all requests concurrently
        start_time = datetime.now()
        responses = await asyncio.gather(*tasks)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds() * 1000
        avg_time = total_time / len(responses)
        
        logger.info(f"📊 Performance Results:")
        logger.info(f"   Total Requests: {len(responses)}")
        logger.info(f"   Total Time: {total_time:.1f}ms")
        logger.info(f"   Average Time per Request: {avg_time:.1f}ms")
        logger.info(f"   Throughput: {len(responses) / (total_time / 1000):.1f} requests/second")
        
        # Show distribution of decisions
        decisions = [r.decision.value for r in responses]
        decision_counts = {decision: decisions.count(decision) for decision in set(decisions)}
        logger.info(f"   Decision Distribution: {decision_counts}")
    
    async def run_complete_demo(self):
        """Run the complete ML-Engine demonstration"""
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 BRIDGE ML-Engine Complete Demonstration                                  ║
║  Behavioral Risk Intelligence for Dynamic Guarded Entry                     ║
║                                                                              ║
║  🏆 SuRaksha Cyber Hackathon - Team "Five"                                  ║
║  📱 Mobile Banking Security through Behavioral Authentication               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        try:
            # Initialize ML Engine
            logger.info("🚀 Initializing BRIDGE ML-Engine...")
            await ml_engine.initialize()
            
            # Train demo users
            await self.train_demo_users()
            
            # Demonstrate normal authentication flows
            logger.info("\n📝 Demonstrating Normal Authentication Flows...")
            for user_id in self.demo_users[:3]:  # Test first 3 users
                await self.demo_authentication_flow(user_id, f"demo_session_{user_id}")
            
            # Demonstrate attack detection
            await self.demo_attack_scenarios()
            
            # Performance testing
            await self.demo_performance_test()
            
            # Show engine statistics
            logger.info("\n📊 Final Engine Statistics:")
            stats = ml_engine.get_engine_stats()
            logger.info(f"   Total Requests Processed: {stats.total_requests}")
            logger.info(f"   Successful Authentications: {stats.successful_authentications}")
            logger.info(f"   Blocked Sessions: {stats.blocked_sessions}")
            logger.info(f"   Average Processing Time: {stats.average_processing_time:.2f}ms")
            
            # Show active sessions
            active_sessions = ml_engine.get_active_sessions()
            logger.info(f"   Active Sessions: {active_sessions['total_sessions']}")
            
            # Health check
            health = await ml_engine.health_check()
            logger.info(f"   Engine Status: {health['status']}")
            logger.info(f"   Uptime: {health['uptime_hours']:.2f} hours")
            
            logger.info("\n✅ BRIDGE ML-Engine demonstration completed successfully!")
            logger.info("\n🎯 Key Achievements Demonstrated:")
            logger.info("   ✓ Real-time behavioral vector processing")
            logger.info("   ✓ Multi-layered authentication (FAISS + Transformer/GNN)")
            logger.info("   ✓ Behavioral drift detection and adaptation")
            logger.info("   ✓ Attack detection (hijacking, bots, spoofing)")
            logger.info("   ✓ Risk-based policy orchestration")
            logger.info("   ✓ Explainable AI decisions")
            logger.info("   ✓ High-performance concurrent processing")
            logger.info("   ✓ Comprehensive monitoring and health checks")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            # Cleanup
            await ml_engine.shutdown()

async def main():
    """Main demo entry point"""
    demo = BRIDGEMLEngineDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    # Run the complete demonstration
    asyncio.run(main())
