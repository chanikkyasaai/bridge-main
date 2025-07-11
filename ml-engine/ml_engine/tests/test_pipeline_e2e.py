import asyncio
import numpy as np
import pytest
from datetime import datetime, timedelta
from ml_engine.core.engine import BankingMLEngine, BehavioralEvent, SessionContext, SessionState, AuthenticationDecision

@pytest.mark.asyncio
async def test_pipeline_e2e():
    engine = BankingMLEngine()
    await engine.initialize()

    # --- Cold Start: New User ---
    user_id = "user_cold_start"
    session_id = "sess_cold_start"
    context = SessionContext(
        session_id=session_id,
        user_id=user_id,
        device_id="dev1",
        session_start_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        session_duration_minutes=0.0,
        device_type="phone",
        device_model="modelX",
        os_version="1.0",
        app_version="1.0",
        network_type="wifi",
        time_of_day="morning",
        usage_pattern="normal",
        interaction_frequency=1.0,
        typical_session_duration=5.0,
        is_known_device=False,
        is_trusted_location=True
    )
    await engine.start_session(session_id, user_id, context)
    # Simulate a few events
    for i in range(3):
        event = BehavioralEvent(
            timestamp=datetime.utcnow(),
            event_type="type",
            features={"pressure": 0.5, "velocity": 0.2},
            session_id=session_id,
            user_id=user_id,
            device_id="dev1",
            raw_metadata={}
        )
        resp = await engine.process_behavioral_event(event)
        print("Cold Start Decision:", resp.decision)
        assert resp.decision in [AuthenticationDecision.ALLOW, AuthenticationDecision.MONITOR]

    # --- Normal User ---
    user_id = "user_normal"
    session_id = "sess_normal"
    context.user_id = user_id
    context.session_id = session_id
    context.is_known_device = True
    await engine.start_session(session_id, user_id, context)
    for i in range(5):
        event = BehavioralEvent(
            timestamp=datetime.utcnow(),
            event_type="swipe",
            features={"pressure": 0.6, "velocity": 0.3},
            session_id=session_id,
            user_id=user_id,
            device_id="dev1",
            raw_metadata={}
        )
        resp = await engine.process_behavioral_event(event)
        print("Normal User Decision:", resp.decision)
        assert resp.decision == AuthenticationDecision.ALLOW

    # --- Attack/Bot Scenario ---
    user_id = "user_attack"
    session_id = "sess_attack"
    context.user_id = user_id
    context.session_id = session_id
    context.is_known_device = False
    await engine.start_session(session_id, user_id, context)
    for i in range(5):
        event = BehavioralEvent(
            timestamp=datetime.utcnow(),
            event_type="tap",
            features={"pressure": 1.0, "velocity": 5.0},  # Abnormal
            session_id=session_id,
            user_id=user_id,
            device_id="dev1",
            raw_metadata={}
        )
        resp = await engine.process_behavioral_event(event)
        print("Attack/Bot Decision:", resp.decision)
        assert resp.decision in [AuthenticationDecision.STEP_UP_AUTH, AuthenticationDecision.PERMANENT_BLOCK, AuthenticationDecision.CHALLENGE, AuthenticationDecision.MONITOR]

    print("E2E pipeline test completed.") 