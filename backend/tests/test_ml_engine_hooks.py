import pytest
import asyncio
import logging

from ml_engine_client import (
    session_created_hook, behavioral_event_hook, session_ended_hook, get_session_ml_status
)

# Dummy data for testing
DUMMY_SESSION_ID = "test-session-123"
DUMMY_USER_ID = "user-abc"
DUMMY_PHONE = "9999999999"
DUMMY_DEVICE_ID = "device-xyz"
DUMMY_CONTEXT = {
    "device_type": "mobile",
    "device_model": "Pixel 5",
    "os_version": "Android 13",
    "app_version": "1.0.0",
    "network_type": "wifi",
    "location_data": {"lat": 12.34, "lon": 56.78},
    "is_known_device": True,
    "is_trusted_location": False
}
DUMMY_EVENT_TYPE = "test_event"
DUMMY_EVENT_DATA = {
    "features": {"x": 1, "y": 2},
    "metadata": {"test": True}
}

@pytest.mark.asyncio
async def test_session_created_hook():
    result = await session_created_hook(
        session_id=DUMMY_SESSION_ID,
        user_id=DUMMY_USER_ID,
        phone=DUMMY_PHONE,
        device_id=DUMMY_DEVICE_ID,
        context=DUMMY_CONTEXT
    )
    print(f"session_created_hook result: {result}")
    assert result is True or result is False  # Should be a boolean

@pytest.mark.asyncio
async def test_behavioral_event_hook():
    result = await behavioral_event_hook(
        session_id=DUMMY_SESSION_ID,
        user_id=DUMMY_USER_ID,
        device_id=DUMMY_DEVICE_ID,
        event_type=DUMMY_EVENT_TYPE,
        event_data=DUMMY_EVENT_DATA
    )
    print(f"behavioral_event_hook result: {result}")
    # Can be None or a dict depending on ML engine status
    assert result is None or isinstance(result, dict)

@pytest.mark.asyncio
async def test_get_session_ml_status():
    result = await get_session_ml_status(DUMMY_SESSION_ID)
    print(f"get_session_ml_status result: {result}")
    # Can be None or a dict
    assert result is None or isinstance(result, dict)

@pytest.mark.asyncio
async def test_session_ended_hook():
    result = await session_ended_hook(
        session_id=DUMMY_SESSION_ID,
        user_id=DUMMY_USER_ID,
        final_decision="test_end",
        session_stats={"duration_minutes": 1, "total_events": 1}
    )
    print(f"session_ended_hook result: {result}")
    assert result is True or result is False 