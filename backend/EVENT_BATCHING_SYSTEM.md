# Event Batching System

## Overview

The Event Batching System optimizes ML Engine integration by collecting behavioral events in batches of 20 before sending them for analysis. This reduces API calls, improves performance, and provides better context for ML analysis.

## Architecture

### Core Components

#### 1. `EventBatch` Class
- Manages events for a single session
- Tracks batch size and timing
- Handles batch state (processing, flushed, etc.)

#### 2. `EventBatcher` Class
- Manages multiple session batches
- Background processing for time-based flushing
- ML Engine callback integration

#### 3. `BatchConfig` Configuration
```python
@dataclass
class BatchConfig:
    batch_size: int = 20          # Events per batch
    max_wait_time: int = 30       # Max seconds to wait
    flush_on_session_end: bool = True
    enable_batching: bool = True
```

## How It Works

### 1. Event Collection
```python
# Each behavioral event is added to the batch
was_processed = await event_batcher.add_event(session_id, behavioral_event)
```

### 2. Batch Processing Triggers
- **Size-based**: When batch reaches 20 events
- **Time-based**: When 30 seconds pass since last event
- **Manual**: On session termination or manual flush

### 3. ML Engine Integration
```python
# Batched events are sent to ML Engine
ml_result = await behavioral_event_hook(user_id, session_id, batched_events)
```

## Configuration

### Default Settings
- **Batch Size**: 20 events
- **Max Wait Time**: 30 seconds
- **Auto Flush**: Enabled on session end
- **Batching**: Enabled by default

### Custom Configuration
```python
from app.core.event_batcher import EventBatcher, BatchConfig

# Custom configuration
config = BatchConfig(
    batch_size=15,           # Smaller batches
    max_wait_time=60,        # Longer wait time
    enable_batching=True
)

# Create custom batcher
custom_batcher = EventBatcher(config)
```

## API Integration

### 1. WebSocket Events
```python
# In websocket.py - process_behavioral_data function
was_processed = await event_batcher.add_event(session_id, behavioral_event)

if was_processed:
    # Event was processed immediately (batch was full)
    session.add_behavioral_data("event_batch_processed", {
        "session_id": session_id,
        "event_type": event_type,
        "processed_immediately": True,
        "timestamp": session.last_activity.isoformat()
    })
else:
    # Event was queued for batch processing
    session.add_behavioral_data("event_batch_queued", {
        "session_id": session_id,
        "event_type": event_type,
        "queued_for_batch": True,
        "timestamp": session.last_activity.isoformat()
    })
```

### 2. Session Termination
```python
# Automatic flush on session end
await event_batcher.flush_session(session_id)
```

### 3. WebSocket Disconnect
```python
# Flush pending events on disconnect
await event_batcher.flush_session(session_id)
```

## Behavioral Events

### Event Types Logged
- `event_batch_processed` - Event processed immediately
- `event_batch_queued` - Event queued for batch
- `event_batch_flushed` - Batch flushed on session end
- `event_batch_error` - Error in batch processing
- `ml_batch_analysis_result` - ML analysis result for batch
- `ml_batch_analysis_failed` - ML analysis failed for batch
- `ml_batch_analysis_error` - Error in ML analysis

### Example Event Flow
```
1. User performs action → event_batch_queued
2. Batch reaches 20 events → event_batch_processed
3. ML Engine processes batch → ml_batch_analysis_result
4. Session ends → event_batch_flushed
```

## Debug Endpoints

### 1. Get Batch Statistics
```http
GET /api/v1/ws/debug/event-batcher
```

**Response:**
```json
{
  "event_batcher_stats": {
    "total_sessions": 5,
    "active_batches": 3,
    "total_pending_events": 45,
    "config": {
      "batch_size": 20,
      "max_wait_time": 30,
      "enable_batching": true
    },
    "is_running": true
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 2. Manual Flush All Batches
```http
POST /api/v1/ws/debug/flush-all-batches
```

**Response:**
```json
{
  "message": "All batches flushed successfully",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Performance Benefits

### 1. Reduced API Calls
- **Before**: 1 ML Engine call per event
- **After**: 1 ML Engine call per 20 events (95% reduction)

### 2. Better Context
- ML Engine receives event sequences
- Improved pattern recognition
- Better risk assessment

### 3. Improved Latency
- Reduced network overhead
- Batch processing optimization
- Background processing

## Error Handling

### 1. Batch Processing Errors
```python
try:
    await event_batcher.add_event(session_id, event)
except Exception as e:
    session.add_behavioral_data("event_batch_error", {
        "error": str(e),
        "timestamp": session.last_activity.isoformat()
    })
```

### 2. ML Engine Errors
```python
# Errors are logged but don't stop batching
session.add_behavioral_data("ml_batch_analysis_error", {
    "error": str(ml_error),
    "events_count": len(events),
    "timestamp": session.last_activity.isoformat()
})
```

### 3. Session Not Found
```python
# Graceful handling of missing sessions
if not session:
    logger.warning(f"Session {session_id} not found for ML processing")
    return
```

## Monitoring

### Key Metrics to Monitor
1. **Batch Processing Rate**
   - Events per second
   - Batches per minute
   - Average batch size

2. **Error Rates**
   - Batch processing errors
   - ML Engine errors
   - Session not found errors

3. **Performance Metrics**
   - Average processing time
   - Queue depth
   - Memory usage

### Database Queries
```sql
-- Check batch processing events
SELECT 
  event_type,
  COUNT(*) as count,
  AVG(CAST(data->>'events_count' AS INTEGER)) as avg_events
FROM behavioral_events 
WHERE event_type LIKE '%batch%'
GROUP BY event_type;

-- Monitor ML analysis results
SELECT 
  data->>'decision' as decision,
  AVG(CAST(data->>'confidence' AS FLOAT)) as avg_confidence,
  COUNT(*) as count
FROM behavioral_events 
WHERE event_type = 'ml_batch_analysis_result'
GROUP BY data->>'decision';
```

## Testing

### Test File: `backend/test_event_batching.py`

Tests cover:
1. **Batch Processing**: 25 events (2 batches)
2. **Statistics**: Batch metrics verification
3. **Manual Flush**: Forced batch processing
4. **Session Termination**: Automatic flush
5. **Error Handling**: Invalid sessions
6. **WebSocket Simulation**: Rapid event sending

### Run Test:
```bash
cd backend
python test_event_batching.py
```

## Best Practices

### 1. Configuration
- Adjust batch size based on ML Engine capacity
- Set appropriate wait times for your use case
- Monitor memory usage with large batch sizes

### 2. Error Handling
- Always handle batch processing exceptions
- Log errors for debugging
- Implement retry mechanisms for failed batches

### 3. Monitoring
- Set up alerts for high error rates
- Monitor batch processing latency
- Track ML Engine response times

### 4. Performance
- Use appropriate batch sizes (15-25 events)
- Monitor memory usage
- Implement circuit breakers for ML Engine

## Troubleshooting

### Common Issues

#### 1. Events Not Being Processed
- Check if batching is enabled
- Verify ML Engine is running
- Check for session errors

#### 2. High Memory Usage
- Reduce batch size
- Implement batch cleanup
- Monitor session count

#### 3. ML Engine Timeouts
- Reduce batch size
- Increase timeout settings
- Implement retry logic

### Debug Commands
```bash
# Check batch statistics
curl http://localhost:8000/api/v1/ws/debug/event-batcher

# Force flush all batches
curl -X POST http://localhost:8000/api/v1/ws/debug/flush-all-batches

# Check session status
curl http://localhost:8000/api/v1/log/session/{session_id}/status
```

## Future Enhancements

### 1. Dynamic Batching
- Adjust batch size based on load
- Adaptive wait times
- Priority-based batching

### 2. Advanced Error Handling
- Circuit breaker pattern
- Exponential backoff
- Dead letter queues

### 3. Performance Optimization
- Async batch processing
- Memory pooling
- Compression for large batches

### 4. Monitoring Integration
- Prometheus metrics
- Grafana dashboards
- Alerting rules

## Conclusion

The Event Batching System provides significant performance improvements while maintaining reliability and error handling. It reduces ML Engine API calls by 95% while providing better context for behavioral analysis. 