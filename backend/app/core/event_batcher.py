"""
Event Batching System for ML Engine Integration
Handles batching of behavioral events before sending to ML Engine
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """Configuration for event batching"""
    batch_size: int = 20
    max_wait_time: int = 30  # seconds
    flush_on_session_end: bool = True
    enable_batching: bool = True

class EventBatch:
    """Represents a batch of events for a session"""
    
    def __init__(self, session_id: str, config: BatchConfig):
        self.session_id = session_id
        self.config = config
        self.events: List[Dict[str, Any]] = []
        self.created_at = datetime.utcnow()
        self.last_event_at = datetime.utcnow()
        self.is_processing = False
        self.is_flushed = False
    
    def add_event(self, event: Dict[str, Any]) -> bool:
        """Add event to batch, return True if batch is full"""
        if self.is_flushed or self.is_processing:
            return False
        
        self.events.append(event)
        self.last_event_at = datetime.utcnow()
        
        # Check if batch is full
        return len(self.events) >= self.config.batch_size
    
    def is_ready_for_processing(self) -> bool:
        """Check if batch is ready for processing"""
        if self.is_flushed or self.is_processing:
            return False
        
        # Check if batch is full
        if len(self.events) >= self.config.batch_size:
            return True
        
        # Check if max wait time has elapsed
        time_since_last = datetime.utcnow() - self.last_event_at
        if time_since_last.total_seconds() >= self.config.max_wait_time:
            return True
        
        return False
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events in the batch"""
        return self.events.copy()
    
    def clear(self):
        """Clear the batch after processing"""
        self.events.clear()
        self.is_processing = False
    
    def mark_processing(self):
        """Mark batch as being processed"""
        self.is_processing = True
    
    def mark_flushed(self):
        """Mark batch as flushed"""
        self.is_flushed = True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch summary"""
        return {
            "session_id": self.session_id,
            "event_count": len(self.events),
            "created_at": self.created_at.isoformat(),
            "last_event_at": self.last_event_at.isoformat(),
            "is_processing": self.is_processing,
            "is_flushed": self.is_flushed
        }

class EventBatcher:
    """Manages event batching for multiple sessions"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.batches: Dict[str, EventBatch] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.ml_callback: Optional[Callable] = None
        self.background_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    def set_ml_callback(self, callback: Callable):
        """Set the ML engine callback function"""
        self.ml_callback = callback
    
    async def start(self):
        """Start the event batcher background processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.background_task = asyncio.create_task(self._background_processor())
        logger.info("Event batcher started")
    
    async def stop(self):
        """Stop the event batcher and flush all batches"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background task
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        # Flush all remaining batches
        await self._flush_all_batches()
        logger.info("Event batcher stopped")
    
    async def add_event(self, session_id: str, event: Dict[str, Any]) -> bool:
        """
        Add event to batch for a session
        Returns True if batch was processed immediately
        """
        if not self.config.enable_batching:
            # If batching is disabled, process immediately
            if self.ml_callback:
                await self._process_single_event(session_id, event)
            return True
        
        # Get or create batch for session
        if session_id not in self.batches:
            self.batches[session_id] = EventBatch(session_id, self.config)
        
        batch = self.batches[session_id]
        
        # Add event to batch
        is_full = batch.add_event(event)
        
        # If batch is full, process immediately
        if is_full:
            await self._process_batch(session_id)
            return True
        
        return False
    
    async def flush_session(self, session_id: str):
        """Flush all events for a specific session"""
        if session_id in self.batches:
            batch = self.batches[session_id]
            if batch.events and not batch.is_flushed:
                await self._process_batch(session_id)
                batch.mark_flushed()
    
    async def _background_processor(self):
        """Background task to process batches based on time"""
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Check all batches for processing
                sessions_to_process = []
                for session_id, batch in self.batches.items():
                    if batch.is_ready_for_processing() and not batch.is_processing:
                        sessions_to_process.append(session_id)
                
                # Process batches
                for session_id in sessions_to_process:
                    await self._process_batch(session_id)
                
                # Clean up empty batches
                self._cleanup_empty_batches()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
    
    async def _process_batch(self, session_id: str):
        """Process a batch of events for a session"""
        if session_id not in self.batches:
            return
        
        batch = self.batches[session_id]
        
        if not batch.events or batch.is_processing:
            return
        
        batch.mark_processing()
        
        try:
            # Get events from batch
            events = batch.get_events()
            
            if not events:
                batch.clear()
                return
            
            # Process with ML engine
            if self.ml_callback:
                await self._process_with_ml(session_id, events)
            
            # Clear the batch
            batch.clear()
            
            logger.info(f"Processed batch for session {session_id}: {len(events)} events")
            
        except Exception as e:
            logger.error(f"Error processing batch for session {session_id}: {e}")
            batch.is_processing = False  # Allow retry
    
    async def _process_single_event(self, session_id: str, event: Dict[str, Any]):
        """Process a single event (when batching is disabled)"""
        try:
            if self.ml_callback:
                await self._process_with_ml(session_id, [event])
        except Exception as e:
            logger.error(f"Error processing single event for session {session_id}: {e}")
    
    async def _process_with_ml(self, session_id: str, events: List[Dict[str, Any]]):
        """Process events with ML engine"""
        if not self.ml_callback:
            return
        
        try:
            # Get session info for ML processing
            from app.core.session_manager import session_manager
            session = session_manager.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found for ML processing")
                return
            
            # Call ML engine with batched events
            ml_result = await self.ml_callback(session.user_id, session_id, events)
            
            if ml_result and ml_result.get("status") in ["success", "fallback"]:
                # Update session with ML decision
                decision = ml_result.get("decision", "allow")
                confidence = ml_result.get("confidence", 0.5)
                
                session.add_behavioral_data("ml_batch_analysis_result", {
                    "session_id": session_id,
                    "decision": decision,
                    "confidence": confidence,
                    "events_processed": len(events),
                    "batch_size": len(events),
                    "timestamp": session.last_activity.isoformat()
                })
                
                # Handle ML decision
                if decision == "block" and confidence > 0.8:
                    session.block_session("ML Engine detected suspicious behavior in batch")
                    session.add_behavioral_data("session_blocked", {
                        "session_id": session_id,
                        "reason": "ML Engine batch decision",
                        "confidence": confidence,
                        "events_analyzed": len(events),
                        "timestamp": session.last_activity.isoformat()
                    })
            else:
                session.add_behavioral_data("ml_batch_analysis_failed", {
                    "session_id": session_id,
                    "error": ml_result.get("message", "Unknown error") if ml_result else "No result",
                    "events_count": len(events),
                    "timestamp": session.last_activity.isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error in ML processing for session {session_id}: {e}")
            
            # Log error in session
            session = session_manager.get_session(session_id)
            if session:
                session.add_behavioral_data("ml_batch_analysis_error", {
                    "session_id": session_id,
                    "error": str(e),
                    "events_count": len(events),
                    "timestamp": session.last_activity.isoformat()
                })
    
    async def _flush_all_batches(self):
        """Flush all remaining batches"""
        for session_id in list(self.batches.keys()):
            await self.flush_session(session_id)
    
    def _cleanup_empty_batches(self):
        """Remove empty batches to free memory"""
        empty_sessions = []
        for session_id, batch in self.batches.items():
            if not batch.events and batch.is_flushed:
                empty_sessions.append(session_id)
        
        for session_id in empty_sessions:
            del self.batches[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        total_events = sum(len(batch.events) for batch in self.batches.values())
        active_batches = len([b for b in self.batches.values() if b.events])
        
        return {
            "total_sessions": len(self.batches),
            "active_batches": active_batches,
            "total_pending_events": total_events,
            "config": {
                "batch_size": self.config.batch_size,
                "max_wait_time": self.config.max_wait_time,
                "enable_batching": self.config.enable_batching
            },
            "is_running": self.is_running
        }

# Global event batcher instance
event_batcher = EventBatcher() 