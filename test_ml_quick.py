#!/usr/bin/env python3
"""
Quick ML Engine Test - All 10 Sessions
=====================================
Fast test to see all vector representations and scores
"""

import asyncio
import aiohttp
import json
from pathlib import Path

ML_ENGINE_URL = "http://localhost:8001"
DATA_DIR = Path("data")

async def quick_ml_test():
    async with aiohttp.ClientSession() as session:
        print("🎯 QUICK ML ENGINE TEST - ALL 10 SESSIONS")
        print("="*50)
        
        results = []
        
        for i in range(1, 11):
            session_file = DATA_DIR / f"test_user_session_{i:02d}.json"
            if not session_file.exists():
                continue
                
            session_id = f"quick-test-{i}"
            
            print(f"\n🔥 SESSION {i}")
            
            # Start session
            start_data = {
                "user_id": "test-user-ml-engine",
                "session_id": session_id,
                "device_info": {"session_number": i}
            }
            
            async with session.post(f"{ML_ENGINE_URL}/session/start", json=start_data) as resp:
                start_result = await resp.json()
                print(f"   📍 Start: {start_result.get('phase')} mode")
            
            # Load and analyze data
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            analysis_data = {
                "user_id": "test-user-ml-engine", 
                "session_id": session_id,
                "logs": session_data["logs"]
            }
            
            async with session.post(f"{ML_ENGINE_URL}/analyze-mobile", json=analysis_data) as resp:
                analysis_result = await resp.json()
                
                # Extract key information
                decision = analysis_result.get('decision', 'N/A')
                confidence = analysis_result.get('confidence', 'N/A')
                session_count = analysis_result.get('session_count', 'N/A')
                phase = analysis_result.get('phase', 'N/A')
                
                details = analysis_result.get('details', {})
                vector = details.get('feature_vector', [])
                events_processed = details.get('events_processed', 'N/A')
                
                print(f"   🎯 Decision: {decision} | Confidence: {confidence}")
                print(f"   📊 Phase: {phase} | Session Count: {session_count}")
                print(f"   📈 Events: {events_processed} | Vector Dims: {len(vector) if vector else 'N/A'}")
                
                if vector and len(vector) >= 10:
                    vector_preview = [f"{v:.3f}" for v in vector[:10]]
                    print(f"   🔢 Vector: [{', '.join(vector_preview)}, ...]")
                
                results.append({
                    'session': i,
                    'decision': decision,
                    'confidence': confidence, 
                    'phase': phase,
                    'session_count': session_count,
                    'vector_preview': vector[:10] if vector else []
                })
            
            # End session
            end_data = {"session_id": session_id, "reason": "completed"}
            async with session.post(f"{ML_ENGINE_URL}/session/end", json=end_data) as resp:
                end_result = await resp.json()
                end_info = end_result.get('end_result', {})
                learning_completed = end_info.get('learning_completed', False)
                sessions_remaining = end_info.get('sessions_remaining', 'N/A')
                print(f"   🔚 End: Learning={learning_completed} | Remaining={sessions_remaining}")
        
        print(f"\n" + "="*60)
        print("📊 SUMMARY - ALL SESSIONS")
        print("="*60)
        
        learning_sessions = [r for r in results if r['session'] <= 6]
        auth_sessions = [r for r in results if r['session'] > 6]
        
        print(f"\n🧠 LEARNING PHASE (Sessions 1-6):")
        for r in learning_sessions:
            print(f"   Session {r['session']}: {r['decision']} (conf: {r['confidence']}) [count: {r['session_count']}]")
        
        print(f"\n🔐 AUTHENTICATION PHASE (Sessions 7-10):")
        for r in auth_sessions:
            print(f"   Session {r['session']}: {r['decision']} (conf: {r['confidence']}) [count: {r['session_count']}]")
        
        print(f"\n🔢 VECTOR SIMILARITY ANALYSIS:")
        if len(results) >= 2:
            import numpy as np
            vectors = [r['vector_preview'] for r in results if len(r['vector_preview']) == 10]
            
            if len(vectors) >= 2:
                print(f"   📐 Comparing first 10 dimensions across sessions:")
                for i in range(min(5, len(vectors))):
                    for j in range(i+1, min(5, len(vectors))): 
                        v1 = np.array(vectors[i])
                        v2 = np.array(vectors[j])
                        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        print(f"   Sessions {i+1} vs {j+1}: similarity = {similarity:.4f}")

if __name__ == "__main__":
    asyncio.run(quick_ml_test())
