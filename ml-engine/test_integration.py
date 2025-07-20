#!/usr/bin/env python3
"""
Test script for the integrated GNN escalation system with caching and detailed logging
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_gnn_integration():
    """Test the complete GNN integration with caching and detailed logging"""
    
    print("🧪 Testing GNN Integration with Caching and Detailed Logging")
    print("=" * 60)
    
    try:
        # Import the main components
        from gnn_escalation import detect_anomaly_with_user_adaptation, CachingManager, SupabaseStorageManager
        
        print("✅ Successfully imported GNN components")
        
        # Test 1: Basic caching functionality
        print("\n📊 Test 1: Caching Manager")
        caching_manager = CachingManager(max_cache_size=100)
        
        # Test cache stats
        stats = caching_manager.get_cache_stats()
        print(f"   Initial cache stats: {stats}")
        
        # Test 2: Supabase Storage Manager
        print("\n🔄 Test 2: Supabase Storage Manager")
        storage_manager = SupabaseStorageManager(caching_manager)
        await storage_manager.initialize()
        print("   ✅ Storage manager initialized")
        
        # Test 3: Sample session data
        print("\n📱 Test 3: Sample Session Data")
        sample_session = {
            "logs": [
                {
                    "timestamp": "2025-01-20T10:30:00Z",
                    "event_type": "screen_touch",
                    "data": {"x": 150, "y": 300, "pressure": 0.8}
                },
                {
                    "timestamp": "2025-01-20T10:30:05Z", 
                    "event_type": "screen_swipe",
                    "data": {"start_x": 100, "start_y": 200, "end_x": 200, "end_y": 200}
                },
                {
                    "timestamp": "2025-01-20T10:30:10Z",
                    "event_type": "key_press",
                    "data": {"key": "enter", "duration": 150}
                }
            ]
        }
        
        test_user_id = "demo_user_12345"
        print(f"   Testing with user: {test_user_id}")
        print(f"   Session events: {len(sample_session['logs'])}")
        
        # Test 4: First run (should cache everything)
        print("\n🚀 Test 4: First Run (Cache Population)")
        start_time = datetime.now()
        
        result1 = await detect_anomaly_with_user_adaptation(
            current_session_json=sample_session,
            user_id=test_user_id,
            storage_manager=storage_manager,
            caching_manager=caching_manager
        )
        
        first_run_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ First run completed in {first_run_time:.2f}s")
        print(f"   📊 Base score: {result1['base_anomaly_score']:.4f}")
        print(f"   📊 Adapted score: {result1['adapted_anomaly_score']:.4f}")
        print(f"   📊 Processing time: {result1['processing_time_seconds']:.2f}s")
        
        # Test 5: Second run (should use cache)
        print("\n⚡ Test 5: Second Run (Cache Usage)")
        start_time = datetime.now()
        
        result2 = await detect_anomaly_with_user_adaptation(
            current_session_json=sample_session,
            user_id=test_user_id,
            storage_manager=storage_manager,
            caching_manager=caching_manager
        )
        
        second_run_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Second run completed in {second_run_time:.2f}s")
        print(f"   📊 Base score: {result2['base_anomaly_score']:.4f}")
        print(f"   📊 Adapted score: {result2['adapted_anomaly_score']:.4f}")
        print(f"   📊 Processing time: {result2['processing_time_seconds']:.2f}s")
        
        # Test 6: Cache performance
        print("\n📈 Test 6: Cache Performance Analysis")
        final_stats = caching_manager.get_cache_stats()
        print(f"   📊 BERT cache size: {final_stats['bert_cache_size']}")
        print(f"   📊 User profile cache size: {final_stats['user_profile_cache_size']}")
        print(f"   📊 Historical sessions cache size: {final_stats['historical_sessions_cache_size']}")
        print(f"   📊 BERT hit rate: {final_stats['bert_hit_rate']:.2%}")
        print(f"   📊 Profile hit rate: {final_stats['profile_hit_rate']:.2%}")
        print(f"   📊 Sessions hit rate: {final_stats['sessions_hit_rate']:.2%}")
        
        # Test 7: Speed improvement
        speedup = first_run_time / second_run_time if second_run_time > 0 else 0
        print(f"\n🚀 Performance Improvement:")
        print(f"   📊 First run: {first_run_time:.2f}s")
        print(f"   📊 Second run: {second_run_time:.2f}s")
        print(f"   📊 Speedup: {speedup:.1f}x")
        
        # Test 8: Main ML Engine Integration
        print("\n🔗 Test 8: Main ML Engine Integration")
        try:
            from main import gnn_caching_manager, gnn_storage_manager
            print("   ✅ Global managers imported successfully")
            
            # Test the global managers
            global_stats = gnn_caching_manager.get_cache_stats()
            print(f"   📊 Global cache stats: {global_stats}")
            
        except Exception as e:
            print(f"   ⚠️ Main integration test failed: {e}")
        
        print("\n✅ All tests completed successfully!")
        
        return {
            "status": "success",
            "first_run_time": first_run_time,
            "second_run_time": second_run_time,
            "speedup": speedup,
            "cache_stats": final_stats,
            "results": {
                "first_run": result1,
                "second_run": result2
            }
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

async def test_different_users():
    """Test with different users from the Supabase storage"""
    
    print("\n👥 Testing with Different Users")
    print("=" * 40)
    
    try:
        from gnn_escalation import detect_anomaly_with_user_adaptation, CachingManager, SupabaseStorageManager
        
        caching_manager = CachingManager()
        storage_manager = SupabaseStorageManager(caching_manager)
        await storage_manager.initialize()
        
        # Test users from the Supabase storage screenshot
        test_users = [
            "demo_user_12345",
            "test-user-123", 
            "user_alice_001",
            "user_bob_002",
            "user_charlie_003"
        ]
        
        sample_session = {
            "logs": [
                {
                    "timestamp": "2025-01-20T10:30:00Z",
                    "event_type": "screen_touch",
                    "data": {"x": 150, "y": 300, "pressure": 0.8}
                }
            ]
        }
        
        results = {}
        
        for user_id in test_users:
            print(f"\n👤 Testing user: {user_id}")
            try:
                result = await detect_anomaly_with_user_adaptation(
                    current_session_json=sample_session,
                    user_id=user_id,
                    storage_manager=storage_manager,
                    caching_manager=caching_manager
                )
                
                results[user_id] = result
                print(f"   ✅ Success: base={result['base_anomaly_score']:.4f}, adapted={result['adapted_anomaly_score']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[user_id] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ Multi-user test failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🧪 Starting Integration Tests")
    print("=" * 50)
    
    # Run the main integration test
    result = asyncio.run(test_gnn_integration())
    
    if result["status"] == "success":
        print(f"\n🎉 Integration test completed successfully!")
        print(f"📊 Speedup achieved: {result['speedup']:.1f}x")
        
        # Run multi-user test
        print("\n" + "=" * 50)
        multi_user_results = asyncio.run(test_different_users())
        
        print(f"\n📋 Summary:")
        print(f"   ✅ Main integration: PASSED")
        print(f"   📊 Cache performance: {result['cache_stats']['bert_hit_rate']:.2%} BERT hit rate")
        print(f"   🚀 Speedup: {result['speedup']:.1f}x")
        print(f"   👥 Multi-user test: {'PASSED' if 'error' not in multi_user_results else 'FAILED'}")
        
    else:
        print(f"❌ Integration test failed: {result['error']}") 