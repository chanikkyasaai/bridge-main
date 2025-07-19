"""
Simple test to verify all ML engine components can be imported and initialized
"""

import sys
import os

# Add the ml-engine directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml-engine'))

try:
    print("Testing imports...")
    
    # Test feature extractor
    from feature_extractor import FeatureExtractor
    fe = FeatureExtractor(vector_dimensions=48)
    print("✓ FeatureExtractor imported and initialized")
    
    # Test bot detector  
    from bot_detector import BotDetector
    bd = BotDetector()
    print("✓ BotDetector imported and initialized")
    
    # Test database manager
    from database import DatabaseManager
    print("✓ DatabaseManager imported")
    
    # Test FAISS vector store
    from faiss.vector_store import FAISSVectorStore, VectorStorageManager
    fs = FAISSVectorStore(vector_dim=48, storage_path="./test_faiss")
    print("✓ FAISS components imported and initialized")
    
    # Test learning manager
    from learning_manager import LearningManager
    print("✓ LearningManager imported")
    
    # Test auth manager
    from auth_manager import AuthenticationManager, ContinuousAuthenticator
    print("✓ AuthenticationManager imported")
    
    print("\n🎉 All imports successful!")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    
    import numpy as np
    import asyncio
    
    async def test_basic_functionality():
        # Test feature extraction
        sample_events = [
            {
                "timestamp": "2025-07-19T10:00:00.000000",
                "event_type": "touch_down",
                "data": {
                    "coordinates": [300, 500],
                    "pressure": 0.5,
                    "inter_touch_gap_ms": 200
                }
            }
        ]
        
        vector = await fe.extract_features(sample_events)
        print(f"✓ Feature extraction works! Vector shape: {vector.shape}")
        
        # Test bot detection
        bot_result = await bd.analyze_events(sample_events)
        print(f"✓ Bot detection works! Result: {bot_result['is_bot']}")
        
        # Test FAISS cluster creation
        cluster_centroids = np.random.rand(2, 48).astype(np.float32)
        result = await fs.create_user_clusters("test_user", cluster_centroids, [0, 1])
        print(f"✓ FAISS cluster creation works! Status: {result['status']}")
        
        # Test similarity search
        query_vector = np.random.rand(48).astype(np.float32)
        similarity_result = await fs.find_nearest_cluster("test_user", query_vector)
        print(f"✓ FAISS similarity search works! Decision: {similarity_result['decision']}")
        
        print("\n🚀 All basic functionality tests passed!")
    
    # Run async tests
    asyncio.run(test_basic_functionality())
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install faiss-cpu scikit-learn numpy")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
