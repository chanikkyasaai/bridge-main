import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Any, Optional
import uuid
import logging
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import json
import os
import torch.nn as nn
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork
from database import DatabaseManager
from datetime import datetime
import hashlib
from functools import lru_cache
import time

# TFT Encoder block for temporal embedding
class TFTEncoder(nn.Module):
    def __init__(self, input_dim=768, d_model=128, n_layers=2, out_dim=768):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.tft_grn = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, d_model) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_dim)
    def forward(self, x):
        # x: (seq_len, input_dim)
        x = self.input_proj(x)
        for layer in self.tft_grn:
            x = layer(x)
        x = self.out_proj(x)
        return x

logger = logging.getLogger(__name__)

class CachingManager:
    """Manages caching for BERT embeddings, user profiles, and historical data"""
    
    def __init__(self, max_cache_size=1000):
        self.bert_cache = {}  # text_hash -> embedding
        self.user_profile_cache = {}  # user_id -> (profile, timestamp)
        self.historical_sessions_cache = {}  # user_id -> (sessions, timestamp)
        self.max_cache_size = max_cache_size
        self.cache_stats = {
            "bert_hits": 0,
            "bert_misses": 0,
            "profile_hits": 0,
            "profile_misses": 0,
            "sessions_hits": 0,
            "sessions_misses": 0
        }
        
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_bert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embedding from cache or None if not found"""
        text_hash = self._get_text_hash(text)
        if text_hash in self.bert_cache:
            self.cache_stats["bert_hits"] += 1
            return self.bert_cache[text_hash]
        self.cache_stats["bert_misses"] += 1
        return None
    
    def set_bert_embedding(self, text: str, embedding: np.ndarray):
        """Cache BERT embedding"""
        text_hash = self._get_text_hash(text)
        if len(self.bert_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.bert_cache))
            del self.bert_cache[oldest_key]
        self.bert_cache[text_hash] = embedding
    
    def get_user_profile(self, user_id: str, max_age_seconds: int = 3600) -> Optional[np.ndarray]:
        """Get user profile from cache if not expired"""
        if user_id in self.user_profile_cache:
            profile, timestamp = self.user_profile_cache[user_id]
            if time.time() - timestamp < max_age_seconds:
                self.cache_stats["profile_hits"] += 1
                return profile
            else:
                # Expired, remove from cache
                del self.user_profile_cache[user_id]
        self.cache_stats["profile_misses"] += 1
        return None
    
    def set_user_profile(self, user_id: str, profile: np.ndarray):
        """Cache user profile with current timestamp"""
        self.user_profile_cache[user_id] = (profile, time.time())
    
    def get_historical_sessions(self, user_id: str, max_age_seconds: int = 1800) -> Optional[List[Dict]]:
        """Get historical sessions from cache if not expired"""
        if user_id in self.historical_sessions_cache:
            sessions, timestamp = self.historical_sessions_cache[user_id]
            if time.time() - timestamp < max_age_seconds:
                self.cache_stats["sessions_hits"] += 1
                return sessions
            else:
                # Expired, remove from cache
                del self.historical_sessions_cache[user_id]
        self.cache_stats["sessions_misses"] += 1
        return None
    
    def set_historical_sessions(self, user_id: str, sessions: List[Dict]):
        """Cache historical sessions with current timestamp"""
        self.historical_sessions_cache[user_id] = (sessions, time.time())
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_bert = self.cache_stats["bert_hits"] + self.cache_stats["bert_misses"]
        total_profile = self.cache_stats["profile_hits"] + self.cache_stats["profile_misses"]
        total_sessions = self.cache_stats["sessions_hits"] + self.cache_stats["sessions_misses"]
        
        return {
            "bert_cache_size": len(self.bert_cache),
            "user_profile_cache_size": len(self.user_profile_cache),
            "historical_sessions_cache_size": len(self.historical_sessions_cache),
            "bert_hit_rate": self.cache_stats["bert_hits"] / total_bert if total_bert > 0 else 0,
            "profile_hit_rate": self.cache_stats["profile_hits"] / total_profile if total_profile > 0 else 0,
            "sessions_hit_rate": self.cache_stats["sessions_hits"] / total_sessions if total_sessions > 0 else 0,
            "total_bert_requests": total_bert,
            "total_profile_requests": total_profile,
            "total_sessions_requests": total_sessions
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.bert_cache.clear()
        self.user_profile_cache.clear()
        self.historical_sessions_cache.clear()
        logger.info("All caches cleared")

class SupabaseStorageManager:
    """Manages reading historical session data from Supabase storage"""
    
    def __init__(self, caching_manager: CachingManager = None):
        self.db_manager = DatabaseManager()
        self.caching_manager = caching_manager or CachingManager()
        
    async def initialize(self):
        """Initialize database connection"""
        await self.db_manager.initialize()
        
    async def get_user_historical_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all historical session logs for a user from Supabase storage with caching"""
        try:
            # Check cache first
            cached_sessions = self.caching_manager.get_historical_sessions(user_id)
            if cached_sessions is not None:
                logger.info(f"📋 Using cached historical sessions for user {user_id} ({len(cached_sessions)} sessions)")
                return cached_sessions
            
            logger.info(f"🔄 Fetching historical sessions from Supabase for user {user_id}")
            start_time = time.time()
            
            # List all files in the user's folder: behavior-logs/logs/{user_id}/
            bucket_name = "behavior-logs"
            folder_path = f"logs/{user_id}/"
            
            # Get list of all files in the user's folder
            files = self.db_manager.supabase.storage.from_(bucket_name).list(folder_path)
            
            historical_sessions = []
            
            for file_info in files:
                if file_info.get('name', '').endswith('.json'):
                    file_path = f"{folder_path}{file_info['name']}"
                    
                    try:
                        # Download and read the file content
                        file_content = self.db_manager.supabase.storage.from_(bucket_name).download(file_path)
                        
                        # Parse JSON content
                        session_data = json.loads(file_content.decode('utf-8'))
                        historical_sessions.append(session_data)
                        
                        logger.info(f"📄 Loaded historical session: {file_info['name']}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load session file {file_info['name']}: {e}")
                        continue
            
            fetch_time = time.time() - start_time
            logger.info(f"✅ Loaded {len(historical_sessions)} historical sessions for user {user_id} in {fetch_time:.2f}s")
            
            # Cache the results
            self.caching_manager.set_historical_sessions(user_id, historical_sessions)
            
            return historical_sessions
            
        except Exception as e:
            logger.error(f"Failed to get historical sessions for user {user_id}: {e}")
            return []

class SessionEventGraph:
    """Maintains a dynamic event graph for a session."""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.last_event_id = None

    def add_event(self, event: Dict[str, Any]):
        event_id = str(uuid.uuid4())
        node_data = {
            'event_id': event_id,
            'type': event.get('event_type'),
            'timestamp': event.get('timestamp'),
            'data': event.get('data', {})
        }
        self.graph.add_node(event_id, **node_data)
        if self.last_event_id:
            prev_time = self.graph.nodes[self.last_event_id]['timestamp']
            curr_time = event.get('timestamp')
            try:
                delay_ms = int((np.datetime64(curr_time) - np.datetime64(prev_time)) / np.timedelta64(1, 'ms'))
            except Exception:
                delay_ms = 0
            transition = f"{self.graph.nodes[self.last_event_id]['type']}  {event.get('event_type')}"
            self.graph.add_edge(self.last_event_id, event_id, delay_ms=delay_ms, transition=transition)
        self.last_event_id = event_id

    def to_pyg_data(self):
        # Convert the graph to torch_geometric.data.Data
        node_features = []
        node_types = []
        timestamps = []
        node_id_map = {nid: i for i, nid in enumerate(self.graph.nodes)}
        for nid, node in self.graph.nodes(data=True):
            # Example: encode type as int, timestamp as float, and flatten data if possible
            node_types.append(hash(node.get('type')) % 1000)
            timestamps.append(float(np.datetime64(node.get('timestamp')).astype('datetime64[ms]').astype(float)))
            # Flatten data dict to a vector (simple version: just use pressure/x/y if present)
            d = node.get('data', {})
            node_features.append([
                d.get('pressure', 0.0),
                d.get('x', 0.0),
                d.get('y', 0.0)
            ])
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = [[], []]
        edge_attr = []
        for src, dst, edata in self.graph.edges(data=True):
            edge_index[0].append(node_id_map[src])
            edge_index[1].append(node_id_map[dst])
            edge_attr.append([edata.get('delay_ms', 0)])
        if len(edge_index[0]) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Update GNNAnomalyDetector to use RealInformer with caching
class GNNAnomalyDetector:
    """Loads a public GNN model (GCN baseline with Informer+Adapter) and predicts anomaly score for a session graph."""
    def __init__(self, model_path: str = None, caching_manager: CachingManager = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tft_encoder = TFTEncoder(input_dim=768, d_model=128, n_layers=2, out_dim=768).to(self.device)
        self.model = SimpleGCN(in_channels=768, hidden_channels=64, out_channels=2).to(self.device)
        self.caching_manager = caching_manager or CachingManager()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tft_encoder.eval()
        
        logger.info(f"🤖 GNNAnomalyDetector initialized on device: {self.device}")
        logger.info(f"📊 Caching manager: {'Enabled' if caching_manager else 'Disabled'}")
    def get_bert_embedding(self, text: str):
        # Check cache first
        cached_embedding = self.caching_manager.get_bert_embedding(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding if not in cache
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=16)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        
        # Cache the embedding
        self.caching_manager.set_bert_embedding(text, embedding)
        return embedding
    def informer_embed(self, node_features):
        # node_features: (num_nodes, 768)
        x = torch.tensor(np.stack(node_features), dtype=torch.float).to(self.device)  # (seq_len, 768)
        with torch.no_grad():
            out = self.tft_encoder(x)  # (seq_len, 768)
        return out.cpu()  # (num_nodes, 768)
    def predict_anomaly(self, pyg_data: Data) -> float:
        pyg_data = pyg_data.to(self.device)
        with torch.no_grad():
            logits = self.model(pyg_data)
            anomaly_score = float(torch.sigmoid(logits[:, 1]).mean().cpu().item())
        return anomaly_score

# --- Test function for logs.json ---
def test_logs_json(logs_path='data/logs.json'):
    # Load logs.json
    with open(logs_path, 'r') as f:
        logs = json.load(f)
    events = logs['logs']
    # Build event graph
    graph = SessionEventGraph()
    for event in events:
        graph.add_event(event)
    # Replace node features with BERT embeddings of event_type
    detector = GNNAnomalyDetector()
    node_features = []
    for nid, node in graph.graph.nodes(data=True):
        emb = detector.get_bert_embedding(str(node.get('type', 'event')))
        node_features.append(emb)
    x = torch.tensor(node_features, dtype=torch.float)
    pyg_data = graph.to_pyg_data()
    pyg_data.x = x
    # Run anomaly detection
    score = detector.predict_anomaly(pyg_data)
    print(f"Anomaly score for logs.json: {score:.4f}")

# Update adaptation pipeline to use Informer

def adapt_and_test_user_sessions(adapter_paths, test_path):
    detector = GNNAnomalyDetector()
    # Process adapters through Informer
    adapter_node_features = []
    for path in adapter_paths:
        with open(path, 'r') as f:
            logs = json.load(f)
        events = logs['logs']
        graph = SessionEventGraph()
        for event in events:
            graph.add_event(event)
        for nid, node in graph.graph.nodes(data=True):
            emb = detector.get_bert_embedding(str(node.get('type', 'event')))
            adapter_node_features.append(emb)
    if adapter_node_features:
        adapter_informer_embs = detector.informer_embed(adapter_node_features)
        user_profile = adapter_informer_embs.mean(dim=0).numpy()
    else:
        user_profile = None
    # Process test session through Informer
    with open(test_path, 'r') as f:
        logs = json.load(f)
    events = logs['logs']
    test_graph = SessionEventGraph()
    for event in events:
        test_graph.add_event(event)
    test_node_features = []
    for nid, node in test_graph.graph.nodes(data=True):
        emb = detector.get_bert_embedding(str(node.get('type', 'event')))
        test_node_features.append(emb)
    test_informer_embs = detector.informer_embed(test_node_features)
    x = test_informer_embs
    pyg_data = test_graph.to_pyg_data()
    pyg_data.x = x
    base_score = detector.predict_anomaly(pyg_data)
    if user_profile is not None:
        adapted_x = x - torch.tensor(user_profile, dtype=torch.float)
        pyg_data.x = adapted_x
        adapted_score = detector.predict_anomaly(pyg_data)
    else:
        adapted_score = None
    print(f"Non-adapted anomaly score for {test_path}: {base_score:.4f}")
    if adapted_score is not None:
        print(f"Adapted anomaly score for {test_path} (using user history): {adapted_score:.4f}")
    else:
        print("No adapter embeddings available.")

# New function for real-time anomaly detection with Supabase integration and caching
async def detect_anomaly_with_user_adaptation(
    current_session_json: Dict[str, Any], 
    user_id: str,
    storage_manager: SupabaseStorageManager = None,
    caching_manager: CachingManager = None
) -> Dict[str, Any]:
    """
    Perform user-adapted anomaly detection using current session data and historical data from Supabase
    
    Args:
        current_session_json: Current session data as JSON object (from WebSocket/backend)
        user_id: User ID to fetch historical sessions for
        storage_manager: Optional SupabaseStorageManager instance (will create if None)
    
    Returns:
        Dict containing anomaly scores and metadata
    """
    try:
        start_time = time.time()
        logger.info(f"🚀 Starting anomaly detection for user {user_id}")
        
        # Initialize caching manager if not provided
        if caching_manager is None:
            caching_manager = CachingManager()
            logger.info("📊 Initialized new caching manager")
        
        # Initialize storage manager if not provided
        if storage_manager is None:
            storage_manager = SupabaseStorageManager(caching_manager)
            await storage_manager.initialize()
        
        # Check for cached user profile
        cached_profile = caching_manager.get_user_profile(user_id)
        if cached_profile is not None:
            logger.info(f"📋 Using cached user profile for {user_id}")
            user_profile = cached_profile
            historical_sessions = []  # Not needed if profile is cached
        else:
            logger.info(f"🔄 Fetching historical sessions for user {user_id}")
            # Get historical sessions from Supabase
            historical_sessions = await storage_manager.get_user_historical_sessions(user_id)
            user_profile = None
        
        # Initialize detector with caching
        detector = GNNAnomalyDetector(caching_manager=caching_manager)
        
        # Process historical sessions as adapters (only if not using cached profile)
        if user_profile is None:
            logger.info(f"🔧 Processing {len(historical_sessions)} historical sessions for user profile")
            adapter_node_features = []
            for i, session_data in enumerate(historical_sessions):
                events = session_data.get('logs', [])
                if not events:
                    continue
                    
                logger.info(f"  📄 Processing historical session {i+1}/{len(historical_sessions)} ({len(events)} events)")
                
                # Build graph for this historical session
                graph = SessionEventGraph()
                for event in events:
                    graph.add_event(event)
                
                # Extract BERT embeddings for all nodes
                for nid, node in graph.graph.nodes(data=True):
                    emb = detector.get_bert_embedding(str(node.get('type', 'event')))
                    adapter_node_features.append(emb)
            
            # Create user profile from historical data
            if adapter_node_features:
                logger.info(f"🧠 Creating user profile from {len(adapter_node_features)} embeddings")
                adapter_informer_embs = detector.informer_embed(adapter_node_features)
                user_profile = adapter_informer_embs.mean(dim=0).numpy()
                
                # Cache the user profile
                caching_manager.set_user_profile(user_id, user_profile)
                logger.info(f"✅ Created and cached user profile from {len(historical_sessions)} historical sessions")
            else:
                user_profile = None
                logger.warning(f"⚠️ No historical sessions found for user {user_id}")
        else:
            logger.info(f"📋 Using cached user profile (skipping historical session processing)")
        
        # Process current session
        current_events = current_session_json.get('logs', [])
        if not current_events:
            raise ValueError("No events found in current session data")
        
        logger.info(f"📊 Processing current session with {len(current_events)} events")
        
        # Build graph for current session
        current_graph = SessionEventGraph()
        for event in current_events:
            current_graph.add_event(event)
        
        logger.info(f"🕸️ Built event graph with {len(current_graph.graph.nodes)} nodes and {len(current_graph.graph.edges)} edges")
        
        # Extract BERT embeddings for current session
        logger.info(f"🔤 Extracting BERT embeddings for {len(current_graph.graph.nodes)} nodes")
        current_node_features = []
        for nid, node in current_graph.graph.nodes(data=True):
            emb = detector.get_bert_embedding(str(node.get('type', 'event')))
            current_node_features.append(emb)
        
        # Apply temporal encoding to current session
        logger.info(f"⏰ Applying temporal encoding (TFT) to current session")
        current_informer_embs = detector.informer_embed(current_node_features)
        
        # Create PyG data for current session
        pyg_data = current_graph.to_pyg_data()
        pyg_data.x = current_informer_embs
        
        # Get base anomaly score
        logger.info(f"🔍 Computing base anomaly score")
        base_score = detector.predict_anomaly(pyg_data)
        logger.info(f"📊 Base anomaly score: {base_score:.4f}")
        
        # Apply user adaptation if profile exists
        adapted_score = None
        if user_profile is not None:
            logger.info(f"👤 Applying user adaptation using cached profile")
            adapted_x = current_informer_embs - torch.tensor(user_profile, dtype=torch.float)
            pyg_data.x = adapted_x
            adapted_score = detector.predict_anomaly(pyg_data)
            logger.info(f"📊 Adapted anomaly score: {adapted_score:.4f}")
        else:
            logger.info(f"⚠️ No user profile available for adaptation")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = caching_manager.get_cache_stats()
        
        # Prepare result
        result = {
            "user_id": user_id,
            "current_session_events": len(current_events),
            "historical_sessions_used": len(historical_sessions),
            "base_anomaly_score": base_score,
            "adapted_anomaly_score": adapted_score,
            "user_profile_available": user_profile is not None,
            "processing_time_seconds": total_time,
            "cache_stats": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        adapted_score_str = f"{adapted_score:.4f}" if adapted_score is not None else "N/A"
        logger.info(f"✅ Anomaly detection completed for user {user_id} in {total_time:.2f}s")
        logger.info(f"📊 Results: base={base_score:.4f}, adapted={adapted_score_str}")
        logger.info(f"📈 Cache performance: BERT hit rate={cache_stats['bert_hit_rate']:.2%}, Profile hit rate={cache_stats['profile_hit_rate']:.2%}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to perform anomaly detection for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Helper function for testing with sample data
async def test_supabase_integration():
    """Test the Supabase integration with sample data"""
    print("Testing Supabase integration...")
    
    # Sample current session data (simulating WebSocket data)
    sample_current_session = {
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
            }
        ]
    }
    
    # Test with a user ID (you can change this to test with different users)
    test_user_id = "demo_user_12345"  # From the Supabase storage screenshot
    
    try:
        result = await detect_anomaly_with_user_adaptation(sample_current_session, test_user_id)
        print(f"Test result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == '__main__':
    # Test Supabase integration
    import asyncio
    print("Testing Supabase integration for GNN escalation...")
    asyncio.run(test_supabase_integration()) 