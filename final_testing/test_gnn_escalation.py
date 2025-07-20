import json
import os
import numpy as np
import torch
from typing import List, Dict
from pathlib import Path
import networkx as nx
import uuid
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Minimal TFTEncoder
class TFTEncoder(nn.Module):
    def __init__(self, input_dim=768, d_model=128, n_layers=2, out_dim=768):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.tft_grn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU()) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, out_dim)
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.tft_grn:
            x = layer(x)
        x = self.out_proj(x)
        return x

# Minimal CachingManager
class CachingManager:
    def __init__(self, max_cache_size=1000):
        self.bert_cache = {}
        self.max_cache_size = max_cache_size
    def _get_text_hash(self, text: str) -> str:
        return str(hash(text))
    def get_bert_embedding(self, text: str):
        return self.bert_cache.get(self._get_text_hash(text))
    def set_bert_embedding(self, text: str, embedding: np.ndarray):
        if len(self.bert_cache) >= self.max_cache_size:
            self.bert_cache.pop(next(iter(self.bert_cache)))
        self.bert_cache[self._get_text_hash(text)] = embedding

class SessionEventGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.last_event_id = None
    def add_event(self, event: Dict):
        event_id = str(uuid.uuid4())
        node_data = {
            'event_id': event_id,
            'type': event.get('event_type'),
            'timestamp': event.get('timestamp'),
            'data': event.get('data', {})
        }
        self.graph.add_node(event_id, **node_data)
        if self.last_event_id:
            self.graph.add_edge(self.last_event_id, event_id)
        self.last_event_id = event_id
    def to_pyg_data(self):
        node_features = []
        node_id_map = {nid: i for i, nid in enumerate(self.graph.nodes)}
        for nid, node in self.graph.nodes(data=True):
            d = node.get('data', {})
            node_features.append([
                d.get('pressure', 0.0),
                d.get('x', 0.0),
                d.get('y', 0.0)
            ])
        x = torch.tensor(node_features, dtype=torch.float) if node_features else torch.zeros((1,3))
        edge_index = [[], []]
        for src, dst in self.graph.edges():
            edge_index[0].append(node_id_map[src])
            edge_index[1].append(node_id_map[dst])
        if len(edge_index[0]) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
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

class GNNAnomalyDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tft_encoder = TFTEncoder(input_dim=768, d_model=128, n_layers=2, out_dim=768).to(self.device)
        self.model = SimpleGCN(in_channels=768, hidden_channels=64, out_channels=2).to(self.device)
        self.caching_manager = CachingManager()
        self.model.eval()
        self.tft_encoder.eval()
    def get_bert_embedding(self, text: str):
        cached = self.caching_manager.get_bert_embedding(text)
        if cached is not None:
            return cached
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=16)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        self.caching_manager.set_bert_embedding(text, embedding)
        return embedding
    def informer_embed(self, node_features):
        x = torch.tensor(np.stack(node_features), dtype=torch.float).to(self.device)
        with torch.no_grad():
            out = self.tft_encoder(x)
        return out.cpu()
    def predict_anomaly(self, pyg_data: Data) -> float:
        pyg_data = pyg_data.to(self.device)
        with torch.no_grad():
            logits = self.model(pyg_data)
            anomaly_score = float(torch.sigmoid(logits[:, 1]).mean().cpu().item())
        return anomaly_score

def load_session(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_gnn_escalation(adapters: List[Dict], test_session: Dict, label: str):
    detector = GNNAnomalyDetector()
    # Process adapters through Informer
    adapter_node_features = []
    for logs in adapters:
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
    events = test_session['logs']
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
    print(f"\n--- {label} ---")
    print(f"Non-adapted anomaly score: {base_score:.4f}")
    if adapted_score is not None:
        print(f"Adapted anomaly score (using user history): {adapted_score:.4f}")
    else:
        print("No adapter embeddings available.")

def main():
    print("=== GNN Escalation Test: Normal vs Traitor ===")
    base_dir = Path("final_testing")
    adapter_paths = [base_dir / f"test_user_session_{i:02d}.json" for i in range(1, 11)]
    adapters = [load_session(str(p)) for p in adapter_paths]
    normal_test = load_session(str(base_dir / "test_user_session_11.json"))
    traitor_test = load_session(str(base_dir / "test_traitor_session.json"))
    run_gnn_escalation(adapters, normal_test, "Normal User (Adapters: 1-10, Test: 11)")
    run_gnn_escalation(adapters, traitor_test, "Traitor User (Adapters: 1-10, Test: traitor)")

if __name__ == "__main__":
    main() 