import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Any
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

# Update GNNAnomalyDetector to use RealInformer
class GNNAnomalyDetector:
    """Loads a public GNN model (GCN baseline with Informer+Adapter) and predicts anomaly score for a session graph."""
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tft_encoder = TFTEncoder(input_dim=768, d_model=128, n_layers=2, out_dim=768).to(self.device)
        self.model = SimpleGCN(in_channels=768, hidden_channels=64, out_channels=2).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.tft_encoder.eval()
    def get_bert_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=16)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
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

if __name__ == '__main__':
    adapter_paths = [f'data/test_user_session_0{i}.json' for i in range(1, 6)]
    test_path = 'data/test_user_session_06.json'
    adapt_and_test_user_sessions(adapter_paths, test_path) 