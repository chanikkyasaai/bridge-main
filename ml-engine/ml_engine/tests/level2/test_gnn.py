import torch
import pytest
from torch_geometric.data import Data
from ml_engine.adapters.level2.layer2_verifier import SessionGraphGNN

def make_synthetic_graph(batch_size=2, num_nodes=5, node_features=32):
    # Create two simple graphs in a batch
    x = torch.randn(batch_size * num_nodes, node_features)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0, 2, 3, 4, 0, 1]
    ], dtype=torch.long)
    edge_index = edge_index.repeat(1, batch_size) + torch.arange(0, batch_size * num_nodes, num_nodes).repeat_interleave(10)
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    return data

@pytest.mark.parametrize("batch_size,num_nodes,node_features", [(2, 5, 32)])
def test_session_graph_gnn(batch_size, num_nodes, node_features):
    gnn = SessionGraphGNN(node_features=node_features, edge_features=16)
    data = make_synthetic_graph(batch_size, num_nodes, node_features)
    anomaly_scores, embedding = gnn(data)
    assert anomaly_scores.shape == (batch_size, 1)
    assert embedding.shape[0] == batch_size
    assert torch.is_tensor(anomaly_scores)
    assert torch.is_tensor(embedding) 