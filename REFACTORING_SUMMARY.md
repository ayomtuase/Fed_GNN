# GAT Graph Construction Refactoring Summary

## Overview

Refactored the Fed_GNN project to build graphs dynamically from top-k node embeddings during the forward pass, instead of constructing static graphs from dataframes before training.

## Changes Made

### 1. **GATLayer** (src/gnn_models.py)

**Old Approach:**

- Took pre-built `edge_index` as input
- Simple linear embedding of features followed by GAT

**New Approach:**

- Added learnable node embeddings: `nn.Embedding(node_num, hidden_dim)`
- New `_build_dynamic_graph()` method that:
  - Computes cosine similarity between learned node embeddings
  - Selects top-k most similar neighbors for each node
  - Builds edge_index dynamically at each forward pass
- Updated `forward()` signature: now takes only features `x`, no longer takes `edge_index`
- Stores learned graph in `self.learned_graph` for inspection

**New Constructor Parameters:**

- `node_num: int = 100` - Number of nodes in the graph
- `topk: int = 20` - Number of top neighbors to connect for each node

### 2. **Data Loading** (src/federated_learning.py)

**Removed Function:**

- `_build_graph_from_dataframe()` - This function built static edges from CSV data

**New Function:**

- `_build_features_from_dataframe()` - Simpler version that extracts:
  - Node features per IP address
  - Graph-level labels
  - `num_nodes` count
- No longer returns `edge_index` since graphs are built dynamically

### 3. **Training Process** (src/federated_learning.py)

**Updated `_train_client_model()` method:**

- Now calls: `z1, logits1 = model(features)` instead of `model(features, edge_index)`
- Removed edge_index extraction and device transfer
- Graph construction happens inside the model's forward pass

### 4. **Model Initialization** (src/federated_learning.py)

**Updated `initialize_models()` method:**

- Added parameters: `node_num: int = 100` and `topk: int = 20`
- Passes these to each GATLayer instance
- Updated checkpoint saving/loading to preserve these parameters

## Benefits of This Approach

1. **Learned Graphs**: The graph structure is learned during training, making it task-adaptive
2. **Efficiency**: Top-k selection is more scalable than creating full edge matrices
3. **Flexibility**: Different numbers of nodes per client are handled automatically
4. **Interpretability**: The learned graph can be inspected via `model.learned_graph`

## Important Notes

### Variable Node Counts

- The `GATLayer` is initialized with a fixed `node_num`
- If client data has different numbers of nodes, you should:
  - Estimate `node_num` from your data distribution
  - Use a `node_num` that accommodates the largest expected graph
  - Pad features with zeros if needed

### Example Usage

```python
# Initialize with node_num matching expected graph size
fed_system.initialize_models(
    input_dim=64,
    hidden_dim=256,
    num_classes=2,
    node_num=150,  # Adjust based on your data
    topk=20
)
```

### Accessing Learned Graphs

After training, access the learned graph for a model:

```python
model = fed_system.client_models[client_id]
print(model.learned_graph)  # Shape: (node_num, topk)
```

## Backward Compatibility

The old API for `load_graph_from_csv()` is maintained but now returns simpler data without edge indices, making the training pipeline cleaner and more memory-efficient.
