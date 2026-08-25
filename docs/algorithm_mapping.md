# Algorithm Mapping: Paper to Code

_Last Updated: January 2026_

This document provides a detailed mapping between the algorithms described in the FedGATSage paper and their implementation in the codebase.

## Overview

The implementation in this repository focuses on graph-level anomaly detection through combined GAT and GraphSAGE feature learning.

The architecture now emphasizes direct graph-based anomaly classification rather than legacy centrality-driven mechanisms.

## Algorithm 1: Graph Construction and Embedding Generation

### Paper Description vs Implementation

| Paper Step | Paper Description                | Code Implementation                         | File Location                          |
| ---------- | -------------------------------- | ------------------------------------------- | -------------------------------------- |
| Step 1     | `G = (V, E) ← ConstructGraph(D)` | Graph construction in `_process_to_graph()` | `src/federated_learning.py:DataLoader` |
| Step 2     | `H ← GAT(X, G)`                  | Unified `GATLayer` for graph data           | `src/gnn_models.py`                    |
| Step 3     | Graph-level classification       | `graph_label` prediction in client model    | `src/federated_learning.py`            |
| Step 4     | Global aggregation               | `_aggregate_updates()`                      | `src/federated_learning.py`            |

### Key Implementation Insight

This implementation uses graph-level anomaly classification with unified GAT-style embeddings and server-side aggregation.

- Graph-level labels are predicted from the client graph representation
- Feature abstraction is based on graph connectivity and traffic attributes
- The model no longer relies on legacy centrality-driven flows

## Algorithm 2: Global Graph Aggregation

### Paper Description vs Implementation

| Paper Step  | Paper Description                                               | Code Implementation         | File Location               |
| ----------- | --------------------------------------------------------------- | --------------------------- | --------------------------- |
| Graph Init  | `Initialize server-side graph representation`                   | `GlobalGAT`                 | `src/gnn_models.py`         |
| Aggregation | Combine client model updates into a global graph representation | `_aggregate_updates()`      | `src/federated_learning.py` |
| Prediction  | Apply global classifier to aggregate features                   | `GlobalGAT.forward()`       | `src/gnn_models.py`         |

## Specialized GAT Variants

### Paper Mention vs Implementation

The current codebase uses a single unified graph detection module for anomaly-aware graph feature extraction.

| Detector Type | Target Attacks      | Code Implementation               |
| ------------- | ------------------- | --------------------------------- |
| Unified GAT   | All anomaly classes | `GATLayer` in `src/gnn_models.py` |

### Key Features:

- **Unified graph detection**: Same module handles anomaly classification without explicit separate detector variants
- **Graph-level anomaly labels**: Prediction uses `graph_label` rather than edge-level classification

## Feature Engineering Correspondence

### Paper Features vs Code Implementation

| Paper Feature Type         | Code Implementation          | File Location                |
| -------------------------- | ---------------------------- | ---------------------------- |
| Numerical traffic features | `extract_features()`         | `src/feature_engineering.py` |
| Temporal features          | `_add_temporal_features()`   | `src/feature_engineering.py` |
| Content features           | `_add_content_features()`    | `src/feature_engineering.py` |
| Behavioral features        | `_add_behavioral_features()` | `src/feature_engineering.py` |

## Federated Learning Process

### Paper Workflow vs Implementation

| Paper Step                | Code Implementation         | File Location               |
| ------------------------- | --------------------------- | --------------------------- |
| Local GAT training        | `_train_client_model()`     | `src/federated_learning.py` |
| Flow embedding generation | `generate_embeddings()`     | `src/federated_learning.py` |
| Server aggregation        | `_aggregate_updates()`      | `src/federated_learning.py` |
| Global GAT processing     | `GlobalGAT.forward()`       | `src/gnn_models.py`         |
| Parameter redistribution  | `_redistribute_models()`    | `src/federated_learning.py` |

## Privacy Mechanisms

### Paper Claims vs Implementation

| Privacy Mechanism               | Paper Description                   | Code Implementation                      |
| ------------------------------- | ----------------------------------- | ---------------------------------------- |
| Graph feature abstraction       | Share compact graph representations | `graph_label`-based client graphs        |
| Individual device protection    | No raw device data shared           | IP addresses abstracted into graph nodes |
| Structural pattern preservation | Maintain network relationships      | Graph node and edge features             |
| Communication efficiency        | Reduced data transfer               | Federated averaging of model parameters  |

## Validation Points

To validate the current anomaly-focused implementation:

1.  **Graph-level prediction**: Verify `graph_label` is used for anomaly classification
2.  **Contrastive learning**: Confirm `nt_xent_loss()` is combined with cross-entropy
3.  **Model aggregation**: Verify `_aggregate_updates()` averages client model states
4.  **Feature extraction**: Check `extract_features()` for numeric traffic features
5.  **Performance**: Compare results with expected anomaly detection metrics

## Running the Complete Pipeline

```bash
# 1. Prepare Data (New)
python preprocess_data.py --input_file data/raw_dataset.csv --output_dir data --num_clients 5

# 2. Run demo mode
python experiments/fedgatsage_experiment.py --data_dir data --demo_mode

# 3. Full experiment matching paper
python experiments/fedgatsage_experiment.py \
  --data_dir data \
  --dataset cic_ton_iot \
  --num_clients 5 \
  --num_rounds 15 \
  --detector_types temporal content behavioral


```
