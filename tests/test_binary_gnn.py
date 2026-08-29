import os
import unittest
import torch
import torch.nn as nn
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from federated_learning import binary_focal_loss, VFLGradientNormalizer, FedGATSageSystem, supervised_contrastive_loss
from gnn_models import GlobalGraphSAGE

class TestBinaryGNN(unittest.TestCase):
    def test_binary_focal_loss(self):
        logits = torch.tensor([[1.5], [-1.0], [0.0]], dtype=torch.float32)
        targets = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
        
        # Run binary focal loss
        loss = binary_focal_loss(logits, targets, alpha=0.5, gamma=2.0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) # Scalar
        self.assertGreater(loss.item(), 0.0)

    def test_vfl_gradient_normalizer(self):
        t1 = torch.tensor([1.0, 2.0], requires_grad=True)
        t2 = torch.tensor([3.0, 4.0], requires_grad=True)
        
        # Apply VFLGradientNormalizer
        norm_t1, norm_t2 = VFLGradientNormalizer.apply(1.0, t1, t2)
        
        # Compute dummy loss to backpropagate
        loss = (norm_t1 ** 2).sum() + (norm_t2 ** 2).sum()
        loss.backward()
        
        # Assert gradients are scaled such that the global norm of all boundary gradients is exactly target_norm (1.0)
        g1 = t1.grad
        g2 = t2.grad
        
        global_norm = torch.sqrt(g1.norm(2)**2 + g2.norm(2)**2)
        self.assertAlmostEqual(global_norm.item(), 1.0, places=5)

    def test_system_initialization(self):
        system = FedGATSageSystem(
            data_dir="data",
            num_clients=2,
            device="cpu"
        )
        system.initialize_models(
            input_dim=2,
            hidden_dim=8,
            num_classes=2,
            client_node_nums=[5, 5]
        )
        
        # Assert the global model is indeed GlobalGraphSAGE
        self.assertIsInstance(system.global_model, GlobalGraphSAGE)
        
        # Assert server classifier final layer output size is indeed 1 (BCE)
        last_layer = list(system.global_model.classifier.modules())[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, 1)

    def test_calculate_metrics_with_auc(self):
        import numpy as np
        from utils import calculate_metrics, plot_roc_curve
        
        # Test binary classification
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob=y_prob)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        
        # Test imperfect prediction
        y_prob_imperfect = np.array([0.1, 0.8, 0.2, 0.9])
        metrics_imperfect = calculate_metrics(y_true, y_pred, y_prob=y_prob_imperfect)
        self.assertLess(metrics_imperfect["roc_auc"], 1.0)
        self.assertGreater(metrics_imperfect["roc_auc"], 0.0)

        # Test single class edge case
        y_true_single = np.array([1, 1, 1, 1])
        metrics_single = calculate_metrics(y_true_single, y_pred, y_prob=y_prob)
        self.assertIsNone(metrics_single["roc_auc"])
        
        # Test plot_roc_curve returns a Figure or None appropriately
        fig = plot_roc_curve(y_true, y_prob)
        self.assertIsNotNone(fig)
        
        # Test single class plotting returns None
        fig_single = plot_roc_curve(y_true_single, y_prob)
        self.assertIsNone(fig_single)

    def test_supervised_contrastive_loss_normal_alignment(self):
        # Create dummy representations for 4 samples: 2 normal (0), 2 anomaly (1)
        # z1 and z2 are the two augmented views
        z1_base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z2_base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1, 0, 1])
        
        # Compute baseline loss where views are aligned and identical within classes
        loss_base = supervised_contrastive_loss(z1_base, z2_base, labels, temperature=0.1)
        self.assertTrue(torch.is_tensor(loss_base))
        self.assertEqual(loss_base.dim(), 0) # Scalar
        self.assertGreater(loss_base.item(), 0.0)
        
        # Case 1: Push the two normal samples (index 0 and 2) apart
        # Index 2 is changed to be orthogonal to index 0
        z1_normal_apart = z1_base.clone()
        z1_normal_apart[2] = torch.tensor([0.0, 1.0])
        z2_normal_apart = z2_base.clone()
        z2_normal_apart[2] = torch.tensor([0.0, 1.0])
        loss_normal_apart = supervised_contrastive_loss(z1_normal_apart, z2_normal_apart, labels, temperature=0.1)
        
        # Since Normal-to-Normal is a positive pair, pushing them apart must INCREASE the loss
        self.assertGreater(loss_normal_apart.item(), loss_base.item())
        
        # Case 2: Push the two anomaly samples (index 1 and 3) apart
        # Index 3 is changed to be orthogonal to index 1
        # Under normal alignment, anomaly-to-anomaly is not a positive pair, so pushing them apart
        # does not disrupt any positive pair.
        z1_anomaly_apart = z1_base.clone()
        z1_anomaly_apart[3] = torch.tensor([1.0, 0.0])
        z2_anomaly_apart = z2_base.clone()
        z2_anomaly_apart[3] = torch.tensor([1.0, 0.0])
        loss_anomaly_apart = supervised_contrastive_loss(z1_anomaly_apart, z2_anomaly_apart, labels, temperature=0.1)
        
        # The loss for pushing anomalies apart should not be higher than the loss of pushing normals apart.
        self.assertLess(loss_anomaly_apart.item(), loss_normal_apart.item())

    def test_no_self_loops(self):
        # Test client GATLayer
        from gnn_models import GATLayer
        layer = GATLayer(input_dim=4, node_num=5, hidden_dim=8, client_topk=3)
        h_emb = torch.randn(10, 8)  # 2 batches of 5 nodes
        edge_index = layer._build_dynamic_graph(h_emb)
        self.assertEqual(edge_index.shape[0], 2)
        # Check if there are any self-loops: no column should have edge_index[0, k] == edge_index[1, k]
        self.assertFalse((edge_index[0] == edge_index[1]).any().item(), "Client GATLayer created self-loops")

        # Test global graph construction in FedGATSageSystem
        system = FedGATSageSystem(data_dir="data", num_clients=2, device="cpu")
        system.initialize_models(input_dim=2, hidden_dim=8, num_classes=2, client_node_nums=[3, 4])
        # Total nodes N_global = 3 + 4 = 7
        # Test B > 1 case
        h_global = torch.randn(14, 8)  # 2 batches of 7 nodes
        edge_index_global = system._build_global_graph(h_global, topk=3)
        self.assertFalse((edge_index_global[0] == edge_index_global[1]).any().item(), "Global graph (B>1) created self-loops")

        # Test B = 1 case
        h_global_single = torch.randn(7, 8)  # 1 batch of 7 nodes
        edge_index_global_single = system._build_global_graph(h_global_single, topk=3)
        self.assertFalse((edge_index_global_single[0] == edge_index_global_single[1]).any().item(), "Global graph (B=1) created self-loops")

    def test_aligned_temporal_masking(self):
        from federated_learning import augment_contrastive
        # Shape: (B, window_size, num_sensors)
        B, window_size, num_sensors = 10, 50, 6
        # Use a tensor where each element is distinct and non-zero to detect changes
        x = torch.arange(1, B * window_size * num_sensors + 1, dtype=torch.float32).view(B, window_size, num_sensors)
        
        # Run multiple times to guarantee we hit temporal masking
        masked_found = False
        for _ in range(10):
            x_aug = augment_contrastive(x)
            mask_zero = (x_aug == 0.0)
            for b in range(B):
                m_b = mask_zero[b]
                if m_b.any():
                    masked_found = True
                    # Every sensor in this batch item should have the exact same mask values
                    first_sensor_mask = m_b[:, 0]
                    for s in range(1, num_sensors):
                        self.assertTrue(torch.equal(first_sensor_mask, m_b[:, s]), f"Mask not aligned for sensor {s}")
        self.assertTrue(masked_found, "Temporal masking did not trigger in any run")

    def test_iqr_flooring(self):
        import numpy as np
        # Check that np.maximum(iqrs, 0.05) works as expected
        iqrs = np.array([0.0, 0.01, 0.04, 0.05, 0.1, 1.0])
        safe_iqrs = np.maximum(iqrs, 0.05)
        np.testing.assert_array_equal(safe_iqrs, np.array([0.05, 0.05, 0.05, 0.05, 0.1, 1.0]))

if __name__ == "__main__":
    unittest.main()
