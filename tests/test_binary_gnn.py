import os
import unittest
import torch
import torch.nn as nn
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from federated_learning import binary_focal_loss, VFLGradientNormalizer, FedGATSageSystem
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
        
        # Assert server classifier final layer output size is indeed 1 (BCE)
        last_layer = list(system.global_model.classifier.modules())[-1]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, 1)

if __name__ == "__main__":
    unittest.main()
