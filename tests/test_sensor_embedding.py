import os
import unittest
import torch
import torch.nn as nn
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnn_models import GATLayer
from federated_learning import FedGATSageSystem

class TestSensorEmbedding(unittest.TestCase):
    def test_gat_layer_sensor_embedding_init(self):
        # Test initialization with default parameters
        layer = GATLayer(
            input_dim=10,
            node_num=20,
            hidden_dim=32,
            use_sensor_embeddings=True,
            sensor_embed_mode="both",
            sensor_embedding_dim=None
        )
        self.assertTrue(layer.use_sensor_embeddings)
        self.assertEqual(layer.sensor_embedding_dim, 32)
        self.assertEqual(layer.sensor_embedding.shape, (20, 32))
        self.assertIsInstance(layer.sensor_project, nn.Identity)

        # Test initialization with custom embedding dimension
        layer_custom = GATLayer(
            input_dim=10,
            node_num=20,
            hidden_dim=32,
            use_sensor_embeddings=True,
            sensor_embed_mode="node_feature",
            sensor_embedding_dim=16
        )
        self.assertTrue(layer_custom.use_sensor_embeddings)
        self.assertEqual(layer_custom.sensor_embedding_dim, 16)
        self.assertEqual(layer_custom.sensor_embedding.shape, (20, 16))
        self.assertIsInstance(layer_custom.sensor_project, nn.Linear)

    def test_gat_layer_forward_modes(self):
        # Generate dummy input of shape (B * node_num, input_dim)
        B = 2
        node_num = 15
        input_dim = 8
        x = torch.randn(B * node_num, input_dim)

        # Test each mode
        for mode in ["node_feature", "graph_construction", "both"]:
            layer = GATLayer(
                input_dim=input_dim,
                node_num=node_num,
                hidden_dim=32,
                use_sensor_embeddings=True,
                sensor_embed_mode=mode,
                sensor_embedding_dim=16
            )
            out = layer(x)
            # Output shape depends on use_concat_skip (default: True, so shape should be B*node_num, hidden_dim * 2)
            self.assertEqual(out.shape, (B * node_num, 64))

    def test_gat_layer_gradient_flow(self):
        # Verify that gradients propagate back to the sensor embeddings
        B = 2
        node_num = 10
        input_dim = 8
        x = torch.randn(B * node_num, input_dim)

        layer = GATLayer(
            input_dim=input_dim,
            node_num=node_num,
            hidden_dim=16,
            use_sensor_embeddings=True,
            sensor_embed_mode="both",
            sensor_embedding_dim=8
        )
        
        # Ensure requires_grad is True for sensor embeddings
        self.assertTrue(layer.sensor_embedding.requires_grad)
        
        out = layer(x)
        loss = out.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(layer.sensor_embedding.grad)
        # Ensure gradient has non-zero values
        self.assertTrue(torch.any(layer.sensor_embedding.grad != 0))

    def test_system_checkpoint_sensor_embedding(self):
        import tempfile
        import shutil

        # Test that FedGATSageSystem checkpointing saves/loads configurations
        checkpoint_dir = tempfile.mkdtemp()
        try:
            system = FedGATSageSystem(
                data_dir="data",
                num_clients=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir
            )
            system.initialize_models(
                input_dim=4,
                hidden_dim=16,
                num_classes=2,
                client_node_nums=[5, 5],
                use_sensor_embeddings=True,
                sensor_embed_mode="graph_construction",
                sensor_embedding_dim=8
            )

            # Assert properties are set on system
            self.assertTrue(system.use_sensor_embeddings)
            self.assertEqual(system.sensor_embed_mode, "graph_construction")
            self.assertEqual(system.sensor_embedding_dim, 8)

            # Assert they are forwarded to GATLayer instances
            for model in system.client_models.values():
                self.assertTrue(model.use_sensor_embeddings)
                self.assertEqual(model.sensor_embed_mode, "graph_construction")
                self.assertEqual(model.sensor_embedding_dim, 8)

            # Save checkpoint
            system.save_checkpoint(checkpoint_dir, round_idx=0)

            # Create a new system and load the checkpoint
            new_system = FedGATSageSystem(
                data_dir="data",
                num_clients=2,
                device="cpu",
                checkpoint_dir=checkpoint_dir
            )
            new_system.load_checkpoint(os.path.join(checkpoint_dir, "checkpoint_round_1.pt"))

            # Assert properties are restored
            self.assertTrue(new_system.use_sensor_embeddings)
            self.assertEqual(new_system.sensor_embed_mode, "graph_construction")
            self.assertEqual(new_system.sensor_embedding_dim, 8)

            for model in new_system.client_models.values():
                self.assertTrue(model.use_sensor_embeddings)
                self.assertEqual(model.sensor_embed_mode, "graph_construction")
                self.assertEqual(model.sensor_embedding_dim, 8)

        finally:
            shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    unittest.main()
