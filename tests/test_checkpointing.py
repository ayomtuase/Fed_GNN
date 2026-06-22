import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn
import sys

# Add src to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from federated_learning import FedGATSageSystem

class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.system = FedGATSageSystem(
            data_dir=self.data_dir,
            num_clients=2,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir
        )

        # Initialize minimal mock models
        self.system.initialize_models(
            input_dim=1,
            hidden_dim=8,
            num_classes=2,
            client_node_nums=[4, 4],
            use_concat_skip=True
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_safe_torch_save(self):
        # Test atomic file save
        save_path = os.path.join(self.checkpoint_dir, "test_save.pt")
        obj = {"data": [1, 2, 3]}
        self.system._safe_torch_save(obj, save_path)
        
        self.assertTrue(os.path.exists(save_path))
        loaded = torch.load(save_path)
        self.assertEqual(loaded["data"], [1, 2, 3])

    def test_create_checkpoint_dict(self):
        # Test creation of checkpoint dict including opt/sched/scaler if they exist
        self.system.results = {"training_losses": [0.5, 0.4], "round_times": [1.0, 1.2]}
        self.system.best_loss = 0.4
        self.system.best_round = 1

        # Mock optimizer, scheduler, scaler
        params = list(self.system.global_model.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scaler = torch.amp.GradScaler("cpu", enabled=True)

        self.system.optimizer = optimizer
        self.system.scheduler = scheduler
        self.system.scaler = scaler

        checkpoint = self.system._create_checkpoint_dict(round_idx=1)

        self.assertEqual(checkpoint["round_idx"], 1)
        self.assertEqual(checkpoint["results"]["training_losses"], [0.5, 0.4])
        self.assertEqual(checkpoint["best_loss"], 0.4)
        self.assertEqual(checkpoint["best_round"], 1)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("scheduler", checkpoint)
        self.assertIn("scaler", checkpoint)

    def test_checkpoint_candidates_and_fallback(self):
        # Save multiple checkpoints
        self.system.results = {"training_losses": [0.5]}
        self.system.save_checkpoint(self.checkpoint_dir, round_idx=0) # round 1
        
        self.system.results = {"training_losses": [0.5, 0.4]}
        self.system.save_checkpoint(self.checkpoint_dir, round_idx=1) # round 2

        # Check candidates lists latest first
        candidates = self.system._get_checkpoint_candidates(self.checkpoint_dir)
        # Expected: checkpoint_latest.pt, checkpoint_round_2.pt, checkpoint_round_1.pt
        self.assertEqual(len(candidates), 3)
        self.assertTrue(candidates[0].endswith("checkpoint_latest.pt"))
        self.assertTrue(candidates[1].endswith("checkpoint_round_2.pt"))
        self.assertTrue(candidates[2].endswith("checkpoint_round_1.pt"))

        # Test corrupt file fallback
        # Corrupt the latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        with open(latest_path, "w") as f:
            f.write("corrupted data")

        # Now loading without path should automatically fall back to checkpoint_round_2.pt
        round_idx = self.system.load_checkpoint()
        self.assertEqual(round_idx, 1) # successfully loaded round 2 (round_idx=1)
        self.assertEqual(len(self.system.results["training_losses"]), 2)

    def test_load_training_state_flag(self):
        # Save a checkpoint
        self.system.results = {"training_losses": [0.5, 0.4], "round_times": [1.0, 1.2]}
        self.system.save_checkpoint(self.checkpoint_dir, round_idx=1)

        # Create a new system and load weights ONLY
        new_system = FedGATSageSystem(
            data_dir=self.data_dir,
            num_clients=2,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir
        )
        # Verify initial results are empty
        self.assertEqual(new_system.results, {"training_losses": [], "round_times": []})

        # Load checkpoint with load_training_state=False
        best_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        round_idx = new_system.load_checkpoint(best_path, load_training_state=False)

        self.assertEqual(round_idx, 1)
        # Results should still be empty
        self.assertEqual(new_system.results, {"training_losses": [], "round_times": []})

        # Load checkpoint with load_training_state=True
        round_idx2 = new_system.load_checkpoint(best_path, load_training_state=True)
        self.assertEqual(round_idx2, 1)
        # Results should be loaded
        self.assertEqual(new_system.results["training_losses"], [0.5, 0.4])

if __name__ == "__main__":
    unittest.main()
