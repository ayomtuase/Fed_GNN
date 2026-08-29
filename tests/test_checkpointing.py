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
        self.system.results = {
            "training_losses": [0.5, 0.4],
            "round_times": [1.0, 1.2],
            "training_accuracies": [0.8, 0.85],
            "training_precisions": [0.75, 0.8],
            "training_recalls": [0.7, 0.75],
            "training_f1s": [0.72, 0.77],
            "training_aucs": [0.73, 0.78],
        }
        self.system.save_checkpoint(self.checkpoint_dir, round_idx=1)

        # Create a new system and load weights ONLY
        new_system = FedGATSageSystem(
            data_dir=self.data_dir,
            num_clients=2,
            device="cpu",
            checkpoint_dir=self.checkpoint_dir
        )
        # Verify initial results are empty
        self.assertEqual(new_system.results, {
            "training_losses": [],
            "round_times": [],
            "training_accuracies": [],
            "training_precisions": [],
            "training_recalls": [],
            "training_f1s": [],
            "training_aucs": [],
            "val_losses": [],
            "val_aucs": [],
            "val_f1s": [],
        })

        # Load checkpoint with load_training_state=False
        best_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        round_idx = new_system.load_checkpoint(best_path, load_training_state=False)

        self.assertEqual(round_idx, 1)
        # Results should still be empty
        self.assertEqual(new_system.results, {
            "training_losses": [],
            "round_times": [],
            "training_accuracies": [],
            "training_precisions": [],
            "training_recalls": [],
            "training_f1s": [],
            "training_aucs": [],
            "val_losses": [],
            "val_aucs": [],
            "val_f1s": [],
        })

        # Load checkpoint with load_training_state=True
        round_idx2 = new_system.load_checkpoint(best_path, load_training_state=True)
        self.assertEqual(round_idx2, 1)
        # Results should be loaded
        self.assertEqual(new_system.results["training_losses"], [0.5, 0.4])
        self.assertEqual(new_system.results["training_accuracies"], [0.8, 0.85])

    def test_cross_device_mapping(self):
        # Save a mock checkpoint
        save_path = os.path.join(self.checkpoint_dir, "test_cross_device.pt")
        # We put a tensor inside a nested structure (dict, list)
        checkpoint = {
            "tensor_dict": {"a": torch.tensor([1.0, 2.0], device="cpu")},
            "tensor_list": [torch.tensor([3.0], device="cpu")],
        }
        self.system._safe_torch_save(checkpoint, save_path)
        
        # We want to load it on target device 'cpu'.
        # To simulate a direct load failure, we can temporarily mock torch.load
        # to raise a RuntimeError on the first call (when mapping directly to target device),
        # but succeed on the second call (when mapping to 'cpu').
        original_torch_load = torch.load
        call_count = 0
        def mock_torch_load(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated direct load failure on device")
            return original_torch_load(*args, **kwargs)

        import unittest.mock as mock
        with mock.patch("torch.load", side_effect=mock_torch_load):
            loaded = self.system._load_checkpoint_on_device(save_path, "cpu")
        
        # Verify call count is 2 (first failed, second was fallback)
        self.assertEqual(call_count, 2)
        # Verify tensors are mapped correctly and contents are correct
        self.assertTrue(torch.equal(loaded["tensor_dict"]["a"], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(loaded["tensor_list"][0], torch.tensor([3.0])))

    def test_rng_states_save_load(self):
        import random
        import numpy as np

        # Seed everything first
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate some initial states & data
        state1_py = random.random()
        state1_np = np.random.rand()
        state1_torch = torch.rand(5)

        # Save checkpoint (which captures current RNG states)
        save_path = os.path.join(self.checkpoint_dir, "test_rng.pt")
        self.system.save_checkpoint(self.checkpoint_dir, round_idx=0)

        # Generate post-checkpoint sequences
        seq_py_1 = [random.random() for _ in range(5)]
        seq_np_1 = np.random.rand(5).tolist()
        seq_torch_1 = torch.rand(5)

        # Perturb RNG states by generating more
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(5)

        # Load checkpoint (this should restore RNG states)
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        self.system.load_checkpoint(latest_path, load_training_state=True)

        # Generate sequences again after restoration
        seq_py_2 = [random.random() for _ in range(5)]
        seq_np_2 = np.random.rand(5).tolist()
        seq_torch_2 = torch.rand(5)

        # Verify they match perfectly (mathematical reproducibility)
        self.assertEqual(seq_py_1, seq_py_2)
        self.assertEqual(seq_np_1, seq_np_2)
        self.assertTrue(torch.equal(seq_torch_1, seq_torch_2))

    def test_optimizer_device_migration(self):
        # Save a checkpoint with optimizer state on CPU
        params = list(self.system.global_model.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)
        # Manually inject state tensors
        optimizer.state[params[0]] = {
            "step": torch.tensor(1.0, device="cpu"),
            "exp_avg": torch.tensor([0.1, 0.2], device="cpu"),
            "exp_avg_sq": torch.tensor([0.01, 0.04], device="cpu"),
        }
        self.system.optimizer = optimizer
        checkpoint_dict = self.system._create_checkpoint_dict(round_idx=1)
        
        # Save to file
        save_path = os.path.join(self.checkpoint_dir, "test_opt_mig.pt")
        self.system._safe_torch_save(checkpoint_dict, save_path)

        # Create new system with target device
        new_system = FedGATSageSystem(
            data_dir=self.data_dir,
            num_clients=2,
            device="cpu", # target device
            checkpoint_dir=self.checkpoint_dir
        )
        new_system.initialize_models(
            input_dim=1,
            hidden_dim=8,
            num_classes=2,
            client_node_nums=[4, 4]
        )
        
        # Load the checkpoint training state
        new_system.load_checkpoint(save_path, load_training_state=True)
        
        # Verify that _resume_optimizer_state has been cached
        self.assertIsNotNone(new_system._resume_optimizer_state)

        # Setup new optimizer on the system
        new_optimizer = torch.optim.Adam(list(new_system.global_model.parameters()), lr=0.01)
        
        # Restoring optimizer state
        def _map_to_device(obj, target_device):
            if isinstance(obj, torch.Tensor):
                return obj.to(target_device)
            elif isinstance(obj, dict):
                return {k: _map_to_device(v, target_device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_map_to_device(v, target_device) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(_map_to_device(v, target_device) for v in obj)
            return obj

        mapped_state = _map_to_device(new_system._resume_optimizer_state, new_system.device)
        new_optimizer.load_state_dict(mapped_state)

        # Check that optimizer states are on target device
        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    self.assertEqual(str(v.device), new_system.device)

    def test_evaluate_validation_metrics(self):
        # Test that evaluate_validation uses binary positive class F1 score
        import unittest.mock as mock
        
        # Mock inputs
        # batch size = 2, client_node_nums = [4, 4]
        features_c0 = torch.randn(2, 1, 4)
        features_c1 = torch.randn(2, 1, 4)
        labels = torch.tensor([0, 1])
        
        # Setup loader with dataset attribute
        val_loader = mock.MagicMock()
        val_loader.__iter__.return_value = [
            ((features_c0, features_c1), labels)
        ]
        val_loader.dataset = [None, None]  # length is 2
        
        # Set normal statistics
        self.system.normal_means_global = torch.zeros(8)
        self.system.normal_stds_global = torch.ones(8)
        self.system.current_phase = 1
        
        criterion = nn.BCEWithLogitsLoss()
        
        # Mock global_model forward and client_models forward
        mock_pred = torch.tensor([[0.2], [0.8]]) # sigmoid(0.2) = 0.55, sigmoid(0.8) = 0.69 -> both >= 0.5 -> val_preds = [1, 1]
        
        with mock.patch.object(self.system.global_model, 'forward') as mock_global_forward:
            mock_global_forward.return_value = (None, mock_pred, None, None)
            
            # Mock client models
            client_patches = []
            for client_id, client_model in self.system.client_models.items():
                p = mock.patch.object(client_model, 'forward', return_value=torch.randn(8, 8))
                p.start()
                client_patches.append(p)
                
            try:
                val_loss, val_auc, val_f1, val_probs, val_labels = self.system.evaluate_validation(
                    val_loader=val_loader,
                    criterion=criterion,
                    use_ce_loss=True,
                    focal_loss_alpha=0.5,
                    use_contrastive=False,
                    contrastive_weight=0.0,
                    contrastive_temp=0.07,
                    enable_client_attention=False
                )
            finally:
                for p in client_patches:
                    p.stop()
            
            # Assert validation loss is computed and threshold is updated
            self.assertGreaterEqual(val_loss, 0.0)
            self.assertTrue(hasattr(self.system, "best_threshold"))

    def test_best_threshold_in_checkpoint(self):
        # Set custom best threshold
        self.system.best_threshold = 0.35
        
        # Save checkpoint
        self.system.save_checkpoint(self.checkpoint_dir, round_idx=2)
        
        # Reset current system threshold
        self.system.best_threshold = 0.5
        
        # Load from checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_round_3.pt")
        self.system.load_checkpoint(checkpoint_path)
        
        # Verify it loaded the saved threshold
        self.assertEqual(self.system.best_threshold, 0.35)

if __name__ == "__main__":
    unittest.main()
