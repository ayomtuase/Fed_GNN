import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import sys

# Add src and experiments to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments')))

import optuna
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
from fedgatsage_tune import create_objective, detect_client_nodes


class TestOptunaTuning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.output_dir = os.path.join(self.temp_dir, "results")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create synthetic client dataset
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # 2 clients, 4 sensors, 160 timesteps
        self.num_clients = 2
        self.client_node_nums = [4, 4]
        for c in range(self.num_clients):
            train_arr = np.random.randn(160, 4).astype(np.float32)
            val_arr = np.random.randn(160, 4).astype(np.float32)
            np.save(os.path.join(train_dir, f"client_{c+1}.npy"), train_arr)
            np.save(os.path.join(val_dir, f"client_{c+1}.npy"), val_arr)

        # Labels for datasets
        np.save(os.path.join(self.data_dir, "train_labels.npy"), np.zeros(160, dtype=np.int64))
        np.save(os.path.join(self.data_dir, "val_labels.npy"), np.zeros(160, dtype=np.int64))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_search_space_static_and_plots(self):
        """Verify that kernel_size and window_size static search space allows visualization without ValueError."""
        def mock_objective(trial: optuna.Trial) -> float:
            lr_client = trial.suggest_float("lr_client", 1e-4, 1e-2, log=True)
            lr_server = trial.suggest_float("lr_server", 1e-5, 1e-3, log=True)
            use_contrastive = trial.suggest_categorical("use_contrastive", [True, False])
            if use_contrastive:
                contrastive_weight = trial.suggest_float("contrastive_weight", 0.01, 0.1, step=0.01)
                contrastive_temp = trial.suggest_float("contrastive_temp", 0.05, 0.2, step=0.01)
                temporal_mask_ratio = trial.suggest_float("temporal_mask_ratio", 0.05, 0.35, step=0.05)
                jitter_noise = trial.suggest_float("jitter_noise", 0.01, 0.10, step=0.01)
            else:
                contrastive_weight = 0.0
                contrastive_temp = 0.07
                temporal_mask_ratio = 0.15
                jitter_noise = 0.03
            client_topk = trial.suggest_float("client_topk", 0.4, 0.8, step=0.1)
            global_topk = trial.suggest_int("global_topk", 10, 20, step=2)
            dp_clip_bound = trial.suggest_float("dp_clip_bound", 5.0, 50.0, step=2.5)
            dp_noise_multiplier = trial.suggest_float("dp_noise_multiplier", 0.001, 0.01, log=True)
            disable_sensor_embeddings = trial.suggest_categorical(
                "disable_sensor_embeddings", [True, False]
            )
            sensor_embed_mode = trial.suggest_categorical(
                "sensor_embed_mode", ["graph_construction", "both"]
            )
            sensor_embedding_dim = trial.suggest_categorical(
                "sensor_embedding_dim", [64, 128, 256, 512]
            )
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
            server_model_type = trial.suggest_categorical("server_model_type", ["GraphSAGE", "GAT"])
            disable_conv = trial.suggest_categorical("disable_conv", [True, False])
            num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
            window_size = trial.suggest_int("window_size", 10, 120, step=10)
            max_kernel = min(31, window_size if window_size % 2 != 0 else window_size - 1)
            kernel_size = trial.suggest_int("kernel_size", 3, max_kernel, step=2)

            return float(window_size * 0.01 + kernel_size * 0.001 + lr_client + (0.1 if use_contrastive else 0.0))

        study = optuna.create_study(direction="minimize")
        study.optimize(mock_objective, n_trials=10)

        for trial in study.trials:
            self.assertLessEqual(trial.params["kernel_size"], trial.params["window_size"])
            self.assertEqual(trial.params["kernel_size"] % 2, 1)

        hist = plot_optimization_history(study)
        self.assertIsNotNone(hist)

        imp = plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
        self.assertIsNotNone(imp)

        par = plot_parallel_coordinate(study)
        self.assertIsNotNone(par)

        sl = plot_slice(study)
        self.assertIsNotNone(sl)

    def test_end_to_end_objective_execution(self):
        """Verify that create_objective runs end-to-end with FedGATSageSystem."""
        objective = create_objective(
            data_dir=self.data_dir,
            checkpoint_base_dir=self.checkpoint_dir,
            num_clients=self.num_clients,
            client_node_nums=self.client_node_nums,
            max_rounds=1,
            batch_size=16,
            device="cpu",
            max_samples=20,
        )

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1)

        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.COMPLETE)
        params = study.trials[0].params
        self.assertIn("kernel_size", params)
        self.assertIn("window_size", params)
        self.assertLessEqual(params["kernel_size"], params["window_size"])
        self.assertEqual(params["kernel_size"] % 2, 1)
        self.assertIn("use_contrastive", params)
        if params["use_contrastive"]:
            self.assertIn("contrastive_weight", params)
            self.assertIn("contrastive_temp", params)
            self.assertIn("temporal_mask_ratio", params)
            self.assertIn("jitter_noise", params)
        else:
            self.assertNotIn("contrastive_weight", params)
            self.assertNotIn("contrastive_temp", params)
            self.assertNotIn("temporal_mask_ratio", params)
            self.assertNotIn("jitter_noise", params)
        self.assertIn("disable_sensor_embeddings", params)
        self.assertIn("sensor_embedding_dim", params)
        self.assertIn("hidden_dim", params)
        self.assertIn("dp_clip_bound", params)
        self.assertIn("server_model_type", params)
        self.assertIn("disable_conv", params)
        self.assertIn("num_heads", params)

    def test_detect_client_nodes_auto_discovery(self):
        """Verify dynamic detection of client count and node dimensions from data folder."""
        # Auto-detect without passing num_clients (num_clients=None)
        num_clients, node_nums = detect_client_nodes(self.data_dir, num_clients=None)
        self.assertEqual(num_clients, 2)
        self.assertEqual(node_nums, [4, 4])

        # Test natural numerical ordering with multi-digit clients (e.g., client_1 .. client_10)
        multi_client_dir = os.path.join(self.temp_dir, "multi_client", "train")
        os.makedirs(multi_client_dir, exist_ok=True)
        for i in range(1, 11):
            arr = np.zeros((10, i), dtype=np.float32)
            np.save(os.path.join(multi_client_dir, f"client_{i}.npy"), arr)

        # Add extraneous non-digit file like client_scaler.npy or client_backup.npy
        np.save(os.path.join(multi_client_dir, "client_scaler.npy"), np.zeros((10, 99), dtype=np.float32))

        multi_num, multi_nodes = detect_client_nodes(os.path.join(self.temp_dir, "multi_client"))
        self.assertEqual(multi_num, 10)
        self.assertEqual(multi_nodes, list(range(1, 11)))

        # Test error raised when directory has no client arrays and num_clients is None
        empty_dir = os.path.join(self.temp_dir, "empty_data")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            detect_client_nodes(empty_dir, num_clients=None)

    def test_detect_client_nodes_0_indexed(self):
        """Verify dynamic detection when client files are 0-indexed (client_0.npy ... client_4.npy)."""
        zero_idx_dir = os.path.join(self.temp_dir, "zero_idx_client", "train")
        os.makedirs(zero_idx_dir, exist_ok=True)
        expected_nodes = [5, 12, 8, 15, 20]
        for idx, nodes in enumerate(expected_nodes):
            arr = np.zeros((10, nodes), dtype=np.float32)
            np.save(os.path.join(zero_idx_dir, f"client_{idx}.npy"), arr)

        num_clients, node_nums = detect_client_nodes(os.path.join(self.temp_dir, "zero_idx_client"))
        self.assertEqual(num_clients, 5)
        self.assertEqual(node_nums, expected_nodes)


if __name__ == "__main__":
    unittest.main()
