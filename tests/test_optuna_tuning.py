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
            contrastive_weight = trial.suggest_float("contrastive_weight", 0.01, 0.1, step=0.01)
            contrastive_temp = trial.suggest_float("contrastive_temp", 0.05, 0.2, step=0.01)
            client_topk = trial.suggest_float("client_topk", 0.4, 0.8, step=0.1)
            global_topk = trial.suggest_int("global_topk", 10, 20, step=2)
            dp_noise_multiplier = trial.suggest_float("dp_noise_multiplier", 0.001, 0.01, log=True)
            sensor_embed_mode = trial.suggest_categorical(
                "sensor_embed_mode", ["graph_construction", "both"]
            )
            window_size = trial.suggest_int("window_size", 10, 120, step=10)
            kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 11, 15, 21, 31])

            return float(window_size * 0.01 + kernel_size * 0.001 + lr_client)

        study = optuna.create_study(direction="minimize")
        study.optimize(mock_objective, n_trials=5)

        hist = plot_optimization_history(study)
        self.assertIsNotNone(hist)

        imp = plot_param_importances(study)
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
        self.assertIn("kernel_size", study.trials[0].params)
        self.assertIn("window_size", study.trials[0].params)

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

        multi_num, multi_nodes = detect_client_nodes(os.path.join(self.temp_dir, "multi_client"))
        self.assertEqual(multi_num, 10)
        self.assertEqual(multi_nodes, list(range(1, 11)))

        # Test error raised when directory has no client arrays and num_clients is None
        empty_dir = os.path.join(self.temp_dir, "empty_data")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            detect_client_nodes(empty_dir, num_clients=None)


if __name__ == "__main__":
    unittest.main()
