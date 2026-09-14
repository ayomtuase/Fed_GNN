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
from fedgatsage_tune import create_objective, detect_client_nodes, parse_args


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
        """Verify that kernel_preset and static search space allow Optuna visualizations without ValueError."""
        kernel_templates = {
            "single_small": [3],
            "single_medium": [7],
            "dual_fast": [3, 7],
            "tri_balanced": [3, 7, 15],
            "tri_deep": [7, 15, 31],
            "quad_multiscale": [3, 7, 15, 31],
        }

        def mock_objective(trial: optuna.Trial) -> float:
            kernel_preset = trial.suggest_categorical("kernel_preset", list(kernel_templates.keys()))
            selected_kernels = kernel_templates[kernel_preset]
            trial.set_user_attr("selected_kernels", selected_kernels)

            lr_client = trial.suggest_float("lr_client", 1e-5, 1e-2, log=True)
            lr_server = trial.suggest_float("lr_server", 1e-5, 1e-2, log=True)
            temporal_mask_ratio = trial.suggest_float("temporal_mask_ratio", 0.05, 0.50)
            jitter_noise = trial.suggest_float("jitter_noise", 0.005, 0.10)
            dp_clip_bound = trial.suggest_float("dp_clip_bound", 5.0, 50.0)
            window_size = trial.suggest_int("window_size", 40, 120, step=10)

            return float(window_size * 0.01 + len(selected_kernels) * 0.001 + lr_client + lr_server + dp_clip_bound * 0.001)

        study = optuna.create_study(direction="minimize")
        study.optimize(mock_objective, n_trials=10)

        for trial in study.trials:
            self.assertIn("kernel_preset", trial.params)
            selected_kernels = trial.user_attrs["selected_kernels"]
            self.assertGreaterEqual(trial.params["window_size"], max(selected_kernels))

        hist = plot_optimization_history(study)
        self.assertIsNotNone(hist)

        imp = plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
        self.assertIsNotNone(imp)

        par = plot_parallel_coordinate(study)
        self.assertIsNotNone(par)

        sl = plot_slice(study)
        self.assertIsNotNone(sl)

    def test_end_to_end_objective_execution(self):
        """Verify that create_objective runs end-to-end with FedGATSageSystem and multi-scale temporal encoder."""
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
        user_attrs = study.trials[0].user_attrs

        # Sampled parameters
        self.assertIn("kernel_preset", params)
        self.assertIn("window_size", params)
        self.assertIn("lr_client", params)
        self.assertIn("lr_server", params)
        self.assertIn("temporal_mask_ratio", params)
        self.assertIn("jitter_noise", params)
        self.assertIn("dp_clip_bound", params)

        # Accommodates max kernel size
        self.assertIn("selected_kernels", user_attrs)
        self.assertGreaterEqual(params["window_size"], max(user_attrs["selected_kernels"]))

        # Frozen parameters should not be in trial.params to avoid shifting distributions
        self.assertNotIn("hidden_dim", params)
        self.assertNotIn("client_topk", params)
        self.assertNotIn("global_topk", params)
        self.assertNotIn("server_model_type", params)
        self.assertNotIn("num_heads", params)
        self.assertNotIn("disable_conv", params)
        self.assertNotIn("use_contrastive", params)
        self.assertNotIn("dp_noise_multiplier", params)

        # Frozen parameters recorded in user_attrs
        self.assertIn("frozen_params", user_attrs)
        frozen = user_attrs["frozen_params"]
        self.assertEqual(frozen["hidden_dim"], 512)
        self.assertEqual(frozen["sensor_embed_mode"], "both")
        self.assertEqual(frozen["sensor_embedding_dim"], 512)
        self.assertEqual(frozen["client_topk"], 0.8)
        self.assertEqual(frozen["global_topk"], 40)
        self.assertEqual(frozen["server_model_type"], "GraphSAGE")
        self.assertEqual(frozen["num_heads"], 2)
        self.assertFalse(frozen["disable_conv"])
        self.assertTrue(frozen["use_contrastive"])
        self.assertEqual(frozen["contrastive_weight"], 0.04)
        self.assertEqual(frozen["contrastive_temp"], 0.19)
        self.assertEqual(frozen["dp_noise_multiplier"], 0.0015321405394566644)

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

    def test_parse_args_defaults(self):
        """Verify default hyperparameter configuration and safe batch size."""
        import unittest.mock as mock
        with mock.patch("sys.argv", ["fedgatsage_tune.py"]):
            args = parse_args()
            self.assertEqual(args.batch_size, 256)


if __name__ == "__main__":
    unittest.main()
