import os
_real_exists = os.path.exists
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add preprocess_data to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output files
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    @patch('preprocess_data.parse_args')
    @patch('os.path.exists')
    @patch('pandas.read_excel')
    def test_preprocessing_pipeline(self, mock_read_excel, mock_exists, mock_parse_args):
        # Setup mock arguments
        mock_args = MagicMock()
        mock_args.normal_file = "mock_normal.xlsx"
        mock_args.attack_file = "mock_attack.xlsx"
        mock_args.output_dir = self.test_dir
        mock_args.chunk_size = 1000
        mock_args.downsample_factor = 1
        mock_args.window_size = 10
        mock_args.seed = 42
        mock_args.val_ratio = 0.25
        mock_args.test_ratio = 0.25
        mock_parse_args.return_value = mock_args

        # Mock os.path.exists to return True for the excel files
        def side_effect_exists(path):
            if "mock_normal.xlsx" in path or "mock_attack.xlsx" in path:
                return True
            return _real_exists(path)
        mock_exists.side_effect = side_effect_exists

        # Create dummy columns representing stages 1 to 6
        columns = ["Timestamp"]
        for stage in range(1, 7):
            columns.append(f"{stage}_sensor_A")
            columns.append(f"{stage}_sensor_B")
        columns.append("Normal/Attack")

        # Create normal mock data (25600 rows)
        num_normal_rows = 25600
        normal_data = {col: np.random.rand(num_normal_rows) for col in columns if col not in ["Timestamp", "Normal/Attack"]}
        normal_data["Timestamp"] = [f"2026-08-21 {i}" for i in range(num_normal_rows)]
        normal_data["Normal/Attack"] = ["Normal"] * num_normal_rows
        df_normal = pd.DataFrame(normal_data)

        # Create attack mock data (4000 rows)
        num_attack_rows = 4000
        attack_data = {col: np.random.rand(num_attack_rows) for col in columns if col not in ["Timestamp", "Normal/Attack"]}
        attack_data["Timestamp"] = [f"2026-08-21 {i}" for i in range(num_attack_rows)]
        attack_data["Normal/Attack"] = ["Attack"] * num_attack_rows
        df_attack = pd.DataFrame(attack_data)

        # Mock pandas.read_excel to return normal first, then attack
        mock_read_excel.side_effect = [df_normal, df_attack]

        # Run preprocessing main
        preprocess_data.main()

        # Check that output labels files exist and have correct shapes
        train_labels_path = os.path.join(self.test_dir, "train_labels.npy")
        val_labels_path = os.path.join(self.test_dir, "validation_labels.npy")
        test_labels_path = os.path.join(self.test_dir, "test_labels.npy")

        self.assertTrue(os.path.exists(train_labels_path))
        self.assertTrue(os.path.exists(val_labels_path))
        self.assertTrue(os.path.exists(test_labels_path))

        # Check mapped features for clients
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(self.test_dir, split)
            self.assertTrue(os.path.exists(split_dir))
            for stage in range(1, 7):
                feat_path = os.path.join(split_dir, f"client_{stage}.npy")
                self.assertTrue(os.path.exists(feat_path))
                
                # Load array and check shape
                # Expected length: 6000 for train, 1000 for val/test
                # Expected length: 6000 for train, 400 for val, 1600 for test
                if split == "train":
                    expected_len = 6000
                elif split == "validation":
                    expected_len = 400
                else:
                    expected_len = 1600
                expected_features = 2 # 2 sensors per stage in our mock data
                
                fp = np.load(feat_path, mmap_mode='r')
                self.assertEqual(fp.shape, (expected_len, expected_features))

        # Load labels and check shape
        train_labels = np.load(train_labels_path)
        val_labels = np.load(val_labels_path)
        test_labels = np.load(test_labels_path)

        self.assertEqual(train_labels.shape, (6000,))
        self.assertEqual(val_labels.shape, (400,))
        self.assertEqual(test_labels.shape, (1600,))

if __name__ == "__main__":
    unittest.main()
