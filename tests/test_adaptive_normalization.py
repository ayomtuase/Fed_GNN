import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from federated_learning import FedGATSageSystem

class TestAdaptiveNormalization(unittest.TestCase):
    def test_normalize_errors_adaptive_math(self):
        # Initialize FedGATSageSystem with adaptive normalization enabled
        system = FedGATSageSystem(
            data_dir="dummy",
            num_clients=2,
            use_adaptive_normalization=True,
            adaptive_window_size=3
        )
        
        # Mock validation medians and IQRs for the t=0 fallback
        system.val_medians = np.array([1.0, 2.0])
        system.val_iqrs = np.array([0.5, 1.0])
        
        # Raw errors: T=5 snapshots, N=2 nodes
        errors = np.array([
            [1.2, 2.5],  # t=0
            [1.5, 3.0],  # t=1
            [2.0, 4.0],  # t=2
            [2.5, 5.0],  # t=3
            [3.0, 6.0]   # t=4
        ])
        
        normalized = system.normalize_errors_adaptive(errors)
        
        # Verify shape
        self.assertEqual(normalized.shape, errors.shape)
        
        # Verify t=0 fallback (uses static val_medians and val_iqrs)
        # safe_iqrs = max(val_iqrs, 0.05) -> [0.5, 1.0]
        # a_i(0) = |errors[0] - val_medians| / safe_iqrs
        # node 0: |1.2 - 1.0| / 0.5 = 0.2 / 0.5 = 0.4
        # node 1: |2.5 - 2.0| / 1.0 = 0.5 / 1.0 = 0.5
        np.testing.assert_almost_equal(normalized[0], [0.4, 0.5])
        
        # Verify t=1 (history range is [0:1], size=1)
        # history: [[1.2, 2.5]]
        # moving_medians = [1.2, 2.5]
        # moving_iqrs = percentile(75) - percentile(25) = 0.0
        # denominator = moving_iqrs + 0.05 = 0.05
        # a_i(1) = |errors[1] - moving_medians| / denominator
        # node 0: |1.5 - 1.2| / 0.05 = 0.3 / 0.05 = 6.0
        # node 1: |3.0 - 2.5| / 0.05 = 0.5 / 0.05 = 10.0
        np.testing.assert_almost_equal(normalized[1], [6.0, 10.0])
        
        # Verify t=3 (history range is [0:3], size=3 because window_size=3, range [3-3:3] -> [0:3])
        # history: [[1.2, 2.5], [1.5, 3.0], [2.0, 4.0]]
        # moving_medians: median of [1.2, 1.5, 2.0] is 1.5; of [2.5, 3.0, 4.0] is 3.0
        # moving_iqrs:
        #   node 0 values: [1.2, 1.5, 2.0]
        #     p75 = 1.75, p25 = 1.35 (using np.percentile standard linear interpolation)
        #     iqr = 1.75 - 1.35 = 0.4
        #   node 1 values: [2.5, 3.0, 4.0]
        #     p75 = 3.5, p25 = 2.75
        #     iqr = 3.5 - 2.75 = 0.75
        # Let's calculate exactly using np.percentile:
        hist_node_0 = np.array([1.2, 1.5, 2.0])
        hist_node_1 = np.array([2.5, 3.0, 4.0])
        
        expected_median_0 = np.median(hist_node_0)
        expected_median_1 = np.median(hist_node_1)
        expected_iqr_0 = np.percentile(hist_node_0, 75) - np.percentile(hist_node_0, 25)
        expected_iqr_1 = np.percentile(hist_node_1, 75) - np.percentile(hist_node_1, 25)
        
        denom_0 = expected_iqr_0 + 0.05
        denom_1 = expected_iqr_1 + 0.05
        
        expected_val_0 = np.abs(errors[3, 0] - expected_median_0) / denom_0
        expected_val_1 = np.abs(errors[3, 1] - expected_median_1) / denom_1
        
        np.testing.assert_almost_equal(normalized[3, 0], expected_val_0)
        np.testing.assert_almost_equal(normalized[3, 1], expected_val_1)
        
        # Verify t=4 (history range is [4-3:4] -> [1:4], size=3)
        # history: [[1.5, 3.0], [2.0, 4.0], [2.5, 5.0]]
        hist_node_0_t4 = np.array([1.5, 2.0, 2.5])
        hist_node_1_t4 = np.array([3.0, 4.0, 5.0])
        
        expected_median_0_t4 = np.median(hist_node_0_t4)
        expected_median_1_t4 = np.median(hist_node_1_t4)
        expected_iqr_0_t4 = np.percentile(hist_node_0_t4, 75) - np.percentile(hist_node_0_t4, 25)
        expected_iqr_1_t4 = np.percentile(hist_node_1_t4, 75) - np.percentile(hist_node_1_t4, 25)
        
        denom_0_t4 = expected_iqr_0_t4 + 0.05
        denom_1_t4 = expected_iqr_1_t4 + 0.05
        
        expected_val_0_t4 = np.abs(errors[4, 0] - expected_median_0_t4) / denom_0_t4
        expected_val_1_t4 = np.abs(errors[4, 1] - expected_median_1_t4) / denom_1_t4
        
        np.testing.assert_almost_equal(normalized[4, 0], expected_val_0_t4)
        np.testing.assert_almost_equal(normalized[4, 1], expected_val_1_t4)

if __name__ == '__main__':
    unittest.main()
