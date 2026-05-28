"""
SWAT Data Preprocessing Script for FedGATSAGE
========================================

This script prepares raw SWAT network traffic data (CSV format) for FedGATSage.
It performs the following steps:
1. Loads the raw dataset.
2. Splits the data into training and testing sets.
3. Partitions the training set among federated clients.
4. Saves output into the target directory as:
   data/
     ├── client_1.csv
     ├── client_2.csv
     ├── ...
     └── test.csv

Usage:
    python preprocess_data.py --input_file path/to/dataset.csv --output_dir data --num_clients 5
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SWAT Data Preprocessing for FedGATSage"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the raw CSV dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--num_clients", type=int, default=5, help="Number of federated clients"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of data to use for testing"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def infer_label_column(df):
    label_candidates = ["attack", "label", "is_attack", "class"]
    normalized = {col.strip().lower(): col for col in df.columns}
    for candidate in label_candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def save_client_data(train_df, output_dir, num_clients, seed):
    """Save the training split into client-specific CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    client_dfs = np.array_split(shuffled, num_clients)
    for i, client_df in enumerate(client_dfs, start=1):
        client_path = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(client_path, index=False)
        logger.info(
            f"Saved client {i} data to {client_path} ({len(client_df)} records)"
        )


def prepare_swat_dataset(df, output_dir, num_clients, test_ratio, seed):
    """Split raw SWAT-style data into train/test and client-specific training shards."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    label_col = infer_label_column(df)
    stratify = None
    if label_col is not None and df[label_col].nunique() > 1:
        stratify = df[label_col]
        logger.info(f"Using '{label_col}' as label column for stratified split")
    else:
        logger.info(
            "No suitable label column detected or insufficient classes; performing random train/test split"
        )

    train_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, shuffle=True, stratify=stratify
    )

    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test set to {test_path} ({len(test_df)} records)")

    save_client_data(train_df, output_dir, num_clients, seed)


def main():
    args = parse_args()

    logger.info(f"Starting preprocessing with input: {args.input_file}")

    if not os.path.exists(args.input_file):
        logger.error(f"Could not find the input file: {args.input_file}")
        logger.warning("Generating a dummy dataset for demonstration purposes...")
        create_dummy_dataset(args.input_file)

    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Successfully loaded dataset with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    prepare_swat_dataset(
        df, args.output_dir, args.num_clients, args.test_ratio, args.seed
    )
    logger.info("All done! Data preprocessing is complete.")


# def create_dummy_dataset(filepath):
#     """Create a dummy dataset for testing/demonstration"""
#     logger.info(f"Generating dummy data at {filepath}")

#     num_rows = 1000
#     data = {
#         'Src IP': [f'192.168.1.{i%255}' for i in range(num_rows)],
#         'Dst IP': [f'10.0.0.{i%255}' for i in range(num_rows)],
#         'Src Port': np.random.randint(1024, 65535, num_rows),
#         'Dst Port': np.random.randint(1, 1024, num_rows),
#         'Protocol': np.random.choice(['TCP', 'UDP'], num_rows),
#         'Flow Duration': np.random.randint(100, 100000, num_rows),
#         'Tot Fwd Pkts': np.random.randint(1, 100, num_rows),
#         'Tot Bwd Pkts': np.random.randint(1, 100, num_rows),
#         'TotLen Fwd Pkts': np.random.randint(64, 15000, num_rows),
#         'TotLen Bwd Pkts': np.random.randint(64, 15000, num_rows),
#         'Flow IAT Mean': np.random.uniform(0.1, 100.0, num_rows),
#         'Flow IAT Std': np.random.uniform(0.0, 10.0, num_rows),
#         'Flow Pkts/s': np.random.uniform(0.1, 1000.0, num_rows),
#         'Attack': np.random.choice(['Benign', 'DoS', 'PortScan', 'WebAttack'], num_rows, p=[0.7, 0.1, 0.1, 0.1])
#     }

#     df = pd.DataFrame(data)

#     for metric in ['betweenness', 'pagerank', 'degree', 'closeness', 'eigenvector', 'k_core', 'modularity']:
#         df[f'src_{metric}'] = np.random.uniform(0, 1, num_rows)
#         df[f'dst_{metric}'] = np.random.uniform(0, 1, num_rows)

#     df.to_csv(filepath, index=False)
#     logger.info("Dummy dataset created.")

if __name__ == "__main__":
    main()
