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

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


def norm(train, test):
    """Normalize training and test data to [0, 1] range using MinMaxScaler."""
    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train)
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)
    return train_ret, test_ret



def save_client_data(train_df, output_dir, num_clients, seed):
    """Save the training split into client-specific CSV files (vertical split)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle train_df rows (with seed for reproducibility) to mix Normal/Attack snapshots.
    # The same shuffled row order is kept for all clients to ensure row alignment.
    shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Get all feature columns (excluding 'attack')
    feature_cols = [col for col in shuffled.columns if col != "attack"]
    
    # Split the feature columns as evenly as possible among clients
    client_feature_groups = np.array_split(feature_cols, num_clients)
    
    for i, feature_group in enumerate(client_feature_groups, start=1):
        client_cols = list(feature_group) + ["attack"]
        client_df = shuffled[client_cols]
        client_path = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(client_path, index=False)
        logger.info(
            f"Saved client {i} data to {client_path} "
            f"({client_df.shape[0]} records, {client_df.shape[1] - 1} features)"
        )


def prepare_swat_dataset(df, output_dir, num_clients, test_ratio, seed):
    """
    Split raw SWAT-style data into train/test, normalize,
    and partition training data vertically among federated clients.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    # Explicitly drop Timestamp column if present
    if "Timestamp" in df.columns:
        logger.info("Dropping 'Timestamp' column")
        df = df.drop(columns=["Timestamp"])

    # Drop any non-numeric feature columns (e.g. timestamp strings) before normalization
    non_numeric_cols = [
        col
        for col in df.columns
        if col != "Normal/Attack" and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        logger.info(
            f"Dropping non-numeric columns before normalization: {non_numeric_cols}"
        )
        df = df.drop(columns=non_numeric_cols)

    # Handle the "Normal/Attack" label column
    if "Normal/Attack" in df.columns:
        df["attack"] = df.pop("Normal/Attack").map({"Attack": 1, "Normal": 0})
        logger.info("Converted 'Normal/Attack' column to binary labels")
    else:
        logger.error("Label column 'Normal/Attack' not found in dataset")
        return

    # Split into train and test
    train_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, shuffle=True
    )

    # Extract labels
    train_labels = train_df["attack"].values
    test_labels = test_df["attack"].values

    # Drop labels from feature data
    train_features = train_df.drop(columns=["attack"]).values
    test_features = test_df.drop(columns=["attack"]).values

    # Normalize
    x_train, x_test = norm(train_features, test_features)
    logger.info("Normalized data to [0, 1] range")

    # Reconstruct DataFrames
    feature_cols = train_df.drop(columns=["attack"]).columns.tolist()
    train_df = pd.DataFrame(x_train, columns=feature_cols)
    test_df = pd.DataFrame(x_test, columns=feature_cols)

    train_df["attack"] = train_labels
    test_df["attack"] = test_labels

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    # Save test set
    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved test set to {test_path} ({len(test_df)} records)")

    # Save training data split vertically among clients
    save_client_data(train_df, output_dir, num_clients, seed)


def main():
    args = parse_args()

    logger.info(f"Starting preprocessing with input: {args.input_file}")

    # if not os.path.exists(args.input_file):
    #     logger.error(f"Could not find the input file: {args.input_file}")
    #     logger.warning("Generating a dummy dataset for demonstration purposes...")
    #     create_dummy_dataset(args.input_file)

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
