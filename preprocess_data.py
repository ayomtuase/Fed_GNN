"""
SWAT Data Preprocessing Script for FedGATSAGE
========================================

This script prepares raw SWAT network traffic data (CSV format) for FedGATSage.
It performs stage-based vertical splitting, downsampling, sliding window extraction,
index shuffling, and train-validation-test splitting.
"""

import argparse
import logging
import os
import re
import numpy as np
import pandas as pd
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
        "--input_file", type=str, default="data/swat.csv", help="Path to the raw CSV dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed_data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--downsample_factor", type=int, default=10, help="Downsampling factor"
    )
    parser.add_argument(
        "--window_size", type=int, default=10, help="Window size for feature extraction"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Ratio of validation data"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of test data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Starting SWAT preprocessing with input: {args.input_file}")

    if not os.path.exists(args.input_file):
        logger.error(f"Could not find the input file: {args.input_file}")
        return

    try:
        # Load the original swat dataset
        logger.info("Loading dataset (this may take a moment)...")
        df = pd.read_csv(args.input_file)
        logger.info(f"Successfully loaded dataset with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 1. Drop the first 2160 rows of the dataset
    logger.info("Dropping the first 2160 rows of the dataset")
    df = df.iloc[2160:].reset_index(drop=True)

    # Clean columns by stripping whitespace
    df.columns = [col.strip() for col in df.columns]

    # Remove the timestamp column
    if "Timestamp" in df.columns:
        logger.info("Removing 'Timestamp' column")
        df = df.drop(columns=["Timestamp"])

    # 2. Separate out the labels
    if "Normal/Attack" in df.columns:
        # Map Normal/Attack to binary labels: Attack -> 1, Normal -> 0
        labels = df["Normal/Attack"].astype(str).str.strip().map({"Attack": 1, "Normal": 0})
        labels = labels.fillna(0).astype(int).values
        df = df.drop(columns=["Normal/Attack"])
        logger.info("Separated and mapped labels ('Normal' -> 0, 'Attack' -> 1)")
    else:
        logger.error("Label column 'Normal/Attack' not found in dataset")
        return

    # Drop any other non-numeric feature columns
    non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_cols:
        logger.info(f"Dropping non-numeric feature columns: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    # Fill NaN values with column means or 0
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    # 3. Normalize the dataset features
    logger.info("Normalizing dataset features to [0, 1] range using MinMaxScaler")
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_features = scaler.fit_transform(df.values)
    df = pd.DataFrame(normalized_features, columns=df.columns)

    # 4. Split the data vertically by the stages (1 to 6)
    logger.info("Splitting the data vertically by stages (1 to 6)")
    client_features = {stage: [] for stage in range(1, 7)}
    client_cols = {stage: [] for stage in range(1, 7)}

    for col in df.columns:
        match = re.search(r'\d+', col)
        if match:
            numeric_part = match.group()
            stage = int(numeric_part[0])
            if 1 <= stage <= 6:
                client_features[stage].append(df[col].values)
                client_cols[stage].append(col)
            else:
                logger.warning(f"Feature column '{col}' has numeric part starting with digit {stage}, outside 1-6 range. Skipping.")
        else:
            logger.warning(f"Feature column '{col}' does not contain a numeric part. Skipping.")

    # Convert lists of 1D arrays into 2D numpy arrays of shape (N, num_features_stage)
    client_data = {}
    for stage in range(1, 7):
        if not client_features[stage]:
            logger.error(f"No features found for Stage {stage}!")
            return
        client_data[stage] = np.stack(client_features[stage], axis=1)
        logger.info(f"Client {stage} (Stage {stage}) shape: {client_data[stage].shape} with features: {client_cols[stage]}")

    # 5. Downsampling
    downsample_factor = args.downsample_factor
    logger.info(f"Downsampling features and labels by factor: {downsample_factor}")
    N = len(labels)
    downsampled_len = N // downsample_factor

    downsampled_client_data = {}
    for stage in range(1, 7):
        feat = client_data[stage]
        # Trim features to be divisible by downsample_factor
        feat_trimmed = feat[:downsampled_len * downsample_factor]
        # Reshape to (downsampled_len, downsample_factor, num_features)
        feat_reshaped = feat_trimmed.reshape(downsampled_len, downsample_factor, -1)
        # Calculate mean for features
        downsampled_client_data[stage] = feat_reshaped.mean(axis=1)

    labels_trimmed = labels[:downsampled_len * downsample_factor]
    labels_reshaped = labels_trimmed.reshape(downsampled_len, downsample_factor)
    # If 1 is present in the set to be downsampled, the label is 1, else 0
    downsampled_labels = (labels_reshaped.sum(axis=1) > 0).astype(int)

    logger.info(f"Shape after downsampling: features length = {downsampled_len}, labels length = {len(downsampled_labels)}")

    # 6. Extract windows of the dataset (sliding windows of size window_size)
    window_size = args.window_size
    logger.info(f"Extracting windows of size: {window_size}")

    if downsampled_len < window_size:
        logger.error(f"Downsampled length {downsampled_len} is less than window size {window_size}")
        return

    num_windows = downsampled_len - window_size + 1
    windowed_client_data = {}
    for stage in range(1, 7):
        feat = downsampled_client_data[stage]
        num_features = feat.shape[1]
        
        # We want the window shape to be (num_windows, num_features, window_size)
        windows = []
        for i in range(num_windows):
            windows.append(feat[i : i + window_size].T)
        windowed_client_data[stage] = np.array(windows)
        logger.info(f"Client {stage} windowed features shape: {windowed_client_data[stage].shape}")

    # Labels for the windows: the label of the window is the label of its last element
    windowed_labels = downsampled_labels[window_size - 1:]
    logger.info(f"Windowed labels shape: {windowed_labels.shape}")

    # 7. Shuffling
    logger.info(f"Performing aligned shuffling with seed: {args.seed}")
    indices = np.arange(num_windows)
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    # Apply the shuffled indices to all window arrays and labels
    shuffled_client_data = {}
    for stage in range(1, 7):
        shuffled_client_data[stage] = windowed_client_data[stage][indices]
    shuffled_labels = windowed_labels[indices]

    # 8. Train-Validation-Test split
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio < 0:
        logger.error(f"Sum of val_ratio ({val_ratio}) and test_ratio ({test_ratio}) exceeds 1.0!")
        return

    logger.info(f"Splitting dataset into train-validation-test sets ({train_ratio:.2f}:{val_ratio:.2f}:{test_ratio:.2f})")
    train_end = int(num_windows * train_ratio)
    val_end = int(num_windows * (train_ratio + val_ratio))

    train_client_data = {}
    val_client_data = {}
    test_client_data = {}

    for stage in range(1, 7):
        train_client_data[stage] = shuffled_client_data[stage][:train_end]
        val_client_data[stage] = shuffled_client_data[stage][train_end:val_end]
        test_client_data[stage] = shuffled_client_data[stage][val_end:]

    train_labels = shuffled_labels[:train_end]
    val_labels = shuffled_labels[train_end:val_end]
    test_labels = shuffled_labels[val_end:]

    logger.info(f"Split sizes: Train={train_end}, Validation={val_end - train_end}, Test={num_windows - val_end}")

    # 9. Save all results as numpy arrays
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save labels in root
    np.save(os.path.join(args.output_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(args.output_dir, "val_labels.npy"), val_labels)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), test_labels)
    logger.info("Saved labels to output directory root")

    # Create subdirectories for splits
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "validation")
    test_dir = os.path.join(args.output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for stage in range(1, 7):
        # Force cast to 32-bit floats before saving.
        # This halves disk space, halves RAM usage, and prevents MPS crashes.
        np.save(os.path.join(train_dir, f"client_{stage}.npy"), train_client_data[stage].astype(np.float32))
        np.save(os.path.join(val_dir, f"client_{stage}.npy"), val_client_data[stage].astype(np.float32))
        np.save(os.path.join(test_dir, f"client_{stage}.npy"), test_client_data[stage].astype(np.float32))
        logger.info(f"Saved Client {stage} train/validation/test arrays as float32")

    logger.info("All preprocessing tasks successfully completed!")


if __name__ == "__main__":
    main()
