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
from sklearn.preprocessing import StandardScaler

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
        "--normal_file",
        type=str,
        default="data/SWaT_Dataset_Normal_v0.xlsx",
        help="Path to the raw normal Excel dataset",
    )
    parser.add_argument(
        "--attack_file",
        type=str,
        default="data/SWaT_Dataset_Attack_v0.xlsx",
        help="Path to the raw attack Excel dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed_data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=7200, help="Chunk size for macro-chunking"
    )
    parser.add_argument(
        "--downsample_factor", type=int, default=10, help="Downsampling factor"
    )
    parser.add_argument(
        "--window_size", type=int, default=120, help="Window size for feature extraction"
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

    logger.info("Starting SWAT preprocessing")
    logger.info(f"  Normal file: {args.normal_file}")
    logger.info(f"  Attack file: {args.attack_file}")

    if not os.path.exists(args.normal_file):
        logger.error(f"Could not find the normal file: {args.normal_file}")
        return
    if not os.path.exists(args.attack_file):
        logger.error(f"Could not find the attack file: {args.attack_file}")
        return

    try:
        # Load the normal and attack datasets
        logger.info("Loading Normal dataset (this may take a moment)...")
        df_normal = pd.read_excel(args.normal_file, header=1)
        logger.info(f"Successfully loaded Normal dataset with {len(df_normal)} records")

        logger.info("Loading Attack dataset (this may take a moment)...")
        df_attack = pd.read_excel(args.attack_file, header=1)
        logger.info(f"Successfully loaded Attack dataset with {len(df_attack)} records")

        # Strip column whitespaces before merging to ensure matching column names
        df_normal.columns = [col.strip() for col in df_normal.columns]
        df_attack.columns = [col.strip() for col in df_attack.columns]

        # 1. Drop the first 6 hours of the normal dataset (21600 rows)
        logger.info("Dropping the first 6 hours of the normal dataset (21600 rows)")
        df_normal = df_normal.iloc[21600:].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # Remove the timestamp column
    if "Timestamp" in df_normal.columns:
        df_normal = df_normal.drop(columns=["Timestamp"])
    if "Timestamp" in df_attack.columns:
        df_attack = df_attack.drop(columns=["Timestamp"])

    # 2. Separate out the labels
    if "Normal/Attack" in df_normal.columns:
        # Map Normal/Attack to binary labels: Attack -> 1, Normal -> 0
        normal_labels = df_normal["Normal/Attack"].astype(str).str.strip().map({"Attack": 1, "Normal": 0})
        normal_labels = normal_labels.fillna(0).astype(int).values
        df_normal = df_normal.drop(columns=["Normal/Attack"])
    else:
        logger.error("Label column 'Normal/Attack' not found in normal dataset")
        return

    if "Normal/Attack" in df_attack.columns:
        attack_labels = df_attack["Normal/Attack"].astype(str).str.strip().map({"Attack": 1, "Normal": 0})
        attack_labels = attack_labels.fillna(0).astype(int).values
        df_attack = df_attack.drop(columns=["Normal/Attack"])
    else:
        logger.error("Label column 'Normal/Attack' not found in attack dataset")
        return

    # Drop any other non-numeric feature columns
    non_numeric_cols_normal = [col for col in df_normal.columns if not pd.api.types.is_numeric_dtype(df_normal[col])]
    if non_numeric_cols_normal:
        logger.info(f"Dropping non-numeric feature columns from normal dataset: {non_numeric_cols_normal}")
        df_normal = df_normal.drop(columns=non_numeric_cols_normal)

    non_numeric_cols_attack = [col for col in df_attack.columns if not pd.api.types.is_numeric_dtype(df_attack[col])]
    if non_numeric_cols_attack:
        logger.info(f"Dropping non-numeric feature columns from attack dataset: {non_numeric_cols_attack}")
        df_attack = df_attack.drop(columns=non_numeric_cols_attack)

    # Align columns between normal and attack datasets
    common_cols = [col for col in df_normal.columns if col in df_attack.columns]
    df_normal = df_normal[common_cols]
    df_attack = df_attack[common_cols]
    cols = list(df_normal.columns)

    # Phase 1: Downsampling raw df_normal and df_attack
    logger.info(f"Downsampling normal dataset with factor {args.downsample_factor} using median/majority vote")
    
    def process_and_downsample_raw(df_raw, labels_raw, downsample_factor):
        split_len = len(labels_raw)
        downsampled_len = split_len // downsample_factor
        
        # 1. Downsample features using median
        raw_vals = df_raw.values[:downsampled_len * downsample_factor]
        reshaped_feats = raw_vals.reshape(downsampled_len, downsample_factor, -1)
        downsampled_feats = np.median(reshaped_feats, axis=1)
        
        # 2. Downsample labels using majority vote (mode)
        raw_labels = labels_raw[:downsampled_len * downsample_factor]
        reshaped_labels = raw_labels.reshape(downsampled_len, downsample_factor)
        downsampled_labels = (reshaped_labels.sum(axis=1) >= (downsample_factor / 2)).astype(int)
        
        df_downsampled = pd.DataFrame(downsampled_feats, columns=df_raw.columns)
        return df_downsampled, downsampled_labels

    df_normal_downsampled, normal_labels_downsampled = process_and_downsample_raw(
        df_normal, normal_labels, args.downsample_factor
    )
    df_attack_downsampled, attack_labels_downsampled = process_and_downsample_raw(
        df_attack, attack_labels, args.downsample_factor
    )

    del df_normal
    del df_attack
    import gc
    gc.collect()

    # Phase 2: Split definitions (on downsampled data)
    train_len = int(0.80 * len(df_normal_downsampled))
    train_df = df_normal_downsampled.iloc[:train_len].reset_index(drop=True)
    train_labels = normal_labels_downsampled[:train_len]

    val_df = df_normal_downsampled.iloc[train_len:].reset_index(drop=True)
    val_labels = normal_labels_downsampled[train_len:]

    test_df = df_attack_downsampled.reset_index(drop=True)
    test_labels = attack_labels_downsampled

    # Fix Data Leakage: Impute NaN values using column means from Train set only
    logger.info("Imputing NaN values using column means from Train set only (avoiding data leakage)")
    train_mean = train_df.mean(numeric_only=True)
    train_df = train_df.fillna(train_mean).fillna(0)
    val_df = val_df.fillna(train_mean).fillna(0)
    test_df = test_df.fillna(train_mean).fillna(0)

    # Phase 3: Isolated Scaling (Fitted on DOWNSAMPLED train data)
    logger.info("Fitting StandardScaler on downsampled Train features only")
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    logger.info("Applying scaler to Train, Validation, and Test features individually")
    scaled_train_df = pd.DataFrame(scaler.transform(train_df.values), columns=cols)
    scaled_val_df = pd.DataFrame(scaler.transform(val_df.values), columns=cols)
    scaled_test_df = pd.DataFrame(scaler.transform(test_df.values), columns=cols)

    del train_df
    del val_df
    del test_df
    gc.collect()

    # Phase 4: Client Splitting by Stages
    stage_cols = {stage: [] for stage in range(1, 7)}
    for col in cols:
        match = re.match(r'^[A-Za-z_]*([1-6])', col)
        if match:
            stage = int(match.group(1))
            stage_cols[stage].append(col)
        else:
            logger.warning(f"Feature column '{col}' does not match expected pattern or is outside 1-6 range. Skipping.")

    for stage in range(1, 7):
        if not stage_cols[stage]:
            logger.error(f"No features found for Stage {stage}!")
            return

    # Save StandardScaler and Column Metadata
    import pickle
    os.makedirs(args.output_dir, exist_ok=True)
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    concat_cols = []
    for stage in range(1, 7):
        concat_cols.extend(stage_cols[stage])

    scaler_data = {
        "scaler": scaler,
        "columns": cols,
        "concat_cols": concat_cols,
        "client_columns": stage_cols
    }
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)
    logger.info(f"Saved StandardScaler and column metadata to {scaler_path}")

    # Split scaled dataframes by client stages
    def get_stage_features(df_scaled):
        features_by_stage = {}
        for stage in range(1, 7):
            features_by_stage[stage] = df_scaled[stage_cols[stage]].values
        return features_by_stage

    train_downsampled_feats = get_stage_features(scaled_train_df)
    val_downsampled_feats = get_stage_features(scaled_val_df)
    test_downsampled_feats = get_stage_features(scaled_test_df)

    del scaled_train_df
    del scaled_val_df
    del scaled_test_df
    gc.collect()

    # --- NEW PHASE 4: Direct-to-Disk Saving of Continuous Downsampled Arrays ---
    def stream_windows_to_disk(downsampled_features, downsampled_labels, out_dir, split_name):
        # Create output directories if they do not exist
        os.makedirs(out_dir, exist_ok=True)
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # 1. Write labels to disk directly
        labels_path = os.path.join(out_dir, f"{split_name}_labels.npy")
        np.save(labels_path, downsampled_labels)
        
        # 2. Write continuous features to disk client by client
        for stage in range(1, 7):
            path = os.path.join(split_dir, f"client_{stage}.npy")
            feat = downsampled_features[stage]
            np.save(path, feat)

        return labels_path

    logger.info("Streaming Train windows directly to disk...")
    stream_windows_to_disk(train_downsampled_feats, train_labels, args.output_dir, "train")

    logger.info("Streaming Validation windows directly to disk...")
    stream_windows_to_disk(val_downsampled_feats, val_labels, args.output_dir, "validation")

    logger.info("Streaming Test windows directly to disk...")
    stream_windows_to_disk(test_downsampled_feats, test_labels, args.output_dir, "test")

    logger.info("All preprocessing tasks successfully completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("preprocess_error.txt", "w") as f:
            traceback.print_exc(file=f)
        logger.error(f"FATAL EXCEPTION in main: {e}")
        raise e
