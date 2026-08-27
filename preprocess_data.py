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
from sklearn.model_selection import train_test_split

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
        "--downsample_factor", type=int, default=1, help="Downsampling factor"
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


# save_split_stage is no longer needed as window streaming writes directly to disk.


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
        len_normal = len(df_normal)

        logger.info("Merging datasets (Normal first, Attack after)...")
        df = pd.concat([df_normal, df_attack], ignore_index=True)
        logger.info(f"Merged dataset total records: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to load or merge datasets: {e}")
        return

    # Clean columns by stripping whitespace (in case of any new merged columns)
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

    # Calculate split boundaries for the timeline split
    # Train: All of 7-day normal + first 50% of attack
    # Val: Next 25% of attack
    # Test: Final 25% of attack
    attack_start = len_normal
    attack_len = len(df) - attack_start
    attack_split1 = attack_start + int(0.50 * attack_len)
    attack_split2 = attack_start + int(0.75 * attack_len)

    # Fix Data Leakage: Fit column means on the training set only (from 0 to attack_split1)
    logger.info("Imputing NaN values using column means from Train set only (avoiding data leakage)")
    train_mean = df.iloc[:attack_split1].mean(numeric_only=True)
    df = df.fillna(train_mean).fillna(0)

    # Phase 1: Split definitions
    train_df = df.iloc[:attack_split1].reset_index(drop=True)
    train_labels = labels[:attack_split1]

    val_df = df.iloc[attack_split1:attack_split2].reset_index(drop=True)
    val_labels = labels[attack_split1:attack_split2]

    test_df = df.iloc[attack_split2:].reset_index(drop=True)
    test_labels = labels[attack_split2:]

    logger.info("Split size details:")
    logger.info(f"  Train:      size={len(train_df)}")
    logger.info(f"  Validation: size={len(val_df)}")
    logger.info(f"  Test:       size={len(test_df)}")

    # Phase 2: Isolated Scaling (The Anti-Leakage Step)
    logger.info("Fitting StandardScaler on Train features only")
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    logger.info("Applying scaler to Train, Validation, and Test features individually")
    scaled_train_df = pd.DataFrame(scaler.transform(train_df.values), columns=train_df.columns)
    scaled_val_df = pd.DataFrame(scaler.transform(val_df.values), columns=val_df.columns)
    scaled_test_df = pd.DataFrame(scaler.transform(test_df.values), columns=test_df.columns)

    # Phase 3: Client Splitting by Stages and Downsampling
    # Identify stage columns based on features starting with digits 1-6 using robust regex
    stage_cols = {stage: [] for stage in range(1, 7)}
    for col in df.columns:
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

    def process_and_downsample(df_split, labels_split, downsample_factor):
        features_by_stage = {}
        for stage in range(1, 7):
            features_by_stage[stage] = df_split[stage_cols[stage]].values

        split_len = len(labels_split)
        downsampled_len = split_len // downsample_factor

        downsampled_features = {}
        for stage in range(1, 7):
            feat = features_by_stage[stage]
            feat_trimmed = feat[:downsampled_len * downsample_factor]
            feat_reshaped = feat_trimmed.reshape(downsampled_len, downsample_factor, -1)
            downsampled_features[stage] = feat_reshaped.mean(axis=1)

        labels_trimmed = labels_split[:downsampled_len * downsample_factor]
        labels_reshaped = labels_trimmed.reshape(downsampled_len, downsample_factor)
        downsampled_labels = (labels_reshaped.sum(axis=1) > 0).astype(int)

        return downsampled_features, downsampled_labels

    logger.info("Performing isolated downsampling")
    train_downsampled_feats, train_downsampled_labels = process_and_downsample(
        scaled_train_df, train_labels, args.downsample_factor
    )
    val_downsampled_feats, val_downsampled_labels = process_and_downsample(
        scaled_val_df, val_labels, args.downsample_factor
    )
    test_downsampled_feats, test_downsampled_labels = process_and_downsample(
        scaled_test_df, test_labels, args.downsample_factor
    )

    # --- NEW PHASE 4: Windowing and Direct-to-Disk Streaming (Anti-OOM) ---
    from numpy.lib.stride_tricks import sliding_window_view

    def stream_windows_to_disk(downsampled_features, downsampled_labels, window_size, out_dir, split_name):
        # 1. Calculate total expected windows to pre-allocate disk space
        total_windows = max(0, len(downsampled_labels) - window_size + 1)
        
        # 2. Pre-allocate memmap file for labels on disk
        labels_path = os.path.join(out_dir, f"{split_name}_labels.npy")
        fp_labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.int64, shape=(total_windows,))
        
        # 3. Pre-allocate memmap files for features on disk
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        fp_feats = {}
        for stage in range(1, 7):
            num_features = downsampled_features[stage].shape[1]
            path = os.path.join(split_dir, f"client_{stage}.npy")
            # Shape is (Total Windows, Window Size, Features) - Transposed standard format
            fp_feats[stage] = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(total_windows, window_size, num_features))
            
        # 4. Stream split to disk directly
        if total_windows > 0:
            # Write labels to disk
            fp_labels[:] = downsampled_labels[window_size - 1:]
            
            # Write features to disk efficiently
            for stage in range(1, 7):
                feat = downsampled_features[stage]
                # sliding_window_view creates the overlaps instantly without blowing up RAM
                view = sliding_window_view(feat, window_shape=window_size, axis=0)
                # Transpose from (T - W + 1, F, W) to (T - W + 1, W, F)
                view_transposed = view.transpose(0, 2, 1)
                fp_feats[stage][:] = view_transposed
            
        # 5. Safely flush memory buffer to physical storage
        fp_labels.flush()
        for stage in range(1, 7):
            fp_feats[stage].flush()
            
        return labels_path

    logger.info("Streaming Train windows directly to disk...")
    stream_windows_to_disk(train_downsampled_feats, train_downsampled_labels, args.window_size, args.output_dir, "train")

    logger.info("Streaming Validation windows directly to disk...")
    stream_windows_to_disk(val_downsampled_feats, val_downsampled_labels, args.window_size, args.output_dir, "validation")

    logger.info("Streaming Test windows directly to disk...")
    stream_windows_to_disk(test_downsampled_feats, test_downsampled_labels, args.window_size, args.output_dir, "test")

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
