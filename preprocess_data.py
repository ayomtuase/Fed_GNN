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

        logger.info("Merging datasets (Normal first, Attack after)...")
        df = pd.concat([df_normal, df_attack], ignore_index=True)
        logger.info(f"Merged dataset total records: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to load or merge datasets: {e}")
        return

    # 1. Drop the first 6 hours of the merged dataset (21600 rows)
    logger.info("Dropping the first 6 hours of the merged dataset (21600 rows)")
    df = df.iloc[21600:].reset_index(drop=True)

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

    # Fill NaN values with column means or 0
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    # Phase 1: Macro-Chunk Definition and Stratification
    chunk_size = args.chunk_size
    N_total = len(df)
    num_chunks = N_total // chunk_size
    logger.info(f"Slicing dataframe and labels into {num_chunks} chunks of size {chunk_size} (dropping {N_total % chunk_size} remaining rows)")

    chunks = []
    chunk_labels = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        chunk_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        chunk_lbl = labels[start_idx:end_idx]
        
        # Label the chunk: 1 if containing any anomaly (sum > 0), else 0
        chunk_level_label = 1 if np.sum(chunk_lbl) > 0 else 0
        
        chunks.append(chunk_df)
        chunk_labels.append(chunk_level_label)

    chunk_labels = np.array(chunk_labels)
    chunk_indices = np.arange(num_chunks)

    # Stratified Splitting
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    
    # First split: Train_Val vs Test
    train_val_indices, test_indices = train_test_split(
        chunk_indices,
        test_size=test_ratio,
        stratify=chunk_labels,
        random_state=args.seed
    )
    
    # Adjust val_ratio to be relative to the remaining train_val size
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_val_labels = chunk_labels[train_val_indices]
    
    # Second split: Train vs Val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_ratio,
        stratify=train_val_labels,
        random_state=args.seed
    )

    train_chunks = [chunks[idx] for idx in train_indices]
    train_chunk_labels = [labels[idx * chunk_size : (idx + 1) * chunk_size] for idx in train_indices]

    val_chunks = [chunks[idx] for idx in val_indices]
    val_chunk_labels = [labels[idx * chunk_size : (idx + 1) * chunk_size] for idx in val_indices]

    test_chunks = [chunks[idx] for idx in test_indices]
    test_chunk_labels = [labels[idx * chunk_size : (idx + 1) * chunk_size] for idx in test_indices]

    train_normal = sum(1 for idx in train_indices if chunk_labels[idx] == 0)
    train_attack = sum(1 for idx in train_indices if chunk_labels[idx] == 1)
    val_normal = sum(1 for idx in val_indices if chunk_labels[idx] == 0)
    val_attack = sum(1 for idx in val_indices if chunk_labels[idx] == 1)
    test_normal = sum(1 for idx in test_indices if chunk_labels[idx] == 0)
    test_attack = sum(1 for idx in test_indices if chunk_labels[idx] == 1)
    logger.info("Chunk label distribution:")
    logger.info(f"  Train: Normal={train_normal}, Attack={train_attack}")
    logger.info(f"  Val:   Normal={val_normal}, Attack={val_attack}")
    logger.info(f"  Test:  Normal={test_normal}, Attack={test_attack}")

    # Phase 2: Isolated Scaling (The Anti-Leakage Step)
    logger.info("Fitting MinMaxScaler on concatenated Train chunks only")
    df_train_concat = pd.concat(train_chunks, ignore_index=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train_concat.values)

    def transform_chunks(chunks_list):
        transformed = []
        for chunk in chunks_list:
            scaled_vals = scaler.transform(chunk.values)
            transformed.append(pd.DataFrame(scaled_vals, columns=chunk.columns))
        return transformed

    logger.info("Applying scaler to Train, Validation, and Test chunks individually")
    scaled_train_chunks = transform_chunks(train_chunks)
    scaled_val_chunks = transform_chunks(val_chunks)
    scaled_test_chunks = transform_chunks(test_chunks)

    # Phase 3: Client Splitting by Stages and Isolated Downsampling
    # Identify stage columns based on features starting with digits 1-6
    stage_cols = {stage: [] for stage in range(1, 7)}
    for col in df.columns:
        match = re.search(r'\d+', col)
        if match:
            numeric_part = match.group()
            stage = int(numeric_part[0])
            if 1 <= stage <= 6:
                stage_cols[stage].append(col)
            else:
                logger.warning(f"Feature column '{col}' has numeric part starting with digit {stage}, outside 1-6 range. Skipping.")
        else:
            logger.warning(f"Feature column '{col}' does not contain a numeric part. Skipping.")

    for stage in range(1, 7):
        if not stage_cols[stage]:
            logger.error(f"No features found for Stage {stage}!")
            return

    def process_and_downsample_chunks(chunks_list, chunk_labels_list, downsample_factor):
        processed_chunks_features = []
        processed_chunks_labels = []
        for chunk, lbl in zip(chunks_list, chunk_labels_list):
            features_by_stage = {}
            for stage in range(1, 7):
                features_by_stage[stage] = chunk[stage_cols[stage]].values

            chunk_len = len(lbl)
            downsampled_len = chunk_len // downsample_factor

            downsampled_features = {}
            for stage in range(1, 7):
                feat = features_by_stage[stage]
                feat_trimmed = feat[:downsampled_len * downsample_factor]
                feat_reshaped = feat_trimmed.reshape(downsampled_len, downsample_factor, -1)
                downsampled_features[stage] = feat_reshaped.mean(axis=1)

            labels_trimmed = lbl[:downsampled_len * downsample_factor]
            labels_reshaped = labels_trimmed.reshape(downsampled_len, downsample_factor)
            downsampled_labels = (labels_reshaped.sum(axis=1) > 0).astype(int)

            processed_chunks_features.append(downsampled_features)
            processed_chunks_labels.append(downsampled_labels)
        return processed_chunks_features, processed_chunks_labels

    logger.info("Performing isolated downsampling within chunks")
    train_downsampled_feats, train_downsampled_labels = process_and_downsample_chunks(
        scaled_train_chunks, train_chunk_labels, args.downsample_factor
    )
    val_downsampled_feats, val_downsampled_labels = process_and_downsample_chunks(
        scaled_val_chunks, val_chunk_labels, args.downsample_factor
    )
    test_downsampled_feats, test_downsampled_labels = process_and_downsample_chunks(
        scaled_test_chunks, test_chunk_labels, args.downsample_factor
    )

    # Phase 4: Windowing, Safe Shuffling, and Stage-wise Saving
    def window_chunks(downsampled_features_list, downsampled_labels_list, window_size):
        windowed_chunks_features = []
        windowed_chunks_labels = []
        for feat_dict, lbl in zip(downsampled_features_list, downsampled_labels_list):
            downsampled_len = len(lbl)
            if downsampled_len < window_size:
                raise ValueError(f"Downsampled chunk length {downsampled_len} is less than window size {window_size}")

            num_windows = downsampled_len - window_size + 1
            chunk_windowed_feats = {}
            for stage in range(1, 7):
                feat = feat_dict[stage]
                windows = []
                for i in range(num_windows):
                    windows.append(feat[i : i + window_size].T)
                chunk_windowed_feats[stage] = np.array(windows)

            chunk_windowed_labels = lbl[window_size - 1:]
            windowed_chunks_features.append(chunk_windowed_feats)
            windowed_chunks_labels.append(chunk_windowed_labels)
        return windowed_chunks_features, windowed_chunks_labels

    logger.info("Performing internal windowing within chunks")
    train_windowed_feats, train_windowed_labels = window_chunks(
        train_downsampled_feats, train_downsampled_labels, args.window_size
    )
    val_windowed_feats, val_windowed_labels = window_chunks(
        val_downsampled_feats, val_downsampled_labels, args.window_size
    )
    test_windowed_feats, test_windowed_labels = window_chunks(
        test_downsampled_feats, test_downsampled_labels, args.window_size
    )

    logger.info("Recombining windowed chunks by stage")
    master_train_features = {}
    for stage in range(1, 7):
        master_train_features[stage] = np.concatenate(
            [c[stage] for c in train_windowed_feats], axis=0
        )
    master_train_labels = np.concatenate(train_windowed_labels, axis=0)

    master_val_features = {}
    for stage in range(1, 7):
        master_val_features[stage] = np.concatenate(
            [c[stage] for c in val_windowed_feats], axis=0
        )
    master_val_labels = np.concatenate(val_windowed_labels, axis=0)

    master_test_features = {}
    for stage in range(1, 7):
        master_test_features[stage] = np.concatenate(
            [c[stage] for c in test_windowed_feats], axis=0
        )
    master_test_labels = np.concatenate(test_windowed_labels, axis=0)

    # Aligned Selective Shuffling
    logger.info(f"Performing aligned selective shuffling on Train split only with seed: {args.seed}")
    num_train_windows = len(master_train_labels)
    train_indices = np.arange(num_train_windows)
    np.random.seed(args.seed)
    np.random.shuffle(train_indices)

    final_train_features = {}
    for stage in range(1, 7):
        final_train_features[stage] = master_train_features[stage][train_indices]
    final_train_labels = master_train_labels[train_indices]

    final_val_features = master_val_features
    final_val_labels = master_val_labels

    final_test_features = master_test_features
    final_test_labels = master_test_labels

    logger.info("Final dataset shapes:")
    logger.info(f"  Train labels shape: {final_train_labels.shape}")
    logger.info(f"  Val labels shape:   {final_val_labels.shape}")
    logger.info(f"  Test labels shape:  {final_test_labels.shape}")
    for stage in range(1, 7):
        logger.info(
            f"  Stage {stage} features: Train={final_train_features[stage].shape}, "
            f"Val={final_val_features[stage].shape}, Test={final_test_features[stage].shape}"
        )

    # Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "train_labels.npy"), final_train_labels)
    np.save(os.path.join(args.output_dir, "val_labels.npy"), final_val_labels)
    np.save(os.path.join(args.output_dir, "test_labels.npy"), final_test_labels)
    logger.info("Saved labels to output directory root")

    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "validation")
    test_dir = os.path.join(args.output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for stage in range(1, 7):
        np.save(os.path.join(train_dir, f"client_{stage}.npy"), final_train_features[stage].astype(np.float32))
        np.save(os.path.join(val_dir, f"client_{stage}.npy"), final_val_features[stage].astype(np.float32))
        np.save(os.path.join(test_dir, f"client_{stage}.npy"), final_test_features[stage].astype(np.float32))
        logger.info(f"Saved Client {stage} train/validation/test arrays as float32")

    logger.info("All preprocessing tasks successfully completed!")


if __name__ == "__main__":
    main()
