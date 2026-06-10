#!/usr/bin/env python3
"""
Preprocessing scripts for datasets requiring manual download.
- ISIC 2019: Skin lesion classification
- PhysioNet: ECG/clinical time-series
"""

import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
RANDOM_SEED = 42


def preprocess_isic2019(image_dir=None, output_dir=None, img_size=(224, 224), 
                         max_samples=None):
    """
    Preprocess ISIC 2019 dataset.
    
    Download instructions:
    1. Register at https://isic2019challenge.isic-archive.com/
    2. Download training images and ground truth CSV
    3. Place in DATA_DIR/ISIC2019/
    
    Alternatively, download from:
    - Images: https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
    - Labels: https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import pandas as pd
    
    if output_dir is None:
        output_dir = DATA_DIR / "ISIC2019"
    if image_dir is None:
        image_dir = DATA_DIR / "ISIC2019" / "ISIC_2019_Training_Input"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load ground truth
    gt_path = DATA_DIR / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {gt_path}\n"
            "Download from: https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
        )
    
    df = pd.read_csv(str(gt_path))
    label_cols = [c for c in df.columns if c != 'image']
    
    # Sample
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=RANDOM_SEED)
    
    X_list = []
    y_list = []
    
    for idx, row in df.iterrows():
        img_path = image_dir / f"{row['image']}.jpg"
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(str(img_path)).convert('RGB')
            tensor = transform(img)
            X_list.append(tensor.numpy().flatten())
            y_list.append(row[label_cols].values.argmax())
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Save
    np.savez_compressed(
        str(output_dir / "isic2019_preprocessed.npz"),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )
    
    print(f"ISIC 2019 preprocessed: {len(X_list)} images")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Saved to: {output_dir / 'isic2019_preprocessed.npz'}")
    
    return X_train, X_test, y_train, y_test


def preprocess_physionet(data_path=None, output_dir=None, seq_length=188):
    """
    Preprocess PhysioNet ECG dataset (MIT-BIH Arrhythmia).
    
    Download instructions:
    1. Download from https://physionet.org/content/mitdb/1.0.0/
    2. Or use wfdb: pip install wfdb
    3. Place in DATA_DIR/physionet/
    
    Alternatively, use the PTB-XL dataset:
    - https://physionet.org/content/ptb-xl/1.0.3/
    """
    if output_dir is None:
        output_dir = DATA_DIR
    if data_path is None:
        data_path = DATA_DIR / "physionet"
    
    try:
        import wfdb
        from wfdb.processing import normalize
    
        print("Processing MIT-BIH Arrhythmia Database...")
        data_path.mkdir(parents=True, exist_ok=True)
        
        records = wfdb.get_record_list('mitdb')
        
        X_list = []
        y_list = []
        
        for record_name in records[:20]:  # Use first 20 records
            try:
                record = wfdb.rdrecord(str(data_path / record_name), pb_dir='mitdb')
                annotation = wfdb.rdann(str(data_path / record_name), 'atr', pb_dir='mitdb')
                
                # Extract beats
                signals = record.p_signal[:, 0]  # Use MLII lead
                beats = wfdb.processing.xqrs_detect(
                    sig=signals, fs=record.fs, verbose=False
                )
                
                for beat_idx in beats:
                    start = max(0, beat_idx - seq_length // 2)
                    end = min(len(signals), beat_idx + seq_length // 2)
                    beat = signals[start:end]
                    
                    # Pad or truncate to fixed length
                    if len(beat) < seq_length:
                        beat = np.pad(beat, (0, seq_length - len(beat)), 'constant')
                    else:
                        beat = beat[:seq_length]
                    
                    # Get label
                    label_idx = np.argmin(np.abs(annotation.sample - beat_idx))
                    label = annotation.symbol[label_idx]
                    # Map to binary: N=normal, other=abnormal
                    y_binary = 0 if label == 'N' else 1
                    
                    X_list.append(beat)
                    y_list.append(y_binary)
            except Exception as e:
                print(f"  Skipping {record_name}: {e}")
                continue
        
        if len(X_list) == 0:
            raise RuntimeError("No ECG beats extracted. Using fallback dataset.")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X = StandardScaler().fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        np.savez_compressed(
            str(output_dir / "physionet_preprocessed.npz"),
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        
        print(f"PhysioNet preprocessed: {len(X_list)} beats")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Saved to: {output_dir / 'physionet_preprocessed.npz'}")
        
        return X_train, X_test, y_train, y_test
    
    except ImportError:
        print("wfdb not installed. Using fallback dataset.")
        print("Install with: pip install wfdb")
        return None


def preprocess_electricity(output_dir=None):
    """
    Preprocess and cache the Electricity dataset.
    Downloads from OpenML if not already cached.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    if output_dir is None:
        output_dir = DATA_DIR
    
    print("Processing Electricity dataset...")
    
    try:
        data = fetch_openml(name="electricity", version=1, as_frame=True, parser="auto")
        df = data.frame
    except Exception:
        # Alternative: download from UCI
        import pandas as pd
        url = "https://www.openml.org/data/download/5414295/electricity.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            raise FileNotFoundError(
                "Cannot download Electricity dataset automatically.\n"
                "Download from: https://www.openml.org/d/151"
            )
    
    # Clean
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    if 'class' in df.columns:
        y = LabelEncoder().fit_transform(df['class'].astype(str))
        X = df.drop(columns=['class', 'date'] if 'date' in df.columns else ['class'])
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
    
    # Handle categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(X.median(numeric_only=True))
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.values)
    
    # Subsample for efficiency
    if len(X) > 20000:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    np.savez_compressed(
        str(output_dir / "electricity_preprocessed.npz"),
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names
    )
    
    print(f"Electricity preprocessed: {len(X_train) + len(X_test)} samples")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test


def download_all_datasets():
    """Download and preprocess all datasets."""
    print("="*60)
    print("DOWNLOADING AND PREPROCESSING ALL DATASETS")
    print("="*60)
    
    # 1. Adult Census (auto-download)
    print("\n[1/7] Adult Census (auto-download)")
    from dataloaders import load_dataset
    try:
        load_dataset("adult")
        print("  OK - Adult Census loaded")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 2. German Credit (auto-download)
    print("\n[2/7] German Credit (auto-download)")
    try:
        load_dataset("german")
        print("  OK - German Credit loaded")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 3. California Housing (auto-download)
    print("\n[3/7] California Housing (auto-download)")
    try:
        load_dataset("california")
        print("  OK - California Housing loaded")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 4. CIFAR-10 (auto-download)
    print("\n[4/7] CIFAR-10 (auto-download)")
    try:
        load_dataset("cifar10")
        print("  OK - CIFAR-10 loaded")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 5. ISIC 2019 (manual)
    print("\n[5/7] ISIC 2019 (manual download)")
    try:
        preprocess_isic2019()
    except FileNotFoundError as e:
        print(f"  MANUAL DOWNLOAD REQUIRED: {e}")
    
    # 6. PhysioNet (semi-auto)
    print("\n[6/7] PhysioNet (semi-auto)")
    try:
        preprocess_physionet()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # 7. Electricity (semi-auto)
    print("\n[7/7] Electricity (semi-auto)")
    try:
        preprocess_electricity()
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n" + "="*60)
    print("DONE. Datasets saved in:", DATA_DIR)
    print("="*60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess benchmark datasets")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to preprocess (isic2019, physionet, electricity, all)")
    args = parser.parse_args()
    
    if args.dataset == "isic2019":
        preprocess_isic2019()
    elif args.dataset == "physionet":
        preprocess_physionet()
    elif args.dataset == "electricity":
        preprocess_electricity()
    elif args.dataset == "all" or args.dataset is None:
        download_all_datasets()