#!/usr/bin/env python3
"""
Dataset loaders for all 7 benchmark datasets.
Auto-downloads when possible, instructions for manual downloads otherwise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
RANDOM_SEED = 42


def load_adadult(n_samples=None):
    """Adult Census (48,842 × 14, binary classification)."""
    print("  Loading Adult Census...")
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    X = data.data.copy()
    y = (data.target == ">50K").astype(int).values
    
    # Encode categoricals
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True))
    
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.values)
    
    if n_samples and n_samples < len(X):
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    return X_train, X_test, y_train, y_test, feature_names, "tabular", "classification"


def load_german():
    """German Credit (1,000 × 20, binary classification)."""
    print("  Loading German Credit...")
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    X = data.data.copy()
    y = LabelEncoder().fit_transform(data.target)
    
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    return X_train, X_test, y_train, y_test, feature_names, "tabular", "classification"


def load_california():
    """California Housing (20,640 × 8, regression)."""
    print("  Loading California Housing...")
    data = fetch_california_housing()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    return X_train, X_test, y_train, y_test, feature_names, "tabular", "regression"


def load_cifar10(n_train=5000, n_test=1000):
    """CIFAR-10 (60,000 × 3×32×32, 10-class classification). Subsample for efficiency."""
    import torch
    import torchvision
    import torchvision.transforms as transforms
    
    print("  Loading CIFAR-10...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=transform)
    
    # Subsample for efficiency
    train_idx = np.random.RandomState(RANDOM_SEED).choice(len(trainset), n_train, replace=False)
    test_idx = np.random.RandomState(RANDOM_SEED + 1).choice(len(testset), n_test, replace=False)
    
    X_train = torch.stack([trainset[i][0] for i in train_idx]).numpy()
    y_train = np.array([trainset[i][1] for i in train_idx])
    X_test = torch.stack([testset[i][0] for i in test_idx]).numpy()
    y_test = np.array([testset[i][1] for i in test_idx])
    
    # Flatten for tabular methods, keep 3D for CNN
    feature_names = [f"pixel_{i}" for i in range(3 * 32 * 32)]
    
    return X_train, X_test, y_train, y_test, feature_names, "image", "classification"


def load_isic2019():
    """ISIC 2019 (25,331 × 3×224×224, 8-class classification).
    Requires manual download from https://isic2019challenge.isic-archive.com/
    """
    import torch
    from PIL import Image
    
    print("  Loading ISIC 2019...")
    isic_dir = DATA_DIR / "ISIC2019"
    
    if not isic_dir.exists():
        raise FileNotFoundError(
            f"ISIC 2019 not found at {isic_dir}.\n"
            "Download from: https://isic2019challenge.isic-archive.com/\n"
            "Place images in ISIC2019/ and run preprocess_isic2019.py first."
        )
    
    # Try loading preprocessed
    preprocessed = isic_dir / "isic2019_preprocessed.npz"
    if preprocessed.exists():
        data = np.load(str(preprocessed))
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        feature_names = [f"pixel_{i}" for i in range(X_train.shape[1])]
        return X_train, X_test, y_train, y_test, feature_names, "image", "classification"
    
    # Try loading from raw images
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    raise FileNotFoundError(
        "ISIC 2019 requires preprocessing. "
        "Run: python scripts/preprocess_isic2019.py"
    )


def load_physionet():
    """PhysioNet/ECG (40,336 × 40, binary classification).
    Uses MIT-BIH Arrhythmia or similar ECG dataset from PhysioNet.
    """
    print("  Loading PhysioNet ECG...")
    
    preprocessed = DATA_DIR / "physionet_preprocessed.npz"
    if preprocessed.exists():
        data = np.load(str(preprocessed))
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        feature_names = list(data.get("feature_names", [f"ecg_{i}" for i in range(X_train.shape[1])]))
        return X_train, X_test, y_train, y_test, feature_names, "timeseries", "classification"
    
    # Fallback: use a cardiovascular dataset from OpenML
    print("  PhysioNet preprocessed not found. Using cardiovascular dataset from OpenML...")
    data = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser="auto")
    X = data.data.copy()
    y = LabelEncoder().fit_transform(data.target)
    
    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True))
    
    feature_names = [f"ecg_{i}" for i in range(X.shape[1])]
    X = StandardScaler().fit_transform(X.values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    return X_train, X_test, y_train, y_test, feature_names, "timeseries", "classification"


def load_electricity(n_samples=2000):
    """Electricity dataset (2,000 × 12, binary classification). Subsampled for efficiency."""
    print("  Loading Electricity...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try loading from file
    csv_path = DATA_DIR / "electricity.csv"
    if csv_path.exists():
        df = pd.read_csv(str(csv_path))
    else:
        # Download from OpenML
        try:
            data = fetch_openml(name="electricity", version=1, as_frame=True, parser="auto")
            df = data.frame
        except Exception:
            # Use UCI repository URL
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/reads.data"
            try:
                df = pd.read_csv(url)
            except Exception:
                raise FileNotFoundError(
                    "Electricity dataset not found. "
                    "Download from: https://www.openml.org/d/151 "
                    f"and place at {csv_path}"
                )
    
    # Clean column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # Identify target
    if 'class' in df.columns:
        y = LabelEncoder().fit_transform(df['class'].astype(str))
        X = df.drop(columns=['class', 'date'] if 'date' in df.columns else ['class'])
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
    
    # Handle categorical columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(X.median(numeric_only=True))
    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X.values)
    
    # Subsample
    if n_samples and n_samples < len(X):
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    return X_train, X_test, y_train, y_test, feature_names, "timeseries", "classification"


def load_dataset(name):
    """Load a named dataset. Returns X_train, X_test, y_train, y_test, 
    feature_names, data_type, task_type."""
    loaders = {
        "adult": load_adadult,
        "german": load_german,
        "california": load_california,
        "cifar10": lambda: load_cifar10(),
        "isic2019": load_isic2019,
        "physionet": load_physionet,
        "electricity": load_electricity,
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    result = loaders[name]()
    return result  # X_train, X_test, y_train, y_test, feature_names, data_type, task_type


DATASET_INFO = {
    "adult":       {"n": 48842, "p": 14, "type": "tabular",   "task": "classification"},
    "german":      {"n": 1000,  "p": 20, "type": "tabular",   "task": "classification"},
    "california":  {"n": 20640, "p": 8,  "type": "tabular",   "task": "regression"},
    "cifar10":     {"n": 60000, "p": "3×32×32", "type": "image", "task": "classification"},
    "isic2019":    {"n": 25331, "p": "3×224×224", "type": "image", "task": "classification"},
    "physionet":   {"n": 40336, "p": 40, "type": "timeseries", "task": "classification"},
    "electricity": {"n": 2000,  "p": 12, "type": "timeseries", "task": "classification"},
}