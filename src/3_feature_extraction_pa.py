"""Feature extraction for ASVspoof2019 PA preprocessed audio.

This script reads `data_pa/processed/{train,dev}/label.csv` and corresponding
`.npy` audio files under `data_pa/processed/{split}/audio/`. For each clip it
computes MFCC (n_mfcc=20, center=False) and reduces the time axis by mean and
std to produce a single feature vector per clip (length 40). Missing `.npy`
files are filled with a sentinel vector (`MISSING_FILL_VALUE`) while preserving
the row order from the CSV. Missing paths are logged to
`data_pa/features/missing_{split}.txt`.
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# Configuration
SR = 16000
N_MFCC = 20

DATA_ROOT = Path("data_pa")
PROCESSED_ROOT = DATA_ROOT / "processed"
OUT_FEATURE_DIR = DATA_ROOT / "features"
SPLITS = ["train", "dev"]

# Sentinel value for missing feature vectors (float32)
MISSING_FILL_VALUE = -9999.0


def _read_label_csv(split: str) -> pd.DataFrame:
    csv_path = PROCESSED_ROOT / split / "label.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found for split='{split}': {csv_path}")
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Label CSV missing required columns: {csv_path}")
    return df


def _npy_path_for_filename(split: str, filename: str) -> Path:
    stem = Path(filename).stem
    return PROCESSED_ROOT / split / "audio" / f"{stem}.npy"


def _compute_mfcc_feature(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, center=False)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_features_for_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    df = _read_label_csv(split)
    audio_dir = PROCESSED_ROOT / split / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found for split='{split}': {audio_dir}")

    features: List[np.ndarray] = []
    labels: List[int] = []
    missing_files: List[str] = []

    print(f"🎧 Extracting MFCC features for split='{split}' ({len(df)} rows)")

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        filename = getattr(row, "filename")
        label = int(getattr(row, "label"))

        npy_path = _npy_path_for_filename(split, filename)

        if not npy_path.exists():
            feat_vector = np.full((N_MFCC * 2,), MISSING_FILL_VALUE, dtype=np.float32)
            features.append(feat_vector)
            labels.append(label)
            missing_files.append(str(npy_path))
            continue

        try:
            audio = np.load(str(npy_path))
        except Exception as exc:
            feat_vector = np.full((N_MFCC * 2,), MISSING_FILL_VALUE, dtype=np.float32)
            features.append(feat_vector)
            labels.append(label)
            missing_files.append(f"{npy_path} (load error: {exc})")
            continue

        feat_vector = _compute_mfcc_feature(audio)
        features.append(feat_vector)
        labels.append(label)

    X = np.vstack(features) if features else np.empty((0, N_MFCC * 2), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    if missing_files:
        OUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        missing_path = OUT_FEATURE_DIR / f"missing_{split}.txt"
        with open(missing_path, "w") as mf:
            for p in missing_files:
                mf.write(p + "\n")
        print(f"⚠️  {len(missing_files)} missing items for split='{split}'. See {missing_path}")

    return X, y


def main() -> None:
    OUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        X, y = extract_features_for_split(split)

        feat_file = OUT_FEATURE_DIR / f"{split}_features.npy"
        label_file = OUT_FEATURE_DIR / f"{split}_labels.npy"

        np.save(str(feat_file), X)
        np.save(str(label_file), y)

        print(f"\n✅ Saved {split} features -> {feat_file} (shape={X.shape})")
        print(f"✅ Saved {split} labels   -> {label_file} (shape={y.shape})\n")


if __name__ == "__main__":
    main()
