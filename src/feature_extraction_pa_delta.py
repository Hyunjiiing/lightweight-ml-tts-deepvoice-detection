import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

# --- Config -----------------------------------------------------------------
DATA_PROCESSED = Path("data_pa") / "processed"
OUT_FEATURE_DIR = Path("data_pa") / "features"
SPLITS = ["train", "dev"]

# audio / mfcc params
SR = 16000
N_MFCC = 20
CENTER = False

# sentinel for missing files
MISSING_FILL_VALUE = -9999.0

# output feature names (120-d)
OUT_SUFFIX = "_features_120.npy"

# ensure output dir
OUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_label_csv(split_dir: Path):
    """Read label CSV from processed split directory. Expects 'filename' and 'label' columns; tolerates variants."""
    csv_path = split_dir / "label.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing label.csv for split at {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names
    cols = [c.lower() for c in df.columns]
    if "filename" in cols and "label" in cols:
        # standard
        df.columns = df.columns.str.lower()
        return df[["filename", "label"]]
    # try common variants
    if len(df.columns) >= 2:
        # assume first two columns are filename,label
        df2 = df.iloc[:, :2].copy()
        df2.columns = ["filename", "label"]
        return df2
    raise ValueError(f"Unrecognized label.csv format: columns={df.columns.tolist()}")


def _npy_path_for_filename(split_dir: Path, filename: str) -> Path:
    """Map a CSV filename entry to the preprocessed .npy path under split/audio/"""
    # Many label CSVs use basename without extension; original preprocess saved as <basename>.npy
    audio_dir = split_dir / "audio"
    # try several heuristics:
    candidate = audio_dir / f"{filename}"
    if candidate.exists():
        return candidate

    # If filename includes an audio extension like .flac or .wav, try the stem
    # (drop extension) because preprocess saved files as <stem>.npy
    stem = Path(filename).stem
    candidate = audio_dir / f"{stem}.npy"
    if candidate.exists():
        return candidate

    # try filename + .npy (handles cases where CSV lists basename without .npy)
    candidate = audio_dir / f"{filename}.npy"
    if candidate.exists():
        return candidate

    # last resort: try Path(filename).name (no dir components)
    candidate = audio_dir / Path(filename).name
    if candidate.exists():
        return candidate

    # fallback (non-existent) — caller will handle missing
    return audio_dir / f"{stem}.npy"


def compute_feature_from_audio(y: np.ndarray):
    """Compute 120-d feature vector from raw audio array y."""
    # compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, center=CENTER)
    # delta and delta-delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # means and stds across time axis (axis=1)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    delta_mean = delta.mean(axis=1)
    delta_std = delta.std(axis=1)

    delta2_mean = delta2.mean(axis=1)
    delta2_std = delta2.std(axis=1)

    feat_vector = np.concatenate(
        [mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std], axis=0
    )
    # ensure shape is (120,)
    assert feat_vector.shape[0] == N_MFCC * 6
    return feat_vector.astype(np.float32)


def extract_features_for_split(split: str):
    logging.info(f"Processing split='{split}'")
    split_dir = DATA_PROCESSED / split
    df = _read_label_csv(split_dir)

    n_rows = len(df)
    feat_dim = N_MFCC * 6  # 20 * 6 = 120
    features = np.full((n_rows, feat_dim), fill_value=MISSING_FILL_VALUE, dtype=np.float32)
    labels = np.zeros((n_rows,), dtype=np.int64)

    missing_list = []

    for idx, row in enumerate(tqdm(df.itertuples(index=False), total=n_rows, desc=f"Extracting ({split})")):
        # pandas namedtuple fields may be ('filename', 'label') depending on normalization above
        filename = getattr(row, "filename")
        label = getattr(row, "label")
        labels[idx] = int(label)

        npy_path = _npy_path_for_filename(split_dir, filename)
        if not npy_path.exists():
            missing_list.append(str(npy_path))
            # leave sentinel row
            continue

        try:
            y = np.load(str(npy_path))
            # some .npy may contain (audio, sr) tuple — handle common case by detecting shape/dtype
            if isinstance(y, np.ndarray) and y.ndim == 0:
                # scalar object saved — attempt to read object -> not expected; treat as missing
                missing_list.append(str(npy_path))
                continue
            # if y is 2D (e.g., (n_channels, n_samples)), convert to mono by averaging
            if y.ndim > 1:
                y = np.mean(y, axis=0)
            feat = compute_feature_from_audio(y)
            features[idx] = feat
        except Exception as e:
            logging.warning(f"Failed to compute features for {npy_path}: {e}")
            missing_list.append(str(npy_path))
            # leave sentinel

    # save features & labels
    out_feat_path = OUT_FEATURE_DIR / f"{split}{OUT_SUFFIX}"
    out_label_path = OUT_FEATURE_DIR / f"{split}_labels.npy"
    np.save(out_feat_path, features)
    np.save(out_label_path, labels)

    # save missing list if any
    if missing_list:
        missing_path = OUT_FEATURE_DIR / f"missing_{split}.txt"
        with open(missing_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(missing_list))
        logging.warning(f"{len(missing_list)} missing files for split='{split}' -> {missing_path}")
    else:
        logging.info(f"No missing files for split='{split}'")

    logging.info(f"Saved features -> {out_feat_path} (shape={features.shape})")
    logging.info(f"Saved labels   -> {out_label_path} (shape={labels.shape})")


def main():
    for split in SPLITS:
        extract_features_for_split(split)
    logging.info("Feature extraction (120-d with delta, delta-delta) completed.")


if __name__ == "__main__":
    main()
