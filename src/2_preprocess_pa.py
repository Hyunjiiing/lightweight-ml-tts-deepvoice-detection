import os
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------------
# PA 전처리 스크립트
# 입력 구조 (예):
#   data_pa/train/real
#   data_pa/train/fake
#   data_pa/dev/real
#   data_pa/dev/fake
# 출력 구조:
#   data_pa/processed/train/audio/*.npy
#   data_pa/processed/train/label.csv
#   data_pa/processed/dev/audio/*.npy
#   data_pa/processed/dev/label.csv
# -------------------------------

SR = 16000
FIX_SEC = 3.0
FIX_LEN = int(SR * FIX_SEC)

# 루트 디렉터리 이름 (요청대로 `data_pa` 사용)
DATA_ROOT = "data_pa"
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

SPLITS = ["train", "dev"]
CLASSES = [("real", 0), ("fake", 1)]


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def preprocess_split(split_name: str):
    """주어진 split(train/dev)을 전처리하고 결과를 저장합니다.

    - 원본: DATA_ROOT/{split}/{real,fake}
    - 출력 오디오(.npy): PROCESSED_DIR/{split}/audio/*.npy
    - 출력 라벨 CSV:      PROCESSED_DIR/{split}/label.csv
    """
    split_root = Path(DATA_ROOT) / split_name
    out_audio_dir = Path(PROCESSED_DIR) / split_name / "audio"
    out_csv_path = Path(PROCESSED_DIR) / split_name / "label.csv"

    _ensure_dir(out_audio_dir)

    rows = []  # (filename, label)
    processed_count = 0
    errors = []

    for class_name, label in CLASSES:
        src_dir = split_root / class_name
        if not src_dir.exists():
            print(f"⚠️  Skipping missing folder: {src_dir}")
            continue

        file_list = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in (".flac", ".wav")])
        if not file_list:
            print(f"⚠️  No audio files found in: {src_dir}")

        for src_path in tqdm(file_list, desc=f"{split_name}/{class_name}"):
            try:
                # 1) load
                y, sr = librosa.load(str(src_path), sr=SR, mono=True)

                # 2) trim
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)

                # 3) fix length
                y_fixed = librosa.util.fix_length(y_trimmed, size=FIX_LEN)

                # 4) safe npy filename
                base = src_path.stem  # os.path.splitext equivalent via pathlib
                dest_name = base + ".npy"
                dest_path = out_audio_dir / dest_name

                # 5) save
                np.save(str(dest_path), y_fixed)

                # 6) record label — filename should be original filename (with ext)
                rows.append((src_path.name, label))
                processed_count += 1

            except Exception as e:
                errors.append((str(src_path), str(e)))

    # save CSV
    if rows:
        df = pd.DataFrame(rows, columns=["filename", "label"])
        df.to_csv(str(out_csv_path), index=False)
    else:
        # ensure empty CSV exists for consistency
        df = pd.DataFrame(columns=["filename", "label"])
        df.to_csv(str(out_csv_path), index=False)

    # 요약 출력
    print(f"\n✅ Done preprocessing split='{split_name}'")
    print(f"  - Processed files: {processed_count}")
    print(f"  - Audio .npy saved to: {out_audio_dir}")
    print(f"  - Label CSV saved to: {out_csv_path}")
    if errors:
        print(f"  - Errors: {len(errors)} (showing up to 10)")
        for p, err in errors[:10]:
            print(f"     {p}: {err}")


def main():
    print("📌 Preprocessing PA dataset")
    print(f"Input root: {DATA_ROOT}")
    print(f"Output processed root: {PROCESSED_DIR}\n")

    for split in SPLITS:
        preprocess_split(split)


if __name__ == "__main__":
    main()
