import os
import random
import shutil

# -------------------------------
# PA 프로토콜용 분할 스크립트
# - data/train/real
# - data/train/fake
# - data/dev/real
# - data/dev/fake
# -------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PA_ROOT = os.path.join(ROOT, "PA")
TRAIN_PROTO = os.path.join(PA_ROOT, "ASVspoof2019_PA_cm_protocols", "ASVspoof2019.PA.cm.train.trn.txt")
DEV_PROTO = os.path.join(PA_ROOT, "ASVspoof2019_PA_cm_protocols", "ASVspoof2019.PA.cm.dev.trl.txt")
TRAIN_AUDIO = os.path.join(PA_ROOT, "ASVspoof2019_PA_train", "flac")
DEV_AUDIO = os.path.join(PA_ROOT, "ASVspoof2019_PA_dev", "flac")

# 출력 디렉터리 (data_pa로 변경)
OUT_TRAIN_REAL = os.path.join(ROOT, "data_pa", "train", "real")
OUT_TRAIN_FAKE = os.path.join(ROOT, "data_pa", "train", "fake")
OUT_DEV_REAL = os.path.join(ROOT, "data_pa", "dev", "real")
OUT_DEV_FAKE = os.path.join(ROOT, "data_pa", "dev", "fake")

# 파라미터
N_TRAIN_PER_CLASS = 500
N_DEV_PER_CLASS = 150
SEED = 42
random.seed(SEED)


def parse_protocol(path):
    """protocol 파일에서 (파일명, 'real'|'fake') 리스트 반환

    PA 프로토콜 형식 예:
        PA_0079 PA_T_0000001 aaa - bonafide
    두번째 토큰이 파일 아이디(예: PA_T_0000001), 마지막 토큰이 라벨(bonafide/spoof)
    """
    pairs = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Protocol file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            file_id = parts[1]
            fn = file_id + ".flac"
            tag = parts[-1].lower()
            if tag == "bonafide":
                label = "real"
            elif tag == "spoof":
                label = "fake"
            else:
                # 보수적으로 라인 전체에서 단어 검사
                lower = line.lower()
                if "bonafide" in lower:
                    label = "real"
                elif "spoof" in lower:
                    label = "fake"
                else:
                    label = None

            if label:
                pairs.append((fn, label))

    return pairs


def sample_and_copy(pairs, audio_dir, out_real_dir, out_fake_dir, n_per_class):
    """pairs에서 real/fake 각각 n_per_class개 샘플링하여 out 디렉터리로 복사

    반환값: (n_real_copied, n_fake_copied, missing_files_list)
    """
    os.makedirs(out_real_dir, exist_ok=True)
    os.makedirs(out_fake_dir, exist_ok=True)

    real_files = [f for f, l in pairs if l == "real"]
    fake_files = [f for f, l in pairs if l == "fake"]

    pick_real = random.sample(real_files, min(n_per_class, len(real_files))) if real_files else []
    pick_fake = random.sample(fake_files, min(n_per_class, len(fake_files))) if fake_files else []

    missing = []
    n_real = 0
    n_fake = 0

    for fn in pick_real:
        src = os.path.join(audio_dir, fn)
        dst = os.path.join(out_real_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            n_real += 1
        else:
            missing.append(src)

    for fn in pick_fake:
        src = os.path.join(audio_dir, fn)
        dst = os.path.join(out_fake_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            n_fake += 1
        else:
            missing.append(src)

    print(f"✅ Copied {n_real} real and {n_fake} fake from {os.path.basename(audio_dir)} to {os.path.dirname(out_real_dir)} / {os.path.dirname(out_fake_dir)}")
    return n_real, n_fake, missing


if __name__ == "__main__":
    print("📘 Parsing train protocol and sampling into data/train/...")
    train_pairs = parse_protocol(TRAIN_PROTO)
    ntr, nte, miss_train = sample_and_copy(train_pairs, TRAIN_AUDIO, OUT_TRAIN_REAL, OUT_TRAIN_FAKE, N_TRAIN_PER_CLASS)

    print("📗 Parsing dev protocol and sampling into data/dev/...")
    dev_pairs = parse_protocol(DEV_PROTO)
    ndtr, ndte, miss_dev = sample_and_copy(dev_pairs, DEV_AUDIO, OUT_DEV_REAL, OUT_DEV_FAKE, N_DEV_PER_CLASS)

    print("\n🎯 Summary:")
    print(f" train -> real: {ntr}, fake: {nte}")
    print(f" dev   -> real: {ndtr}, fake: {ndte}")
    if miss_train or miss_dev:
        print("\n⚠️ Missing files encountered:")
        for m in miss_train[:50]:
            print("  ", m)
        for m in miss_dev[:50]:
            print("  ", m)
    else:
        print(" No missing files reported.")
