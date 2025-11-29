import os
from pathlib import Path
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm
from typing import Optional, Dict, Any


# --- Configuration ---------------------------------------------------------
DATA_FEATURE_DIR = Path("data_pa") / "features"
MISSING_FILL_VALUE = -9999.0
OUT_MODELS_DIR = Path("models")
OUT_RESULTS_DIR = Path("results")

# Feature file preference: prefer legacy 40-d features (as requested).
# If 40-d not present, fall back to 120-d features if available.
FEATURE_PREFERRED_SUFFIX = "_features.npy"
FEATURE_FALLBACK_SUFFIX = "_features_120.npy"

OUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model selection
MODEL_NAME = os.environ.get("MODEL_NAME", "RF")  # RF | SVM | LR (extendable)

# Training hyperparameters
RF_PARAMS = dict(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1)

# Use tqdm during dev prediction (optional)
USE_TQDM_FOR_EVAL = True


def load_split(split: str):
    """Load features and labels for a split and remove sentinel rows.

    Returns X_clean, y_clean
    """
    # Use 120-d features by default (no fallback). This forces experiments to use the
    # enhanced feature set. If the 120-d file is missing, raise an error so the
    # user regenerates features.
    preferred = DATA_FEATURE_DIR / f"{split}{FEATURE_PREFERRED_SUFFIX}"
    fallback = DATA_FEATURE_DIR / f"{split}{FEATURE_FALLBACK_SUFFIX}"
    label_path = DATA_FEATURE_DIR / f"{split}_labels.npy"

    # Choose feature file: preferred (40-d) if present, otherwise fallback (120-d).
    if preferred.exists():
        feat_path = preferred
    elif fallback.exists():
        feat_path = fallback
        print(f"⚠️  Preferred 40-d feature not found for split={split}. Falling back to {fallback.name}.")
    else:
        raise FileNotFoundError(
            f"Missing feature files for split={split}: tried {preferred} and {fallback}, and labels {label_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found for split={split}: {label_path}")

    X = np.load(feat_path)
    y = np.load(label_path)

    # Remove rows containing the sentinel value (preserve ordering for remaining)
    mask = ~(X == MISSING_FILL_VALUE).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]

    # If all rows were sentinel after loading features, raise to prompt regeneration.
    if X_clean.shape[0] == 0:
        raise RuntimeError(
            f"All rows are sentinel after loading {feat_path.name}. Regenerate features or check missing files."
        )

    print(f"{split}: loaded {feat_path.name} -> {X.shape} -> sentinel 제거 후 {X_clean.shape}")
    return X_clean, y_clean


def init_classifier(model_name: str = "RF"):
    """Factory to initialize classifier by name. Extendable for other models."""
    model_name = model_name.upper()
    if model_name == "RF":
        return RandomForestClassifier(**RF_PARAMS)
    # placeholders for future models
    if model_name == "SVM":
        from sklearn.svm import SVC

        # Use RBF kernel with recommended defaults
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    if model_name in ("LR", "LOGREG"):
        from sklearn.linear_model import LogisticRegression

        # Recommended settings for logistic regression experiments
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str = "ROC Curve") -> Optional[float]:
    """Compute and save ROC curve. Returns AUC or None if not computable."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None

    if auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=200)
        plt.close()
    return auc


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, labels: list = ["Real", "Fake"]) -> None:
    """Save confusion matrix visualization to out_path."""
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def build_stats(y_true: np.ndarray, y_pred: np.ndarray, auc: Optional[float], cm: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    stats = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": cm.tolist(),
    }
    return stats


def save_metrics(report_path: Path, stats: dict) -> None:
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model evaluation report\n")
        f.write(f"Generated: {datetime.datetime.now()}\n\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def main():
    X_train, y_train = load_split("train")
    X_dev, y_dev = load_split("dev")

    if len(X_train) == 0 or len(X_dev) == 0:
        raise RuntimeError("Empty training or dev set after sentinel removal.")

    clf = init_classifier(MODEL_NAME)
    model_name_upper = MODEL_NAME.upper()
    print(f"Training model: {model_name_upper}")
    clf.fit(X_train, y_train)

    # Evaluation on dev set
    if USE_TQDM_FOR_EVAL:
        # iterative prediction to show progress (useful for very large dev sets)
        y_pred = []
        y_prob = [] if hasattr(clf, "predict_proba") else None
        for x in tqdm(X_dev, desc="Predicting (dev)"):
            x = x.reshape(1, -1)
            y_pred.append(int(clf.predict(x)[0]))
            if y_prob is not None:
                y_prob.append(float(clf.predict_proba(x)[:, 1][0]))
        y_pred = np.array(y_pred, dtype=np.int64)
        y_prob = np.array(y_prob, dtype=np.float32) if y_prob is not None else None
    else:
        y_pred = clf.predict(X_dev)
        y_prob = clf.predict_proba(X_dev)[:, 1] if hasattr(clf, "predict_proba") else None

    cm = confusion_matrix(y_dev, y_pred)
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_dev, y_prob)
        except Exception:
            auc = None

    stats = build_stats(y_dev, y_pred, auc, cm)

    print("=== Dev 성능 ===")
    print(f"Accuracy : {stats['accuracy']:.3f}")
    print(f"Precision: {stats['precision']:.3f}")
    print(f"Recall   : {stats['recall']:.3f}")
    print(f"F1-score : {stats['f1']:.3f}")
    if stats["roc_auc"] is not None:
        print(f"ROC-AUC  : {stats['roc_auc']:.3f}")
    print("Confusion Matrix:")
    print(cm)

    # Save model (include model name)
    model_path = OUT_MODELS_DIR / f"model_{model_name_upper}.pkl"
    joblib.dump(clf, model_path)


    # Create per-model results directory: results/<MODEL>/
    model_results_dir = OUT_RESULTS_DIR / model_name_upper
    model_results_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save (include model name) inside per-model folder
    roc_path = model_results_dir / f"roc_curve_{model_name_upper}.png"
    cm_path = model_results_dir / f"confusion_matrix_{model_name_upper}.png"
    if y_prob is not None and stats["roc_auc"] is not None:
        plot_roc_curve(y_dev, y_prob, roc_path, title=f"ROC Curve - {model_name_upper} (dev)")
    plot_confusion_matrix(cm, cm_path)

    # Save metrics summary (include model name) inside per-model folder
    report_path = model_results_dir / f"metrics_{model_name_upper}.txt"
    stats_with_meta = {"model": model_name_upper, **stats}
    save_metrics(report_path, stats_with_meta)

    print(f"\nModel saved to: {model_path}")
    if (y_prob is not None) and (stats["roc_auc"] is not None):
        print(f"ROC plot saved to: {roc_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Metrics saved to: {report_path}")


if __name__ == "__main__":
    main()
