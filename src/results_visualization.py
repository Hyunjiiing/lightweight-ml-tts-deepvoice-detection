import os
import matplotlib.pyplot as plt

# Sample sizes and metrics (from our previous discussion)
sample_sizes = [120, 500, 1000, 1500]

accuracy = [0.8000, 0.8500, 0.7767, 0.7800]
precision = [0.7812, 0.9070, 0.7515, 0.7561]
recall = [0.8333, 0.7800, 0.8267, 0.8267]
f1 = [0.8065, 0.8387, 0.7873, 0.7898]
auc = [0.8783, 0.9030, 0.8628, 0.8838]

# Helper to create and save a single-metric line plot
def make_metric_plot(x, y, metric_name, filename):
    # Ensure output directory exists before saving. If creating the requested
    # directory fails (e.g. permission denied for root '/results'), fall back
    # to a project-local `results/data_difference` directory.
    dirpath = os.path.dirname(filename)
    try:
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
    except PermissionError:
        # Fallback to project-local results directory
        fallback_dir = os.path.join(os.getcwd(), "results", "data_difference")
        os.makedirs(fallback_dir, exist_ok=True)
        # replace filename with fallback location keeping the original basename
        filename = os.path.join(fallback_dir, os.path.basename(filename))
        print(f"Permission denied creating '{dirpath}'; falling back to '{fallback_dir}'.")
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel("Number of training samples")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs. Number of training samples")
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Create plots
make_metric_plot(sample_sizes, accuracy, "Accuracy", "/results/data_difference/accuracy_vs_samples.png")
make_metric_plot(sample_sizes, precision, "Precision", "/results/data_difference/precision_vs_samples.png")
make_metric_plot(sample_sizes, recall, "Recall", "/results/data_difference/recall_vs_samples.png")
make_metric_plot(sample_sizes, f1, "F1-score", "/results/data_difference/f1_vs_samples.png")
make_metric_plot(sample_sizes, auc, "ROC-AUC", "/results/data_difference/auc_vs_samples.png")

"/results/data_difference/accuracy_vs_samples.png", "/results/data_difference/precision_vs_samples.png", "/results/data_difference/recall_vs_samples.png", "/results/data_difference/f1_vs_samples.png", "/results/data_difference/auc_vs_samples.png"
