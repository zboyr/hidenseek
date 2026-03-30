"""Generate all figures for the updated report."""
import json
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)


def load_all_metrics():
    """Load all run metrics grouped by config."""
    results = {}
    for d in sorted(glob.glob("reports/output_*")):
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        m = json.load(open(mp))
        mode = m.get("mode", "duplicate")
        det = m.get("detective_model", "?")
        paras = tuple(sorted(m.get("paraphrasers", [])))
        no_hist = m.get("auditor_no_history", False)
        trials = m.get("num_trials", 0)
        acc = m.get("accuracy", 0)
        key = (mode, det, paras, no_hist, trials)
        results.setdefault(key, []).append((d, acc))
    return results


# ============================================================
# Figure 1: Classification pairwise matching — main result
# ============================================================
def fig1_classification_main():
    for d in sorted(glob.glob("reports/output_*")):
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        m = json.load(open(mp))
        if m.get("mode") != "classification":
            continue
        e = json.load(open(os.path.join(d, "eval_output.json")))

        correct_counts = e["per_trial_correct"]
        dist = Counter(correct_counts)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: per-trial bar
        trials = range(len(correct_counts))
        colors = ["#2ecc71" if c == 4 else "#e74c3c" for c in correct_counts]
        axes[0].bar(trials, correct_counts, color=colors, edgecolor="white", linewidth=0.5)
        axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Random E[correct]=1")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Correct Matches (out of 4)")
        axes[0].set_title("Per-Trial Correct Matches")
        axes[0].set_ylim(0, 4.5)
        axes[0].legend()

        # Right: distribution
        x_vals = [0, 1, 2, 3, 4]
        counts = [dist.get(v, 0) for v in x_vals]
        bar_colors = ["#95a5a6", "#e67e22", "#f39c12", "#3498db", "#2ecc71"]
        axes[1].bar(x_vals, counts, color=bar_colors, edgecolor="white", width=0.7)
        axes[1].set_xlabel("Number of Correct Matches")
        axes[1].set_ylabel("Number of Trials")
        axes[1].set_title(f"Match Distribution (n=50, all-correct={e['num_all_correct']}/50={e['overall_accuracy']:.0%})")
        axes[1].set_xticks(x_vals)
        axes[1].set_xticklabels(["0/4", "1/4", "2/4", "3/4", "4/4"])

        plt.tight_layout()
        plt.savefig(f"{OUT}/fig1_classification_main.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("fig1 done")
        break


# ============================================================
# Figure 2: Per-paraphraser match rate (classification)
# ============================================================
def fig2_paraphraser_matchrate():
    for d in sorted(glob.glob("reports/output_*")):
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        m = json.load(open(mp))
        if m.get("mode") != "classification":
            continue
        e = json.load(open(os.path.join(d, "eval_output.json")))

        para_correct = defaultdict(int)
        para_total = defaultdict(int)
        for trial in e["trials"]:
            gt = {int(k): v for k, v in trial["ground_truth"].items()}
            pred = {int(k): v for k, v in trial["predicted"].items()}
            labels_a = trial["shuffled_a_labels"]
            for i, pid in enumerate(labels_a):
                para_total[pid] += 1
                if gt.get(i) == pred.get(i):
                    para_correct[pid] += 1

        names = ["human_par3", "sonnet_46", "gemini_25_pro", "gpt_54"]
        display = ["Human\n(PAR3)", "Sonnet 4.6", "Gemini\n2.5 Pro", "GPT-5.4"]
        rates = [para_correct[n] / para_total[n] * 100 for n in names]
        colors = ["#1abc9c", "#9b59b6", "#e67e22", "#3498db"]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(display, rates, color=colors, edgecolor="white", width=0.6)
        ax.axhline(y=25, color="gray", linestyle="--", alpha=0.6, label="Random baseline (25%)")
        ax.set_ylabel("Individual Match Rate (%)")
        ax.set_title("Per-Paraphraser Match Rate in Classification Task")
        ax.set_ylim(0, 80)
        ax.legend()
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{rate:.0f}%", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUT}/fig2_paraphraser_matchrate.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("fig2 done")
        break


# ============================================================
# Figure 3: Detective model comparison (duplicate mode)
# ============================================================
def fig3_detective_comparison():
    all_metrics = load_all_metrics()
    weak_paras = tuple(sorted(["dipper_lowmed", "gemini_25_flash", "human_par3", "llama_31_8b"]))

    data = {}  # (det_short, hist_str) -> [accs]
    for key, runs in all_metrics.items():
        mode, det, paras, no_hist, trials = key
        if mode != "duplicate" or trials != 10 or paras != weak_paras:
            continue
        det_short = det.split("/")[-1]
        hist_str = "No History" if no_hist else "History"
        data[(det_short, hist_str)] = [acc for _, acc in runs]

    detectives = ["claude-sonnet-4.6", "gpt-4.1", "gemini-2.5-pro"]
    det_display = ["Sonnet 4.6", "GPT-4.1", "Gemini 2.5 Pro"]
    modes = ["History", "No History"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(detectives))
    width = 0.35

    for i, mode in enumerate(modes):
        means = []
        stds = []
        for det in detectives:
            accs = data.get((det, mode), [])
            means.append(np.mean(accs) * 100 if accs else 0)
            stds.append(np.std(accs) * 100 if accs else 0)
        bars = ax.bar(x + i * width, means, width, yerr=stds,
                      label=mode, capsize=4, alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{m:.1f}%", ha="center", fontsize=9)

    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Random baseline (10%)")
    ax.set_xlabel("Detective Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Duplicate Detection: Detective Model Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(det_display)
    ax.set_ylim(0, 85)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_detective_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig3 done")


# ============================================================
# Figure 4: Weak vs Strong paraphrasers (duplicate mode)
# ============================================================
def fig4_weak_vs_strong():
    all_metrics = load_all_metrics()
    weak_paras = tuple(sorted(["dipper_lowmed", "gemini_25_flash", "human_par3", "llama_31_8b"]))
    strong_paras = tuple(sorted(["gemini_25_pro", "gpt_54", "human_par3", "sonnet_46"]))

    weak_accs = []
    strong_accs = []
    for key, runs in all_metrics.items():
        mode, det, paras, no_hist, trials = key
        if mode != "duplicate" or trials != 10:
            continue
        if det == "anthropic/claude-sonnet-4.6" and no_hist:
            if paras == weak_paras:
                weak_accs = [acc for _, acc in runs]
            elif paras == strong_paras:
                strong_accs = [acc for _, acc in runs]

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Weak Paraphrasers\n(Llama-8B, Gemini Flash,\nGPT-5-Nano, DIPPER)",
              "Strong Paraphrasers\n(Sonnet 4.6, Gemini 2.5 Pro,\nGPT-5.4, Human)"]
    data = [weak_accs, strong_accs]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors = ["#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, accs in enumerate(data):
        ax.scatter([i + 1] * len(accs), accs, color="black", zorder=5, s=20, alpha=0.5)

    means = [np.mean(a) * 100 for a in data]
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="Random baseline (10%)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Duplicate Detection: Weak vs Strong Paraphrasers\n"
                 f"(Weak mean={means[0]:.1f}%, Strong mean={means[1]:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_weak_vs_strong.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig4 done")


# ============================================================
# Figure 5: Summary table as figure
# ============================================================
def fig5_summary_table():
    rows = [
        ["Duplicate Detection", "Weak", "Sonnet 4.6", "History", "10", "10", "61.1%"],
        ["Duplicate Detection", "Weak", "Sonnet 4.6", "No History", "10", "10", "61.1%"],
        ["Duplicate Detection", "Weak", "GPT-4.1", "History", "10", "5", "55.6%"],
        ["Duplicate Detection", "Weak", "GPT-4.1", "No History", "10", "5", "57.8%"],
        ["Duplicate Detection", "Weak", "Gemini 2.5 Pro", "History", "10", "5", "62.2%"],
        ["Duplicate Detection", "Weak", "Gemini 2.5 Pro", "No History", "10", "5", "60.0%"],
        ["Duplicate Detection", "Strong", "Sonnet 4.6", "No History", "10", "5", "66.7%"],
        ["Classification", "Strong", "Sonnet 4.6", "No History", "50", "1", "22.0%"],
    ]
    cols = ["Mode", "Paraphrasers", "Detective", "History", "Trials", "Runs", "Accuracy"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Header style
    for j in range(len(cols)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight classification row
    for j in range(len(cols)):
        table[len(rows), j].set_facecolor("#ffeaa7")

    ax.set_title("Summary of All Experiments", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_summary_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("fig5 done")


# ============================================================
# Figure 6: Classification — confusion-style matching heatmap
# ============================================================
def fig6_classification_confusion():
    for d in sorted(glob.glob("reports/output_*")):
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        m = json.load(open(mp))
        if m.get("mode") != "classification":
            continue
        e = json.load(open(os.path.join(d, "eval_output.json")))

        names = ["human_par3", "sonnet_46", "gemini_25_pro", "gpt_54"]
        display = ["Human (PAR3)", "Sonnet 4.6", "Gemini 2.5 Pro", "GPT-5.4"]
        n = len(names)
        # confusion[true_pid][predicted_pid] = count
        confusion = np.zeros((n, n), dtype=int)
        name_to_idx = {name: i for i, name in enumerate(names)}

        for trial in e["trials"]:
            gt = {int(k): v for k, v in trial["ground_truth"].items()}
            pred = {int(k): v for k, v in trial["predicted"].items()}
            labels_a = trial["shuffled_a_labels"]
            labels_b = trial["shuffled_b_labels"]
            for i_a, true_pid in enumerate(labels_a):
                pred_b_idx = pred.get(i_a, -1)
                if 0 <= pred_b_idx < len(labels_b):
                    pred_pid = labels_b[pred_b_idx]
                    confusion[name_to_idx[true_pid]][name_to_idx[pred_pid]] += 1

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(confusion, cmap="YlOrRd")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display, rotation=30, ha="right")
        ax.set_yticklabels(display)
        ax.set_xlabel("Predicted Paraphraser (Group B)")
        ax.set_ylabel("True Paraphraser (Group A)")
        ax.set_title("Classification Confusion Matrix\n(rows=true, cols=predicted)")

        for i in range(n):
            for j in range(n):
                color = "white" if confusion[i, j] > confusion.max() * 0.6 else "black"
                ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                        color=color, fontweight="bold", fontsize=13)

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(f"{OUT}/fig6_classification_confusion.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("fig6 done")
        break


if __name__ == "__main__":
    fig1_classification_main()
    fig2_paraphraser_matchrate()
    fig3_detective_comparison()
    fig4_weak_vs_strong()
    fig5_summary_table()
    fig6_classification_confusion()
    print("\nAll figures saved to", OUT)
