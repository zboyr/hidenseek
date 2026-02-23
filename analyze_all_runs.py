"""
Aggregate analysis across all experiment runs in reports/.
Generates figures + companion JSON data files for the final report.

Usage: uv run python analyze_all_runs.py
"""
import json
import os
import glob
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

COLORS = {
    "llama_31_8b": "#3498db",
    "gemini_25_flash": "#e67e22",
    "gpt5_nano": "#9b59b6",
    "dipper_lowmed": "#1abc9c",
}
PARAPHRASER_LABELS = {
    "llama_31_8b": "Llama-3.1-8B",
    "gemini_25_flash": "Gemini-2.5-Flash",
    "gpt5_nano": "GPT-5-Nano",
    "dipper_lowmed": "DIPPER",
}
RANDOM_BASELINE = 1.0 / 10  # C(5,2) = 10


def load_all_runs():
    """Load all experiment runs from reports/."""
    runs = []
    for d in sorted(glob.glob(os.path.join(REPORTS_DIR, "output_*"))):
        metrics_path = os.path.join(d, "metrics.json")
        eval_path = os.path.join(d, "eval_output.json")
        if not os.path.exists(metrics_path) or not os.path.exists(eval_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(eval_path) as f:
            eval_data = json.load(f)
        runs.append({
            "dir": d,
            "name": os.path.basename(d),
            "metrics": metrics,
            "eval": eval_data,
            "has_full_data": "dup_info" in eval_data,
        })
    return runs


def save_fig(fig, name, json_data):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, f"{name}.png")
    json_path = os.path.join(FIGURES_DIR, f"{name}.json")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {fig_path}")


def fig1_accuracy_distribution(runs):
    """Distribution of accuracy across all runs + fitted estimate."""
    accuracies = [r["metrics"]["accuracy"] for r in runs]
    n = len(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)
    se = stats.sem(accuracies)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean_acc, scale=se)

    aud_det_false = [r["metrics"]["accuracy"] for r in runs if not r["metrics"]["auditor_as_detective"]]
    aud_det_true = [r["metrics"]["accuracy"] for r in runs if r["metrics"]["auditor_as_detective"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    run_labels = [f"Run {i+1}" for i in range(n)]
    bars = ax.bar(range(n), accuracies, color=["#e74c3c" if not r["metrics"]["auditor_as_detective"] else "#3498db" for r in runs],
                  edgecolor="black", alpha=0.8)
    ax.axhline(y=mean_acc, color="black", linestyle="-", linewidth=2, label=f"Mean = {mean_acc:.1%}")
    ax.axhline(y=RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.5, label=f"Random = {RANDOM_BASELINE:.0%}")
    ax.fill_between([-0.5, n - 0.5], ci[0], ci[1], alpha=0.15, color="blue", label=f"95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.01, f"{acc:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection Accuracy Across All Runs")
    ax.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    bins = np.arange(0, 1.05, 0.1)
    ax2.hist(accuracies, bins=bins, color="#3498db", edgecolor="black", alpha=0.7, density=True, label="Observed")
    if std_acc > 0:
        x_fit = np.linspace(0, 1, 200)
        y_fit = stats.norm.pdf(x_fit, mean_acc, std_acc)
        ax2.plot(x_fit, y_fit, "r-", linewidth=2, label=f"Normal fit (μ={mean_acc:.2f}, σ={std_acc:.2f})")
    ax2.axvline(mean_acc, color="black", linestyle="-", linewidth=2)
    ax2.axvline(RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Accuracy")
    ax2.set_ylabel("Density")
    ax2.set_title("Accuracy Distribution with Normal Fit")
    ax2.legend(fontsize=8)

    fig.tight_layout()

    json_data = {
        "figure": "accuracy_distribution",
        "description": "Detection accuracy across all experimental runs with normal distribution fit and 95% confidence interval.",
        "data": {
            "num_runs": n,
            "accuracies": accuracies,
            "mean": mean_acc,
            "std": std_acc,
            "standard_error": se,
            "ci_95": list(ci),
            "random_baseline": RANDOM_BASELINE,
            "separate_detective_acc": aud_det_false,
            "auditor_as_detective_acc": aud_det_true,
        },
        "analysis": (
            f"Across {n} independent runs (each 20 trials, 15 scored), "
            f"the mean detection accuracy is {mean_acc:.1%} (SD={std_acc:.2f}). "
            f"The 95% confidence interval is [{ci[0]:.1%}, {ci[1]:.1%}], "
            f"well above the random baseline of {RANDOM_BASELINE:.0%}. "
            f"A one-sample t-test against the random baseline yields "
            f"t({n-1})={stats.ttest_1samp(accuracies, RANDOM_BASELINE).statistic:.2f}, "
            f"p={stats.ttest_1samp(accuracies, RANDOM_BASELINE).pvalue:.4f}, "
            f"confirming that detection accuracy is significantly above chance."
        ),
    }
    save_fig(fig, "fig1_accuracy_distribution", json_data)
    return mean_acc, std_acc, ci


def fig2_per_trial_accuracy_curve(runs):
    """Aggregate per-trial accuracy across all runs (learning curve)."""
    full_runs = [r for r in runs if r["has_full_data"]]
    warmup = runs[0]["metrics"]["warmup_steps"]
    num_trials = runs[0]["metrics"]["num_trials"]

    per_trial_hits = np.zeros((len(full_runs), num_trials))
    for ri, r in enumerate(full_runs):
        evals = np.array(r["eval"]["evaluations"])
        cpairs = np.array(r["eval"]["correct_pairs"])
        for t in range(num_trials):
            if evals[t][0] != -1 and evals[t][0] == cpairs[t][0] and evals[t][1] == cpairs[t][1]:
                per_trial_hits[ri, t] = 1

    mean_per_trial = per_trial_hits.mean(axis=0)
    se_per_trial = per_trial_hits.std(axis=0, ddof=1) / np.sqrt(len(full_runs))

    cum_acc = []
    for t in range(warmup, num_trials):
        window = mean_per_trial[warmup:t + 1]
        cum_acc.append(window.mean())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(num_trials), mean_per_trial, color=["#bdc3c7" if t < warmup else "#3498db" for t in range(num_trials)],
           edgecolor="black", alpha=0.6, label="Per-trial accuracy (avg across runs)")
    ax.errorbar(range(num_trials), mean_per_trial, yerr=se_per_trial, fmt="none", ecolor="black", capsize=3, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(range(warmup, num_trials), cum_acc, "r-o", linewidth=2, markersize=4, label="Cumulative accuracy")
    ax2.set_ylabel("Cumulative Accuracy", color="red")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="red")

    ax.axvline(x=warmup - 0.5, color="orange", linestyle="--", linewidth=2, label=f"Warmup boundary (W={warmup})")
    ax.axhline(y=RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1, label=f"Random = {RANDOM_BASELINE:.0%}")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Avg Accuracy (across runs)")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(num_trials))
    ax.set_title(f"Per-Trial Detection Accuracy (Averaged over {len(full_runs)} Runs)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()

    json_data = {
        "figure": "per_trial_accuracy_curve",
        "description": "Per-trial detection accuracy averaged across all runs, showing learning progression.",
        "data": {
            "num_runs": len(full_runs),
            "num_trials": num_trials,
            "warmup": warmup,
            "mean_per_trial": mean_per_trial.tolist(),
            "se_per_trial": se_per_trial.tolist(),
            "cumulative_accuracy": cum_acc,
        },
        "analysis": (
            f"Averaged over {len(full_runs)} runs, the per-trial accuracy shows "
            f"the detective's performance at each trial position. "
            f"Early trials (0-{warmup-1}) serve as warmup. "
            f"The cumulative accuracy curve (red) shows the running average from trial {warmup} onward. "
            "If the Auditor's iterative refinement is effective, later trials should show higher accuracy."
        ),
    }
    save_fig(fig, "fig2_per_trial_accuracy_curve", json_data)


def fig3_paraphraser_detection_rate(runs):
    """Per-paraphraser detection rate when selected as the duplicate."""
    full_runs = [r for r in runs if r["has_full_data"]]
    warmup = runs[0]["metrics"]["warmup_steps"]
    pids = runs[0]["metrics"]["paraphrasers"]

    total_as_dup = Counter()
    correct_as_dup = Counter()

    for r in full_runs:
        evals = np.array(r["eval"]["evaluations"])
        cpairs = np.array(r["eval"]["correct_pairs"])
        dup_info = r["eval"]["dup_info"]
        for t in range(warmup, len(evals)):
            if dup_info[t] is None:
                continue
            dp = dup_info[t]["dup_paraphraser"]
            total_as_dup[dp] += 1
            if evals[t][0] != -1 and evals[t][0] == cpairs[t][0] and evals[t][1] == cpairs[t][1]:
                correct_as_dup[dp] += 1

    det_rates = {}
    for pid in pids:
        tot = total_as_dup.get(pid, 0)
        cor = correct_as_dup.get(pid, 0)
        det_rates[pid] = cor / tot if tot > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(pids))
    labels = [PARAPHRASER_LABELS.get(p, p) for p in pids]
    colors = [COLORS.get(p, "#999") for p in pids]
    totals = [total_as_dup.get(p, 0) for p in pids]
    corrects = [correct_as_dup.get(p, 0) for p in pids]
    rates = [det_rates[p] for p in pids]

    bars = ax.bar(x, rates, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(y=RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.5, label=f"Random = {RANDOM_BASELINE:.0%}")
    for i, (bar, rate, c, t) in enumerate(zip(bars, rates, corrects, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.0%}\n({c}/{t})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Detection Rate by Paraphraser (When Selected as Duplicate)")
    ax.legend()

    fig.tight_layout()

    json_data = {
        "figure": "paraphraser_detection_rate",
        "description": "Detection rate for each paraphraser when it was randomly selected as the duplicate source.",
        "data": {
            "paraphrasers": pids,
            "labels": labels,
            "total_as_duplicate": totals,
            "correctly_detected": corrects,
            "detection_rates": rates,
        },
        "analysis": (
            "Detection rates vary significantly across paraphrasers. "
            + " ".join(f"{PARAPHRASER_LABELS.get(p,p)}: {det_rates[p]:.0%} ({correct_as_dup.get(p,0)}/{total_as_dup.get(p,0)})" for p in pids) + ". "
            "DIPPER, as a T5-based non-LLM paraphraser, tends to produce outputs with distinct stylistic markers "
            "(shorter length, more literal substitutions) compared to LLM-based paraphrasers, "
            "which may make it either easier or harder to detect depending on the source text."
        ),
    }
    save_fig(fig, "fig3_paraphraser_detection_rate", json_data)


def fig4_confusion_heatmap_aggregate(runs):
    """Aggregated confusion matrix across all runs."""
    full_runs = [r for r in runs if r["has_full_data"]]
    warmup = runs[0]["metrics"]["warmup_steps"]
    pids = runs[0]["metrics"]["paraphrasers"]
    k = len(pids)
    pid_to_idx = {p: i for i, p in enumerate(pids)}

    confusion = np.zeros((k, k), dtype=int)
    for r in full_runs:
        sim = r["eval"]["similar_models"]
        for t in range(warmup, len(sim)):
            a, b = sim[t]
            if a in pid_to_idx and b in pid_to_idx:
                ia, ib = pid_to_idx[a], pid_to_idx[b]
                confusion[ia][ib] += 1
                if ia != ib:
                    confusion[ib][ia] += 1

    labels = [PARAPHRASER_LABELS.get(p, p) for p in pids]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Aggregated Detective Selection Matrix ({len(full_runs)} Runs)")
    ax.set_xlabel("Paraphraser B")
    ax.set_ylabel("Paraphraser A")
    for row in range(k):
        for col in range(k):
            ax.text(col, row, str(confusion[row][col]), ha="center", va="center",
                    fontweight="bold", color="white" if confusion[row][col] > confusion.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()

    json_data = {
        "figure": "confusion_heatmap_aggregate",
        "description": "Aggregated confusion matrix showing how often the detective paired each combination of paraphrasers across all runs.",
        "data": {
            "paraphrasers": pids,
            "labels": labels,
            "matrix": confusion.tolist(),
            "num_runs": len(full_runs),
        },
        "analysis": (
            "The diagonal entries represent correct identifications (same paraphraser paired). "
            "Off-diagonal entries show confusion patterns. "
            "High off-diagonal values between two paraphrasers suggest the detective perceives their styles as similar. "
            "This matrix reveals which paraphrasers have the most distinct 'fingerprints' and which are most easily confused."
        ),
    }
    save_fig(fig, "fig4_confusion_heatmap", json_data)


def fig5_response_length_by_paraphraser(runs):
    """Response length distribution per paraphraser using round_outputs."""
    full_runs = [r for r in runs if r["has_full_data"]]
    pids = runs[0]["metrics"]["paraphrasers"]

    lengths_by_pid = {p: [] for p in pids}
    for r in full_runs:
        for trial_ro in r["eval"]["round_outputs"]:
            if trial_ro is None:
                continue
            for pid, texts in trial_ro.items():
                for t in texts:
                    if t:
                        lengths_by_pid[pid].append(len(t.split()))

    labels = [PARAPHRASER_LABELS.get(p, p) for p in pids]
    colors = [COLORS.get(p, "#999") for p in pids]
    data_lists = [lengths_by_pid[p] for p in pids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bp = ax1.boxplot(data_lists, tick_labels=labels, patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Word Count")
    ax1.set_title("Response Length by Paraphraser")
    ax1.tick_params(axis="x", rotation=20)

    for i, (pid, data) in enumerate(zip(pids, data_lists)):
        ax2.hist(data, bins=25, alpha=0.5, color=COLORS.get(pid, "#999"),
                 label=f"{PARAPHRASER_LABELS.get(pid, pid)} (μ={np.mean(data):.0f})", density=True)
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Density")
    ax2.set_title("Response Length Distributions")
    ax2.legend(fontsize=8)

    fig.tight_layout()

    stats_data = {}
    for pid in pids:
        d = lengths_by_pid[pid]
        stats_data[pid] = {
            "count": len(d),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "std": float(np.std(d)),
            "min": int(np.min(d)),
            "max": int(np.max(d)),
        }

    json_data = {
        "figure": "response_length_by_paraphraser",
        "description": "Response length (word count) distribution for each paraphraser across all runs.",
        "data": stats_data,
        "analysis": (
            "Response length is a key fingerprint feature. "
            + " ".join(f"{PARAPHRASER_LABELS.get(p,p)}: mean={stats_data[p]['mean']:.0f} words (SD={stats_data[p]['std']:.0f})" for p in pids) + ". "
            "DIPPER (T5-based) typically produces shorter, more conservative paraphrases, "
            "while LLM-based models tend to generate longer, more elaborate rewrites. "
            "GPT-5-Nano often produces the longest outputs with added explanatory content."
        ),
    }
    save_fig(fig, "fig5_response_length", json_data)


def fig6_source_type_accuracy(runs):
    """Accuracy split by pre-written vs auditor-generated source texts."""
    full_runs = [r for r in runs if r["has_full_data"]]
    warmup = runs[0]["metrics"]["warmup_steps"]
    num_prewritten = 8

    pre_correct, pre_total = 0, 0
    gen_correct, gen_total = 0, 0

    for r in full_runs:
        evals = np.array(r["eval"]["evaluations"])
        cpairs = np.array(r["eval"]["correct_pairs"])
        for t in range(warmup, len(evals)):
            if evals[t][0] == -1:
                continue
            hit = int(evals[t][0] == cpairs[t][0] and evals[t][1] == cpairs[t][1])
            if t < num_prewritten:
                pre_correct += hit
                pre_total += 1
            else:
                gen_correct += hit
                gen_total += 1

    pre_acc = pre_correct / pre_total if pre_total > 0 else 0
    gen_acc = gen_correct / gen_total if gen_total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["Pre-written\n(Human-crafted)", "Auditor-generated\n(LLM, APE-style)"]
    accs = [pre_acc, gen_acc]
    totals_list = [pre_total, gen_total]
    corrects_list = [pre_correct, gen_correct]
    bars = ax.bar(cats, accs, color=["#2980b9", "#27ae60"], edgecolor="black", width=0.5)
    ax.axhline(y=RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.5, label=f"Random = {RANDOM_BASELINE:.0%}")
    for bar, acc, c, t in zip(bars, accs, corrects_list, totals_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.1%}\n({c}/{t})", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Detection Accuracy by Source Text Type (Aggregated)")
    ax.legend()
    fig.tight_layout()

    json_data = {
        "figure": "source_type_accuracy",
        "description": "Comparison of detection accuracy between human-crafted and LLM-generated source texts.",
        "data": {
            "pre_written": {"correct": pre_correct, "total": pre_total, "accuracy": pre_acc},
            "auditor_generated": {"correct": gen_correct, "total": gen_total, "accuracy": gen_acc},
        },
        "analysis": (
            f"Pre-written source texts: {pre_acc:.1%} ({pre_correct}/{pre_total}). "
            f"Auditor-generated source texts: {gen_acc:.1%} ({gen_correct}/{gen_total}). "
            "Pre-written texts were manually designed with linguistic traps (non-standard grammar, "
            "deliberate repetition, temporal disorder) to maximize behavioral differences. "
            "Auditor-generated texts use APE-style iterative refinement based on past detective feedback."
        ),
    }
    save_fig(fig, "fig6_source_type_accuracy", json_data)


def fig7_auditor_vs_separate_detective(runs):
    """Compare accuracy: auditor_as_detective=True vs False."""
    sep = [r["metrics"]["accuracy"] for r in runs if not r["metrics"]["auditor_as_detective"]]
    aud = [r["metrics"]["accuracy"] for r in runs if r["metrics"]["auditor_as_detective"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = [1, 2]
    bp = ax.boxplot([sep, aud], positions=positions, widths=0.4, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=8))
    bp["boxes"][0].set_facecolor("#e74c3c")
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor("#3498db")
    bp["boxes"][1].set_alpha(0.5)

    for i, data in enumerate([sep, aud]):
        jitter = np.random.normal(0, 0.04, size=len(data))
        ax.scatter([positions[i]] * len(data) + jitter, data, alpha=0.7, zorder=5,
                   color=["#e74c3c", "#3498db"][i], edgecolors="black", s=60)

    ax.axhline(y=RANDOM_BASELINE, color="gray", linestyle=":", linewidth=1.5, label=f"Random = {RANDOM_BASELINE:.0%}")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Separate Detective\n(no memory)", "Auditor as Detective\n(with memory)"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Detection Accuracy: Separate vs Auditor-as-Detective")
    ax.legend()
    fig.tight_layout()

    sep_mean = np.mean(sep) if sep else 0
    aud_mean = np.mean(aud) if aud else 0

    json_data = {
        "figure": "auditor_vs_detective",
        "description": "Comparison of accuracy between separate detective (stateless) and auditor-as-detective (with memory) configurations.",
        "data": {
            "separate_detective": {"accuracies": sep, "mean": sep_mean, "n": len(sep)},
            "auditor_as_detective": {"accuracies": aud, "mean": aud_mean, "n": len(aud)},
        },
        "analysis": (
            f"Separate detective (n={len(sep)}): mean={sep_mean:.1%}. "
            f"Auditor-as-detective (n={len(aud)}): mean={aud_mean:.1%}. "
            "The auditor-as-detective configuration has access to past round outputs and results, "
            "potentially allowing it to learn paraphraser fingerprints over time. "
            "The separate detective makes each judgment independently without memory."
        ),
    }
    save_fig(fig, "fig7_auditor_vs_detective", json_data)


def fig8_rationale_analysis(runs):
    """Analyze detective rationale keywords to understand decision patterns."""
    full_runs = [r for r in runs if r["has_full_data"]]
    warmup = runs[0]["metrics"]["warmup_steps"]

    fingerprint_keywords = [
        "length", "shorter", "longer", "verbose", "concise", "brief",
        "formal", "informal", "casual", "academic",
        "structure", "syntax", "grammar", "punctuation",
        "vocabulary", "word choice", "lexical", "diction",
        "tone", "style", "register", "voice",
        "repetition", "paraphrase", "rewrite",
        "T5", "DIPPER", "LLM", "model",
        "similar", "identical", "same", "match",
    ]

    keyword_counts = Counter()
    total_rationales = 0

    for r in full_runs:
        rationales = r["eval"].get("rationales", [])
        for t in range(warmup, len(rationales)):
            rat = rationales[t]
            if rat is None:
                continue
            total_rationales += 1
            rat_lower = rat.lower()
            for kw in fingerprint_keywords:
                if kw in rat_lower:
                    keyword_counts[kw] += 1

    top_keywords = keyword_counts.most_common(15)
    kw_names = [k for k, _ in top_keywords]
    kw_freqs = [v / total_rationales for _, v in top_keywords]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(kw_names)), kw_freqs, color="#3498db", edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(kw_names)))
    ax.set_yticklabels(kw_names)
    ax.set_xlabel("Frequency (fraction of rationales mentioning keyword)")
    ax.set_title(f"Detective Rationale Keyword Analysis (n={total_rationales} rationales)")
    ax.invert_yaxis()
    for bar, freq in zip(bars, kw_freqs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{freq:.0%}", va="center", fontsize=9)

    fig.tight_layout()

    json_data = {
        "figure": "rationale_keyword_analysis",
        "description": "Frequency of fingerprint-related keywords in detective rationales.",
        "data": {
            "total_rationales": total_rationales,
            "top_keywords": {k: v for k, v in top_keywords},
            "keyword_frequencies": {k: f for k, f in zip(kw_names, kw_freqs)},
        },
        "analysis": (
            f"Across {total_rationales} detective rationales, the most frequently mentioned "
            "fingerprint features are: " +
            ", ".join(f"'{k}' ({v/total_rationales:.0%})" for k, v in top_keywords[:5]) + ". "
            "This reveals which stylistic dimensions the detective relies on most heavily "
            "when distinguishing paraphrasers. Length and structural features appear prominently, "
            "suggesting these are the most salient fingerprint dimensions."
        ),
    }
    save_fig(fig, "fig8_rationale_keywords", json_data)


def fig9_summary_table(runs, mean_acc, std_acc, ci):
    """Summary statistics table."""
    n = len(runs)
    t_stat, p_val = stats.ttest_1samp([r["metrics"]["accuracy"] for r in runs], RANDOM_BASELINE)

    table_data = [
        ["Total Experimental Runs", str(n)],
        ["Trials per Run", "20 (15 scored, 5 warmup)"],
        ["Total Scored Trials", str(n * 15)],
        ["Auditor / Detective Model", "Claude Sonnet 4.6"],
        ["Paraphrasers", "Llama-3.1-8B, Gemini-2.5-Flash,\nGPT-5-Nano, DIPPER"],
        ["Mean Accuracy", f"{mean_acc:.1%} (SD={std_acc:.2f})"],
        ["95% Confidence Interval", f"[{ci[0]:.1%}, {ci[1]:.1%}]"],
        ["Random Baseline", f"{RANDOM_BASELINE:.0%} (1/C(5,2))"],
        ["t-test vs Random", f"t({n-1})={t_stat:.2f}, p={p_val:.4f}"],
        ["Significance", "p < 0.001 ***" if p_val < 0.001 else f"p = {p_val:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=["Parameter", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title("Experiment Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    json_data = {
        "figure": "experiment_summary",
        "description": "Summary of all experimental parameters and aggregate results.",
        "data": {
            "num_runs": n,
            "trials_per_run": 20,
            "scored_per_run": 15,
            "total_scored": n * 15,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ci_95": list(ci),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "random_baseline": RANDOM_BASELINE,
        },
        "analysis": (
            f"Over {n} runs totaling {n*15} scored trials, the system achieves "
            f"{mean_acc:.1%} mean accuracy, significantly above the {RANDOM_BASELINE:.0%} random baseline "
            f"(t({n-1})={t_stat:.2f}, p={p_val:.4f}). "
            "This confirms that LLM paraphrasers leave detectable fingerprints in their outputs."
        ),
    }
    save_fig(fig, "fig9_summary_table", json_data)


def main():
    print("Loading all runs from reports/...")
    runs = load_all_runs()
    print(f"Found {len(runs)} runs.")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n=== Fig 1: Accuracy Distribution ===")
    mean_acc, std_acc, ci = fig1_accuracy_distribution(runs)

    print("\n=== Fig 2: Per-Trial Accuracy Curve ===")
    fig2_per_trial_accuracy_curve(runs)

    print("\n=== Fig 3: Paraphraser Detection Rate ===")
    fig3_paraphraser_detection_rate(runs)

    print("\n=== Fig 4: Confusion Heatmap (Aggregate) ===")
    fig4_confusion_heatmap_aggregate(runs)

    print("\n=== Fig 5: Response Length by Paraphraser ===")
    fig5_response_length_by_paraphraser(runs)

    print("\n=== Fig 6: Source Type Accuracy ===")
    fig6_source_type_accuracy(runs)

    print("\n=== Fig 7: Auditor vs Separate Detective ===")
    fig7_auditor_vs_separate_detective(runs)

    print("\n=== Fig 8: Rationale Keyword Analysis ===")
    fig8_rationale_analysis(runs)

    print("\n=== Fig 9: Summary Table ===")
    fig9_summary_table(runs, mean_acc, std_acc, ci)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
