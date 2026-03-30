"""Generate flow diagrams for the presentation slides."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = "reports/figures"
os.makedirs(OUT, exist_ok=True)

# Color palette
C_BG = "#ffffff"
C_AUDITOR = "#2c3e50"
C_PARA = "#2980b9"
C_PARA_DUP = "#e74c3c"
C_DETECTIVE = "#8e44ad"
C_DATA = "#27ae60"
C_OUTPUT = "#f39c12"
C_HUMAN = "#1abc9c"
C_TEXT_A = "#e67e22"
C_TEXT_B = "#3498db"
C_LIGHT = "#ecf0f1"


def draw_box(ax, x, y, w, h, text, color, fontsize=10, text_color="white", alpha=0.9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05", facecolor=color,
                         edgecolor="none", alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color="#7f8c8d", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                zorder=1)


# ============================================================
# Flow 1: Duplicate Detection
# ============================================================
def fig_flow_duplicate():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Row 1: PAR3 Dataset
    draw_box(ax, 5, 7.2, 3.5, 0.7, "PAR3 Dataset (100 texts)", C_DATA, fontsize=11)

    # Arrow down
    draw_arrow(ax, 5, 6.85, 5, 6.25)

    # Row 2: Source Text
    draw_box(ax, 5, 5.9, 2.8, 0.6, "Random Source Text", C_AUDITOR, fontsize=10)

    # Fan out arrows to paraphrasers
    para_x = [1.8, 4.0, 6.0, 8.2]
    para_labels = ["Human\n(PAR3)", "Sonnet 4.6", "Gemini\n2.5 Pro", "GPT-5.4"]
    para_colors = [C_HUMAN, C_PARA, C_PARA, C_PARA]

    for i, (px, label, color) in enumerate(zip(para_x, para_labels, para_colors)):
        draw_arrow(ax, 5, 5.6, px, 4.85)
        draw_box(ax, px, 4.5, 1.8, 0.65, label, color, fontsize=9)

    # Duplicate star on one
    dup_idx = 2  # Gemini duplicated as example
    ax.text(para_x[dup_idx] + 0.9, 4.85, "★ ×2", fontsize=12, color=C_PARA_DUP,
            fontweight="bold", zorder=4)

    # Arrows down to shuffled outputs
    for px in para_x:
        draw_arrow(ax, px, 4.17, 5, 3.25)

    # Row 4: Shuffled outputs
    draw_box(ax, 5, 2.9, 4.5, 0.65, "5 Shuffled Outputs  [?, ?, ?, ?, ?]", C_OUTPUT,
             fontsize=10, text_color="#2c3e50")

    # Arrow down
    draw_arrow(ax, 5, 2.57, 5, 1.85)

    # Row 5: Detective
    draw_box(ax, 5, 1.5, 3.5, 0.65, "Detective LLM → Which 2 match?", C_DETECTIVE, fontsize=10)

    # Baseline annotation
    ax.text(9.5, 0.5, "Baseline: 10%", fontsize=9, ha="right", color="#95a5a6",
            style="italic")

    ax.set_title("Duplicate Detection Flow", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/flow_duplicate.png", dpi=200, bbox_inches="tight",
                facecolor=C_BG)
    plt.close()
    print("flow_duplicate done")


# ============================================================
# Flow 2: Pairwise Classification
# ============================================================
def fig_flow_classification():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Row 1: Two source texts
    draw_box(ax, 3.5, 8.2, 3.0, 0.65, "Text A (from PAR3)", C_TEXT_A, fontsize=10)
    draw_box(ax, 8.5, 8.2, 3.0, 0.65, "Text B (from PAR3)", C_TEXT_B, fontsize=10)

    # Row 2: Paraphrasers (shared)
    para_x = [2.0, 4.5, 7.5, 10.0]
    para_labels = ["Human\n(PAR3)", "Sonnet\n4.6", "Gemini\n2.5 Pro", "GPT-5.4"]
    para_colors = [C_HUMAN, C_PARA, C_PARA, C_PARA]

    for px, label, color in zip(para_x, para_labels, para_colors):
        draw_box(ax, px, 6.2, 1.8, 0.7, label, color, fontsize=9)
        # Arrows from both texts
        draw_arrow(ax, 3.5, 7.87, px, 6.55, color=C_TEXT_A)
        draw_arrow(ax, 8.5, 7.87, px, 6.55, color=C_TEXT_B)

    # Parallel badge
    ax.text(6.0, 7.0, "⚡ 8 parallel calls", fontsize=9, ha="center",
            color="#7f8c8d", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", edgecolor="none"))

    # Arrows down to groups
    for px in para_x:
        draw_arrow(ax, px, 5.85, 3.5, 4.75, color=C_TEXT_A)
        draw_arrow(ax, px, 5.85, 8.5, 4.75, color=C_TEXT_B)

    # Row 3: Shuffled groups
    draw_box(ax, 3.5, 4.4, 3.5, 0.65, "Group A (4 shuffled)", C_TEXT_A, fontsize=10)
    draw_box(ax, 8.5, 4.4, 3.5, 0.65, "Group B (4 shuffled)", C_TEXT_B, fontsize=10)

    # Arrows down to detective
    draw_arrow(ax, 3.5, 4.07, 6.0, 3.05)
    draw_arrow(ax, 8.5, 4.07, 6.0, 3.05)

    # Row 4: Detective
    draw_box(ax, 6.0, 2.7, 5.0, 0.65, "Detective LLM → Match A↔B pairs", C_DETECTIVE,
             fontsize=11)

    # Row 5: Result example
    draw_arrow(ax, 6.0, 2.37, 6.0, 1.55)
    draw_box(ax, 6.0, 1.2, 5.5, 0.65,
             "A0↔B2,  A1↔B0,  A2↔B3,  A3↔B1", C_LIGHT,
             fontsize=10, text_color=C_AUDITOR)

    # Baseline + scoring
    ax.text(11.5, 0.4, "All 4 correct → ✓\nBaseline: 1/24 = 4.17%",
            fontsize=9, ha="right", color="#95a5a6", style="italic")

    ax.set_title("Pairwise Classification Flow", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/flow_classification.png", dpi=200, bbox_inches="tight",
                facecolor=C_BG)
    plt.close()
    print("flow_classification done")


# ============================================================
# Flow 3: Motivation — fingerprint concept
# ============================================================
def fig_motivation():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Source text in center-top
    draw_box(ax, 5, 5.0, 3.5, 0.7, "Original Text", C_AUDITOR, fontsize=12)

    # Fan out to 4 paraphrasers
    models = [
        (1.5, 3.0, "Human", C_HUMAN),
        (4.0, 3.0, "Model A", "#3498db"),
        (6.0, 3.0, "Model B", "#e67e22"),
        (8.5, 3.0, "Model C", "#9b59b6"),
    ]
    for mx, my, label, color in models:
        draw_arrow(ax, 5, 4.65, mx, 3.35)
        draw_box(ax, mx, my, 1.8, 0.6, label, color, fontsize=10)

    # Outputs with fingerprint indicators
    outputs = [
        (1.5, 1.5, '"I must go\nnow, surely."', C_HUMAN),
        (4.0, 1.5, '"It is necessary\nfor me to leave."', "#3498db"),
        (6.0, 1.5, '"I need to\ndepart now."', "#e67e22"),
        (8.5, 1.5, '"I shall take\nmy leave."', "#9b59b6"),
    ]
    for ox, oy, text, color in outputs:
        draw_arrow(ax, ox, 2.7, ox, 2.05)
        box = FancyBboxPatch((ox - 1.0, oy - 0.5), 2.0, 1.0,
                             boxstyle="round,pad=0.08", facecolor=color,
                             edgecolor="none", alpha=0.15, zorder=2)
        ax.add_patch(box)
        # Colored left border
        ax.plot([ox - 1.0, ox - 1.0], [oy - 0.45, oy + 0.45],
                color=color, lw=4, solid_capstyle="round", zorder=3)
        ax.text(ox, oy, text, ha="center", va="center", fontsize=8,
                color="#2c3e50", zorder=4)

    # Question
    ax.text(5, 0.3, "Same meaning, different style → Can we tell who wrote each?",
            ha="center", fontsize=11, color=C_PARA_DUP, fontweight="bold")

    ax.set_title("Paraphraser Fingerprinting", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/flow_motivation.png", dpi=200, bbox_inches="tight",
                facecolor=C_BG)
    plt.close()
    print("flow_motivation done")


# ============================================================
# Flow 4: PAR3 dataset concept
# ============================================================
def fig_par3_concept():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Original foreign text
    draw_box(ax, 2.5, 5.0, 3.5, 0.7, "Foreign Original (e.g. Russian)",
             "#7f8c8d", fontsize=10)

    # Multiple translations
    translations = [
        (7.0, 4.2, "Translation 1 (1920s)", "#2980b9",
         '"But I tell you I intend\n to return soon..."'),
        (7.0, 2.8, "Translation 2 (1960s)", "#e67e22",
         '"But, I tell you, I\'m coming\n back directly..."'),
        (7.0, 1.4, "Translation 3 (2001)", "#27ae60",
         '"I\'ve told you I\'m coming\n back soon..."'),
    ]

    for i, (tx, ty, label, color, text) in enumerate(translations):
        draw_arrow(ax, 4.25, 5.0 - i * 0.3, 5.2, ty)
        # Label
        draw_box(ax, tx, ty, 3.8, 0.5, label, color, fontsize=9)
        # Quote below
        ax.text(tx, ty - 0.5, text, ha="center", va="center", fontsize=8,
                color="#2c3e50", style="italic")

    # Bracket showing "natural paraphrases"
    ax.annotate("", xy=(9.3, 4.4), xytext=(9.3, 1.2),
                arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=1.5,
                                connectionstyle="arc3,rad=0"))
    ax.plot([9.2, 9.4], [4.4, 4.4], color="#2c3e50", lw=1.5)
    ax.plot([9.2, 9.4], [1.2, 1.2], color="#2c3e50", lw=1.5)
    ax.text(9.6, 2.8, "Natural\nHuman\nParaphrases", fontsize=9,
            color="#2c3e50", fontweight="bold", va="center")

    # Arrow showing usage
    ax.text(2.5, 1.0, "Trans. 1 → Source Text\nTrans. 2,3 → Human Paraphraser Output",
            ha="center", fontsize=9, color="#7f8c8d",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeaa7", edgecolor="none"))

    ax.set_title("PAR3 Corpus: Multiple Human Translations as Paraphrases",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/flow_par3.png", dpi=200, bbox_inches="tight",
                facecolor=C_BG)
    plt.close()
    print("flow_par3 done")


if __name__ == "__main__":
    fig_flow_duplicate()
    fig_flow_classification()
    fig_motivation()
    fig_par3_concept()
    print("\nAll flow diagrams saved to", OUT)
