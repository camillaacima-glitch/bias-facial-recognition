import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── PALETTE ────────────────────────────────────────────────
PINK       = "#F48FB1"
DARK_PINK  = "#C2185B"
LIGHT_PINK = "#FCE4EC"
MID_PINK   = "#F06292"
TEXT       = "#880E4F"

def set_style():
    plt.rcParams.update({
        "figure.facecolor":  LIGHT_PINK,
        "axes.facecolor":    LIGHT_PINK,
        "axes.edgecolor":    DARK_PINK,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        PINK,
        "grid.alpha":        0.4,
    })

# ── 1. DISTRIBUTION BAR CHARTS ─────────────────────────────
def plot_distribution(df, label="Train"):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{label} Dataset — Demographic Distribution", 
                 fontsize=14, fontweight="bold", color=TEXT)

    for ax, col in zip(axes, ["race", "gender", "age"]):
        counts = df[col].value_counts()
        bars = ax.bar(counts.index, counts.values, color=PINK, edgecolor=DARK_PINK)
        ax.set_title(col.capitalize(), color=TEXT, fontweight="bold")
        ax.set_xlabel(col, color=TEXT)
        ax.set_ylabel("Count", color=TEXT)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")
        # highlight most and least represented
        bars[0].set_color(DARK_PINK)
        bars[-1].set_color(MID_PINK)

    plt.tight_layout()
    plt.savefig("distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: distribution.png")

# ── 2. DEMOGRAPHIC PARITY BAR CHART ────────────────────────
def plot_demographic_parity(df):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Demographic Parity — Positive Rate by Group",
                 fontsize=14, fontweight="bold", color=TEXT)

    for ax, col in zip(axes, ["race", "gender", "age"]):
        rates = df.groupby(col)["service_test"].mean().sort_values(ascending=False)
        overall = df["service_test"].mean()

        bars = ax.bar(rates.index, rates.values, color=PINK, edgecolor=DARK_PINK)
        ax.axhline(overall, color=DARK_PINK, linestyle="--", linewidth=1.5, label=f"Overall: {overall:.2f}")
        ax.set_title(col.capitalize(), color=TEXT, fontweight="bold")
        ax.set_ylabel("Positive Rate", color=TEXT)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
        ax.grid(axis="y")
        ax.legend(fontsize=8)

        # color bars above/below overall mean
        for bar, val in zip(bars, rates.values):
            bar.set_color(DARK_PINK if val > overall else MID_PINK)

    plt.tight_layout()
    plt.savefig("demographic_parity.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: demographic_parity.png")


# ── 3. ACCURACY GAP ────────────────────────────────────────
def plot_accuracy_gap(df):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Accuracy Gap vs Overall Mean",
                 fontsize=14, fontweight="bold", color=TEXT)

    for ax, col in zip(axes, ["race", "gender", "age"]):
        overall = df["service_test"].mean()
        gap = (df.groupby(col)["service_test"].mean() - overall).sort_values()

        colors = [DARK_PINK if v < 0 else MID_PINK for v in gap.values]
        ax.barh(gap.index, gap.values, color=colors, edgecolor=TEXT)
        ax.axvline(0, color=TEXT, linewidth=1.2)
        ax.set_title(col.capitalize(), color=TEXT, fontweight="bold")
        ax.set_xlabel("Gap from mean", color=TEXT)
        ax.grid(axis="x")

    plt.tight_layout()
    plt.savefig("accuracy_gap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: accuracy_gap.png")

# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("train_labels.csv")
    df["service_test"] = df["service_test"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    plot_distribution(df, "Train")
    plot_demographic_parity(df)
    plot_accuracy_gap(df)
    