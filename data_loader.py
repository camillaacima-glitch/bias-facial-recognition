import pandas as pd
import matplotlib.pyplot as plt

# ── LOAD DATA ──────────────────────────────────────────────
def load_labels(train_path, val_path):
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    print("Train samples:", len(train))
    print("Val samples:  ", len(val))
    print("\nColumns:", list(train.columns))
    return train, val

# ── ANALYZE DISTRIBUTION ───────────────────────────────────
def analyze_distribution(df, label="Train"):
    print(f"\n── {label} Distribution ──")
    for col in ["race", "gender", "age"]:
        print(f"\n{col}:\n{df[col].value_counts()}")

# ── PLOT DISTRIBUTION ──────────────────────────────────────
def plot_distribution(df, label="Train"):
    PINK       = "#F48FB1"
    DARK_PINK  = "#C2185B"
    LIGHT_PINK = "#FCE4EC"
    TEXT       = "#880E4F"

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
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
        bars[0].set_color(DARK_PINK)
        bars[-1].set_color("#F06292")

    plt.tight_layout()
    plt.savefig(f"{label.lower()}_distribution.png")
    plt.show()
    print(f"Saved: {label.lower()}_distribution.png")

# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    train, val = load_labels("train_labels.csv", "val_labels.csv")
    analyze_distribution(train, "Train")
