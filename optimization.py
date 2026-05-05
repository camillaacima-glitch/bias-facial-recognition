import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

def encode_features(df):
    le_gender = LabelEncoder()
    le_race   = LabelEncoder()
    le_age    = LabelEncoder()
    df = df.copy()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["race_enc"]   = le_race.fit_transform(df["race"])
    df["age_enc"]    = le_age.fit_transform(df["age"])
    return df

def balance_smote(df):
    from imblearn.over_sampling import SMOTE
    features = ["gender_enc", "race_enc", "age_enc"]
    X = df[features]
    y = df["service_test"]
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    df_smote = pd.DataFrame(X_res, columns=features)
    df_smote["service_test"] = y_res
    return df_smote

# ── K OPTIMIZATION ─────────────────────────────────────────
# Tests multiple values of k and plots overall + per-group accuracy.
# Goal: find the k that maximizes fairness across groups, not just overall.
def find_best_k(train_df, val_df, k_values=[1, 3, 5, 7, 10, 15, 20]):
    features = ["gender_enc", "race_enc", "age_enc"]
    target   = "service_test"

    overall_accs = []
    group_accs   = {race: [] for race in sorted(val_df["race"].unique())}

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_df[features], train_df[target])
        preds = model.predict(val_df[features])

        overall_accs.append(accuracy_score(val_df[target], preds))

        val_copy = val_df.copy()
        val_copy["predicted"] = preds
        for race in group_accs:
            subset = val_copy[val_copy["race"] == race]
            acc = accuracy_score(subset[target], subset["predicted"])
            group_accs[race].append(acc)

        print(f"k={k:2d} → overall accuracy: {overall_accs[-1]:.3f}")

    best_k = k_values[np.argmax(overall_accs)]
    print(f"\nBest k: {best_k} (overall accuracy: {max(overall_accs):.3f})")
    return k_values, overall_accs, group_accs, best_k

# ── PLOT K OPTIMIZATION ────────────────────────────────────
def plot_k_optimization(k_values, overall_accs, group_accs, best_k):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("KNN — Finding the Optimal k (SMOTE-balanced training)",
                 fontsize=14, fontweight="bold", color=TEXT)

    # Left: overall accuracy
    ax = axes[0]
    ax.plot(k_values, overall_accs, color=DARK_PINK, marker="o",
            linewidth=2, markersize=8, label="Overall accuracy")
    ax.axvline(best_k, color=MID_PINK, linestyle="--",
               linewidth=1.5, label=f"Best k={best_k}")
    ax.set_xlabel("k", color=TEXT)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_title("Overall Accuracy by k", color=TEXT, fontweight="bold")
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid()

    # Right: per-group accuracy
    ax = axes[1]
    colors = [DARK_PINK, MID_PINK, PINK, "#E91E8C", "#AD1457", "#F48FB1", "#FCE4EC"]
    for i, (race, accs) in enumerate(group_accs.items()):
        ax.plot(k_values, accs, marker="o", linewidth=1.5,
                markersize=6, label=race, color=colors[i % len(colors)])
    ax.axvline(best_k, color=TEXT, linestyle="--",
               linewidth=1.5, label=f"Best k={best_k}")
    ax.set_xlabel("k", color=TEXT)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_title("Per-Group Accuracy by k", color=TEXT, fontweight="bold")
    ax.set_xticks(k_values)
    ax.legend(fontsize=7)
    ax.grid()

    plt.tight_layout()
    plt.savefig("k_optimization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: k_optimization.png")
    return best_k

# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    train = pd.read_csv("train_labels.csv")
    val   = pd.read_csv("val_labels.csv")

    for df in [train, val]:
        df["service_test"] = (df["service_test"].astype(str)
                              .str.strip().str.lower()
                              .map({"true": True, "false": False}))

    train = encode_features(train)
    val   = encode_features(val)
    train_smote = balance_smote(train)

    k_values, overall_accs, group_accs, best_k = find_best_k(
        train_smote, val,
        k_values=[1, 3, 5, 7, 10, 15, 20]
    )
    plot_k_optimization(k_values, overall_accs, group_accs, best_k)