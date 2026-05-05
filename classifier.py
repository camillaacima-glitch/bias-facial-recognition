import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ── PALETTE ────────────────────────────────────────────────
PINK       = "#F48FB1"
DARK_PINK  = "#C2185B"
LIGHT_PINK = "#FCE4EC"
MID_PINK   = "#F06292"
SOFT_PINK  = "#F8BBD0"
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

# ── ENCODE FEATURES ────────────────────────────────────────
def encode_features(df):
    le_gender = LabelEncoder()
    le_race   = LabelEncoder()
    le_age    = LabelEncoder()
    df = df.copy()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    df["race_enc"]   = le_race.fit_transform(df["race"])
    df["age_enc"]    = le_age.fit_transform(df["age"])
    return df

# ── BALANCE: UNDERSAMPLE ───────────────────────────────────
# Reduces each group to the size of the smallest group.
# Simple but loses data.
def balance_undersample(df):
    min_count = df["race"].value_counts().min()
    balanced = df.groupby("race").apply(
        lambda x: x.sample(min_count, random_state=42)
    ).reset_index(drop=True)
    print(f"\nUndersample: {len(balanced)} samples ({min_count} per group)")
    return balanced

# ── BALANCE: SMOTE ─────────────────────────────────────────
# Generates synthetic samples for underrepresented groups.
# More sophisticated than undersampling — preserves all original data.
def balance_smote(df):
    features = ["gender_enc", "race_enc", "age_enc"]
    X = df[features]
    y = df["service_test"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_smote = pd.DataFrame(X_res, columns=features)
    df_smote["service_test"] = y_res
    print(f"\nSMOTE: {len(df_smote)} samples (from {len(df)} original)")
    return df_smote

# ── TRAIN AND EVALUATE ─────────────────────────────────────
# Trains a given model, evaluates per-group accuracy on val set.
def train_and_evaluate(train_df, val_df, model, label="Model"):
    features = ["gender_enc", "race_enc", "age_enc"]
    target   = "service_test"

    X_train = train_df[features]
    y_train = train_df[target]
    X_val   = val_df[features]
    y_val   = val_df[target]

    model.fit(X_train, y_train)

    val_df = val_df.copy()
    val_df["predicted"] = model.predict(X_val)

    overall_acc = accuracy_score(y_val, val_df["predicted"])
    print(f"\n── {label} ──")
    print(f"Overall accuracy: {overall_acc:.3f}")

    group_acc = {}
    for race in sorted(val_df["race"].unique()):
        subset = val_df[val_df["race"] == race]
        acc = accuracy_score(subset[target], subset["predicted"])
        group_acc[race] = acc
        print(f"  {race}: {acc:.3f}")

    return group_acc, overall_acc

# ── PLOT COMPARISON ────────────────────────────────────────
def plot_comparison(results, title, filename):
    """
    results: dict of {label: group_acc_dict}
    """
    set_style()
    groups = sorted(list(results.values())[0].keys())
    x = np.arange(len(groups))
    width = 0.25
    colors = [DARK_PINK, MID_PINK, SOFT_PINK]

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=TEXT)

    for i, (label, group_acc) in enumerate(results.items()):
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset,
                      [group_acc[g] for g in groups],
                      width, label=label,
                      color=colors[i], edgecolor=TEXT)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", fontsize=7, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")

# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    train = pd.read_csv("train_labels.csv")
    val   = pd.read_csv("val_labels.csv")

    for df in [train, val]:
        df["service_test"] = (df["service_test"].astype(str)
                              .str.strip().str.lower()
                              .map({"true": True, "false": False}))

    # Encode
    train = encode_features(train)
    val   = encode_features(val)

    # Prepare training sets
    train_under  = balance_undersample(train)
    train_smote  = balance_smote(train)

    # ── LOGISTIC REGRESSION ────────────────────────────────
    print("\n════ LOGISTIC REGRESSION ════")
    lr_imb,   _ = train_and_evaluate(train,       val, LogisticRegression(max_iter=1000, random_state=42), "LR — Imbalanced")
    lr_under, _ = train_and_evaluate(train_under, val, LogisticRegression(max_iter=1000, random_state=42), "LR — Undersample")
    lr_smote, _ = train_and_evaluate(train_smote, val, LogisticRegression(max_iter=1000, random_state=42), "LR — SMOTE")

    plot_comparison(
        {"Imbalanced": lr_imb, "Undersample": lr_under, "SMOTE": lr_smote},
        "Logistic Regression — Per-Group Accuracy by Training Strategy",
        "lr_comparison.png"
    )

    # ── KNN ────────────────────────────────────────────────
    print("\n════ KNN (k=5) ════")
    knn_imb,   _ = train_and_evaluate(train,       val, KNeighborsClassifier(n_neighbors=10), "KNN — Imbalanced")
    knn_under, _ = train_and_evaluate(train_under, val, KNeighborsClassifier(n_neighbors=10), "KNN — Undersample")
    knn_smote, _ = train_and_evaluate(train_smote, val, KNeighborsClassifier(n_neighbors=10), "KNN — SMOTE")

    plot_comparison(
        {"Imbalanced": knn_imb, "Undersample": knn_under, "SMOTE": knn_smote},
        "KNN — Per-Group Accuracy by Training Strategy",
        "knn_comparison.png"
    )
# ── RANDOM FOREST ──────────────────────────────────────────
print("\n════ RANDOM FOREST ════")
rf_imb,   _ = train_and_evaluate(train,       val, RandomForestClassifier(n_estimators=100, random_state=42), "RF — Imbalanced")
rf_under, _ = train_and_evaluate(train_under, val, RandomForestClassifier(n_estimators=100, random_state=42), "RF — Undersample")
rf_smote, _ = train_and_evaluate(train_smote, val, RandomForestClassifier(n_estimators=100, random_state=42), "RF — SMOTE")

plot_comparison(
    {"Imbalanced": rf_imb, "Undersample": rf_under, "SMOTE": rf_smote},
    "Random Forest — Per-Group Accuracy by Training Strategy",
    "rf_comparison.png"
)