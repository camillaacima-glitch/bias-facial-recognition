import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    features = ["gender_enc", "race_enc", "age_enc"]
    X = df[features]
    y = df["service_test"]
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    df_smote = pd.DataFrame(X_res, columns=features)
    df_smote["service_test"] = y_res
    return df_smote

# ── PLOT CONFUSION MATRIX PER GROUP ───────────────────────
# For each racial group, computes and plots the confusion matrix
# using the best model: KNN k=10 trained on SMOTE-balanced data.
# This shows WHERE the model makes errors, not just how often.
def plot_confusion_matrices(val_df, predictions):
    set_style()
    groups = sorted(val_df["race"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Confusion Matrix per Racial Group\n(KNN k=10, SMOTE-balanced training)",
                 fontsize=14, fontweight="bold", color=TEXT)
    axes = axes.flatten()

    for i, group in enumerate(groups):
        subset = val_df[val_df["race"] == group].copy()
        preds  = predictions[val_df["race"] == group]

        cm = confusion_matrix(subset["service_test"], preds)

        ax = axes[i]
        im = ax.imshow(cm, cmap="RdPu")

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: False", "Pred: True"], fontsize=8)
        ax.set_yticklabels(["True: False", "True: True"], fontsize=8)
        ax.set_title(group, color=TEXT, fontweight="bold", fontsize=10)

        # Values inside cells
        for row in range(2):
            for col in range(2):
                val = cm[row, col]
                total = cm.sum()
                ax.text(col, row, f"{val}\n({val/total:.1%})",
                        ha="center", va="center",
                        color="white" if cm[row, col] > cm.max()/2 else TEXT,
                        fontsize=9, fontweight="bold")

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplot
    for j in range(len(groups), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: confusion_matrices.png")

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

    # Best model: KNN k=10 + SMOTE
    train_smote = balance_smote(train)

    features = ["gender_enc", "race_enc", "age_enc"]
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(train_smote[features], train_smote["service_test"])

    predictions = model.predict(val[features])

    plot_confusion_matrices(val, predictions)
