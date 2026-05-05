import pandas as pd
import numpy as np

# ── DEMOGRAPHIC PARITY ─────────────────────────────────────
# For each group, what fraction gets service_test = True?
# In a fair system, this should be equal across groups.
def demographic_parity(df, group_col):
    """
    Demographic Parity: measures whether the positive outcome rate
    (service_test = True) is equal across demographic groups.
    A large spread indicates bias.
    """
    rates = df.groupby(group_col)["service_test"].mean()
    print(f"\n── Demographic Parity by {group_col} ──")
    print(rates.round(3))
    print(f"Max gap: {(rates.max() - rates.min()):.3f}")
    return rates

# ── ACCURACY GAP ───────────────────────────────────────────
# If we treat service_test as a binary classifier output,
# how does accuracy vary across groups?
def accuracy_gap(df, group_col):
    """
    Accuracy Gap: compares the mean positive prediction rate
    per group. A high gap means the system treats groups unequally.
    """
    group_rates = df.groupby(group_col)["service_test"].mean()
    overall = df["service_test"].mean()
    gap = group_rates - overall
    print(f"\n── Accuracy Gap vs Overall Mean by {group_col} ──")
    print(gap.round(3))
    return gap

# ── REPRESENTATION RATE ────────────────────────────────────
# How represented is each group in the dataset?
def representation_rate(df, group_col):
    """
    Representation Rate: what percentage of the dataset
    does each group occupy? Underrepresentation can cause bias.
    """
    rates = df[group_col].value_counts(normalize=True)
    print(f"\n── Representation Rate by {group_col} ──")
    print(rates.round(3))
    return rates

# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("train_labels.csv")
    df["service_test"] = df["service_test"].astype(str).str.strip().str.lower().map({"true": True, "false": False})

    for col in ["race", "gender", "age"]:
        demographic_parity(df, col)
        accuracy_gap(df, col)
        representation_rate(df, col)