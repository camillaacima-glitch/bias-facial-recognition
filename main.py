"""
Demographic Bias in Facial Recognition — Main Pipeline
=======================================================
This script runs the full analysis pipeline in sequence:
1. Load and visualize dataset distributions
2. Compute fairness metrics (demographic parity, accuracy gap)
3. Train and compare classifiers (LR, KNN, RF) with three
   training strategies (imbalanced, undersample, SMOTE)
4. Optimize KNN hyperparameter k
5. Plot confusion matrices for best model (KNN k=10 + SMOTE)

Author: Camilla Cima
Course: Logics for AI — Prof. Primiero
"""

import subprocess
import sys

def run(script):
    print(f"\n{'='*50}")
    print(f"Running {script}...")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script], check=True)
    return result

if __name__ == "__main__":
    scripts = [
        "data_loader.py",
        "metrics.py",
        "visualize.py",
        "classifier.py",
        "optimization.py",
        "confusion.py",
    ]

    for script in scripts:
        run(script)

    print("\n✓ Pipeline complete. All outputs saved as .png in current directory.")
