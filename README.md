# Demographic Bias in Facial Recognition Systems
**Course:** Logics for AI — Module 3: Data Bias and Trust  
**Format:** Data/Algorithm Analysis  
**Author:** Camilla Cima  
**Deadline:** May 8, 2026

---

## Project Overview

This project is a data and algorithm analysis investigating a precise and empirically grounded question about fairness in machine learning systems: **does the demographic distribution of a training dataset systematically influence the accuracy of a facial recognition model across racial groups?** And if so, **can rebalancing the training data mitigate this bias?**

The question connects directly to our digital society: data quality, representation bias, algorithmic bias, and trustworthiness of AI systems are central concerns in the responsible deployment of AI. The analysis uses the **FairFace dataset** and compares three classifiers (Logistic Regression, K-Nearest Neighbors, Random Forest) across three training strategies (imbalanced, undersampling, SMOTE). Fairness is evaluated using manually implemented metrics (demographic parity, accuracy gap, representation rate) and per-group confusion matrices. 

---

## Research Question

> *Given a training set with unequal demographic representation, does a model's accuracy differ significantly across racial groups? Does balancing the training distribution improve accuracy for underrepresented groups? And does the choice of classifier architecture affect fairness independently of the training strategy?*

---

## Data Context

### Source and Description

**FairFace** (Karkkainen & Joo, 2021) is a publicly available facial image dataset specifically designed to address racial balance in face attribute datasets. It was constructed by crawling the YFCC-100M Flickr dataset and manually annotating images for race, gender, and age.

- Source: Kaggle (https://www.kaggle.com/datasets/fairface)
- 86,744 training images, 10,954 validation images
- Labels per image: `file`, `age`, `gender`, `race`, `service_test`
- Racial groups: White, Black, Indian, East Asian, Latino_Hispanic, Southeast Asian, Middle Eastern

### The `service_test` Column

The column `service_test` (True/False) is the prediction target used throughout this project. It represents the binary outcome of a facial recognition service applied to each image. **This assumption is made explicit here because it is not fully documented in the original dataset**: the FairFace paper does not specify which facial recognition service produced these labels, under what conditions, or what True/False specifically means (e.g., successful identification, face detection, or attribute classification). This uncertainty is a documented limitation of the analysis.

### Representation

The training set is not uniformly distributed across racial groups:

| Race | Count | Proportion |
|---|---|---|
| White | ~16,568 | 19.1% |
| Latino_Hispanic | ~13,368 | 15.4% |
| Indian | ~12,317 | 14.2% |
| East Asian | ~12,317 | 14.2% |
| Black | ~12,230 | 14.1% |
| Southeast Asian | ~10,758 | 12.4% |
| Middle Eastern | ~9,198 | 10.6% |

White faces represent nearly twice as many samples as Middle Eastern faces. Age distribution is also heavily skewed: 20–29 year olds account for the largest group (~29.4%), while children (0–2, ~1.1%) and elderly (70+, ~0.9%) are severely underrepresented.

### Assumptions

The following assumptions are made explicit, as they are foundational to the validity of the analysis:

1. **`service_test` reflects a real facial recognition system output.** True/False are treated as the binary outcome of a facial recognition service. If this label encodes something else (e.g., image quality), the analysis measures a different phenomenon than intended.
2. **Labels are accurate.** Race, gender, and age labels are assumed to be correctly annotated. Annotation errors would introduce label noise that affects all metrics.
3. **The validation set is representative.** The 10,954 validation images are assumed to be drawn from the same distribution as the training set and are representative of real-world deployment conditions.
4. **Encoded features are adequate proxies.** Race, gender, and age are encoded as integers and used as classifier features. This assumes that demographic category membership, as a tabular feature, is sufficient to predict `service_test`. This is a strong simplification compared to using raw image features.

---

## Types of Bias

This project identifies and analyzes three distinct types of bias, each operating at a different level of the machine learning pipeline:

### 1. Sampling Bias

Sampling bias occurs when the process by which data is collected systematically over- or under-represents certain groups. In FairFace, although the dataset was explicitly designed to be more balanced than predecessors (e.g., CelebA, which is predominantly White), it still exhibits sampling bias: White faces account for 19.1% of training data while Middle Eastern faces account for only 10.6%. This gap reflects the demographic composition of its users, who are predominantly from Western countries.

**In this analysis**: sampling bias is documented in Step 1 (distribution analysis) and quantified by the representation rate metric.

### 2. Representation Bias

Representation bias is a consequence of sampling bias: when a group is underrepresented in training data, the model has fewer examples from which to learn patterns for that group. This leads to higher variance in predictions for minority groups and systematically lower accuracy. Representation bias is distinct from sampling bias in that it refers specifically to the effect of unequal representation on model learning, not just on data collection.

**In this analysis**: representation bias is the central hypothesis of the project. The project tests whether underrepresented groups (Middle Eastern, Southeast Asian) show lower accuracy, and whether rebalancing (undersample, SMOTE) mitigates this.

### 3. Algorithmic Bias

Algorithmic bias refers to systematic errors introduced or amplified by the learning algorithm itself, independent of the data distribution. Different algorithms exhibit different susceptibility to class imbalance:

- **Logistic Regression** is highly susceptible to algorithmic bias from imbalance: it optimizes a global loss function dominated by the majority class, causing it to systematically underperform on minority groups even when representation bias is corrected.
- **KNN** is less susceptible because its decisions are local, in fact it classifies based on nearby samples regardless of global class proportions.
- **Random Forest** is intrinsically robust to algorithmic bias from imbalance because bagging averages out majority-class dominance across trees.

**In this analysis**: algorithmic bias is demonstrated by comparing the three classifiers on the same imbalanced dataset and showing that their per-group accuracy patterns differ significantly even before any balancing intervention.

---

## Dataset Benchmarking

The FairFace dataset is benchmarked against standard fairness criteria before any modeling:

**Demographic Parity of `service_test`**: the positive rate (True) varies significantly across groups:
- Middle Eastern: 0.620 — highest positive rate
- White: 0.350 — lowest positive rate
- Max gap: **0.270**

This gap indicates that the underlying facial recognition service whose output is captured in `service_test` already exhibits demographic parity violations: the system returns True at very different rates for different racial groups, independently of any classifier trained in this project. This is a critical finding since the bias is not only in the training data distribution but also in the labels themselves.

**Accuracy Gap**: deviation from the overall mean (0.464):
- Middle Eastern: +0.156 (most favored by the system)
- White: −0.114 (most penalized by the system)

The result is counterintuitive: the most underrepresented group receives the highest positive prediction rate. This suggests that the original facial recognition service may itself have been trained on different data or with different objectives. It is a finding that warrants caution in interpretation.

---

## Project Structure

```
FairFace-2/
├── data_loader.py       # Loads CSVs, prints distributions, analyzes value counts
├── metrics.py           # Computes demographic parity, accuracy gap, representation rate
├── visualize.py         # Generates all visualizations with pink palette
├── classifier.py        # Trains LR, KNN (k=10), RF with 3 training strategies
├── optimization.py      # Finds optimal k for KNN via accuracy curve
├── confusion.py         # Plots confusion matrix per racial group (best model)
├── main.py              # Runs full pipeline in sequence
├── [output plots]       # .png files generated by running the scripts
│
│ NOTE: Dataset files (train_labels.csv, val_labels.csv, train/, val/)
│ are not included in this repository due to size constraints.
│ Download FairFace from: https://www.kaggle.com/datasets/fairface
│ and place them in the root folder before running.
```

---

## What I Wrote vs External Libraries

### Written entirely by me (from scratch):

- **`demographic_parity()`** in `metrics.py`  
  Computes the fraction of `service_test = True` outcomes per demographic group using pandas `groupby` and `mean`. In a fair system, this rate should be equal across groups: this is the formal definition of demographic parity. A large max gap indicates systematic bias in the positive prediction rate.

- **`accuracy_gap()`** in `metrics.py`  
  Computes each group's deviation from the overall mean positive rate. Formally: `gap(g) = P(ŷ=1 | group=g) - P(ŷ=1)`. Positive values mean the group receives more positive predictions than average; negative values indicate systematic underservice.

- **`representation_rate()`** in `metrics.py`  
  Computes the proportional size of each group using `value_counts(normalize=True)`. Documents input imbalance before any modeling.

- **`balance_undersample()`** in `classifier.py`  
  Reduces each racial group to the size of the smallest group (~9,000 samples) using pandas `groupby.apply(sample)`. Discards ~83,000 samples. Used as a simple baseline balancing strategy.

- **`balance_smote()`** in `classifier.py`  
  Wraps `SMOTE` from `imbalanced-learn`: extracts features, applies SMOTE oversampling, and reconstructs a dataframe. SMOTE generates synthetic minority samples by interpolating between existing samples in feature space. 

- **`train_and_evaluate()`** in `classifier.py`  
  Accepts any sklearn-compatible classifier, fits it on the training set, predicts on the validation set, computes overall accuracy, and returns per-group accuracy for all racial groups. Fully reusable across all three classifiers and training strategies.

- **`plot_comparison()`** in `classifier.py`  
  Generates grouped bar charts comparing per-group accuracy across training strategies. Accepts `{label: group_acc_dict}` for flexible reuse. Full matplotlib implementation.

- **`find_best_k()`** in `optimization.py`  
  Iterates over k values `[1, 3, 5, 7, 10, 15, 20]`, trains KNN for each, records overall and per-group accuracy. Enables fairness-aware k selection.

- **`plot_k_optimization()`** in `optimization.py`  
  Two-panel plot: overall accuracy by k (left) and per-group accuracy curves by k (right). The right panel enables visual inspection of whether k choice disproportionately affects specific groups.

- **`plot_confusion_matrices()`** in `confusion.py`  
  Computes a 2×2 confusion matrix per racial group using sklearn's `confusion_matrix`, visualizes with `imshow` and RdPu colormap. Each cell shows count and percentage of total predictions.

- All visualization functions in `visualize.py` and full pipeline in `main.py`.

### External libraries used:

| Library | Use |
|---|---|
| `pandas` | CSV loading, groupby, value_counts, sampling |
| `numpy` | Array indexing, argmax |
| `matplotlib` | All visualizations, imshow, colorbar |
| `scikit-learn` | LabelEncoder, classifiers, accuracy_score, confusion_matrix |
| `imbalanced-learn` | SMOTE implementation |

---

## Methodology

### Step 1 — Distribution Analysis (`data_loader.py` + `visualize.py`)

`data_loader.py` loads both CSV files, prints sample counts, column names, and raw value counts for race, gender, and age. `visualize.py` generates bar chart visualizations. This step establishes the empirical basis for the bias hypothesis: the gap of nearly 9 percentage points between White (19.1%) and Middle Eastern (10.6%) constitutes representation bias: a structural property of the dataset that precedes any modeling decision.

### Step 2 — Fairness Metrics (`metrics.py`)

Three metrics are computed manually:

**Demographic Parity**: positive rate per group. Max gap = **0.270** (Middle Eastern 0.620 vs White 0.350).

**Accuracy Gap**: `gap(g) = P(ŷ=1 | group=g) - P(ŷ=1)`. White: −0.114, Middle Eastern: +0.156. Counterintuitively, the least represented group receives the highest positive rate suggesting the original service's own bias. 

**Representation Rate**: confirms input imbalance, in fact White is 1.8× more represented than Middle Eastern.

### Step 3 — Feature Encoding

Race, gender, and age are encoded as integers using `LabelEncoder`. Each column uses a separate encoder instance. This is a necessary simplification: demographic labels are treated as nominal features, not ordinal ones. The encoding makes features available to all three classifiers while keeping the pipeline transparent and interpretable.

### Step 4 — Training Strategies

**Imbalanced**: original dataset as-is. Baseline reflecting real-world conditions.

**Undersample**: each group reduced to ~9,000 samples. Eliminates representation bias but discards 83,000 samples.

**SMOTE**: generates synthetic minority samples by interpolating between k=5 nearest neighbors in feature space. Superior to undersampling because: (1) preserves all original data, (2) creates genuinely new samples rather than duplicates, (3) reduces majority bias more effectively. Applied only to training, the validation set is never modified.

### Step 5 — Classifiers (`classifier.py`)

**Logistic Regression**: global linear model, most sensitive to imbalance because majority class dominates gradient updates.

**KNN (k=10)**: local distance-based model, inherently more robust to global imbalance because decisions are determined by nearby samples regardless of overall class proportions.

**Random Forest (100 trees)**: ensemble of trees trained on random bootstrap subsamples (bagging). Majority-class bias in individual trees is averaged out across the ensemble, which is intrinsically robust to imbalance.

### Step 6 — KNN Hyperparameter Optimization (`optimization.py`)

k tested over {1, 3, 5, 7, 10, 15, 20} on SMOTE-balanced training set.

**Why k=10 and not k=20 (highest overall accuracy)?**
k=20 gains only +0.002 over k=10. More importantly, larger k reintroduces neighborhood-level majority bias: with 20 neighbors, predictions are increasingly dominated by whichever group is densest in feature space. Per-group accuracy curves stabilize between k=7 and k=10 with no meaningful improvement beyond. k=10 is the methodologically defensible choice: near-optimal accuracy without majority bias reintroduction.

### Step 7 — Confusion Matrices (`confusion.py`)

For KNN k=10 + SMOTE (best model), a 2×2 confusion matrix per racial group. Shows not just how often the model errs but how, distinguishing false positives (FP) from false negatives (FN) per group. High FN for a group means systematic under-prediction; high FP means systematic over-prediction. These error patterns have different real-world implications.

---

## Trustworthiness Analysis

**Explainability**: Logistic Regression is the most interpretable (coefficients indicate feature weights). KNN is locally interpretable (neighbors can be inspected for any prediction). Random Forest provides global feature importance via `feature_importances_`. All models use explicit demographic labels as features, making inputs fully transparent, unlike image-based models with latent features.

**Reliability**: All random operations use `random_state=42` for full reproducibility. The validation set is fixed and never modified. All three strategies are evaluated on the same validation set for direct comparability.

**Validation**: Held-out validation set (10,954 samples) never used in training or hyperparameter selection. Per-group accuracy provides fairness-aware validation beyond aggregate metrics. Confusion matrices add a second validation layer decomposing errors by type and group.

---

## Results

### Fairness Metrics

Max demographic parity gap: **0.270**. White: 0.350 positive rate. Middle Eastern: 0.620.

### Classifier Comparison

**Logistic Regression — SMOTE effect:**
| Group | Imbalanced | Undersample | SMOTE |
|---|---|---|---|
| Middle Eastern | 0.54 | 0.58 | **0.67** |
| Indian | 0.55 | 0.55 | **0.60** |
| Southeast Asian | 0.48 | 0.46 | **0.51** |
| Latino_Hispanic | 0.57 | 0.55 | 0.59 |
| White | 0.58 | 0.59 | 0.59 |
| Black | 0.58 | 0.52 | 0.51 |
| East Asian | 0.43 | 0.43 | 0.42 |

**KNN (k=10)**: Middle Eastern 0.81 across all strategies. Less sensitive to balancing than LR.

**Random Forest**: most uniform accuracy (0.59–0.81), nearly identical across all three strategies.

### Confusion Matrix Findings (KNN k=10 + SMOTE)

- **Middle Eastern**: 51.2% TP, 5.1% FP — highest confidence and accuracy
- **White**: 54.3% TN — most often predicted False despite being majority group
- **Southeast Asian**: most balanced matrix — highest model uncertainty
- **East Asian, Indian**: high TN but many FN — systematic under-prediction of positive cases

### Answer to the Research Question

Yes, training set imbalance produces measurable accuracy disparities. SMOTE substantially improves accuracy for underrepresented groups with Logistic Regression. For KNN and Random Forest the effect is smaller due to their inherent robustness. **Model architecture matters as much as data balancing**: fairness interventions must address both data-level and model-level factors simultaneously.

---

## Limitations and Ethical Implications

### Limitations

- **Circularity**: features (race, gender, age) are identical to the group variables analyzed. The model predicts `service_test` from demographic identity rather than image-level features.
- **Tabular SMOTE**: synthetic samples are generated in encoded feature space, not image space. Real image augmentation would be more rigorous.
- **`service_test` uncertainty**: the origin of this label is undocumented. The interpretation used in this project may be incorrect. 
- **LabelEncoder**: arbitrary integer codes assigned to racial categories may introduce spurious distances in KNN.
- **Generalizability**: results are specific to FairFace and these classifiers.

### Ethical Implications

The findings of this project have direct implications for the real-world deployment of facial recognition systems:

**Accuracy disparities translate to unequal treatment**. In a real facial recognition system, a group with 0.50 accuracy (Southeast Asian in some configurations) experiences correct outcomes only half the time, equivalent to a coin flip. A group with 0.81 accuracy (Middle Eastern with KNN) experiences a system that works reliably. This inequality in system performance constitutes a form of technological discrimination, where access to reliable automated services is unevenly distributed across racial groups.

**The most underrepresented group is not necessarily the most disadvantaged in output**. Middle Eastern faces, which are the least represented in training, receive the highest positive prediction rate (0.620) and the highest classifier accuracy (0.81 with KNN). This counterintuitive result suggests that representation bias and output bias are not linearly related, and that interventions targeting only training data distribution may be insufficient or misdirected.

**Balancing is necessary but not sufficient**. SMOTE significantly improves accuracy for underrepresented groups when using Logistic Regression, but has minimal effect on KNN and Random Forest. This means that deploying a "balanced" dataset with the wrong classifier architecture may still produce biased outcomes. Ethical deployment of facial recognition systems requires co-optimization of data collection, balancing strategy, and model architecture.

**Transparency about `service_test` is an ethical obligation**. The undocumented nature of the `service_test` label means that any system trained on it inherits unknown biases from an unknown source. In a real deployment context, this would be unacceptable: regulatory frameworks such as the EU AI Act require documentation of training data provenance and labeling methodology for high-risk AI systems.

---

## How to Run

```bash
# Install dependencies
pip3 install numpy pandas matplotlib scikit-learn imbalanced-learn

# Run full pipeline
python3 main.py

# Or run individual scripts
python3 data_loader.py     # Distribution analysis
python3 metrics.py         # Fairness metrics
python3 visualize.py       # Visualization plots
python3 classifier.py      # LR + KNN + RF comparison
python3 optimization.py    # KNN k optimization
python3 confusion.py       # Confusion matrices
```

---

## Output Files

| File | Description |
|---|---|
| `distribution.png` | Demographic distribution of training set by race, gender, age |
| `demographic_parity.png` | Positive rate per group vs overall mean |
| `accuracy_gap.png` | Per-group deviation from overall accuracy |
| `lr_comparison.png` | Logistic Regression accuracy by group and training strategy |
| `knn_comparison.png` | KNN (k=10) accuracy by group and training strategy |
| `rf_comparison.png` | Random Forest accuracy by group and training strategy |
| `k_optimization.png` | Overall and per-group accuracy across k values (k=1 to k=20) |
| `confusion_matrices.png` | Per-group confusion matrices for best model (KNN k=10 + SMOTE) |
