Project 2: Data Classification Using AI

**DecodeLabs — AI Industrial Training Kit (Batch 2026)**

Overview

This project implements a supervised learning pipeline to classify iris flowers into one of three species based on their physical measurements. It covers the complete machine learning workflow, from raw data to a validated, evaluated model, using the K-Nearest Neighbors (KNN) algorithm.

Objective

Build a basic classification model using a small, well-known dataset, and demonstrate the full ML pipeline: load, explore, split, scale, train, predict, and evaluate.

 Dataset: Iris Benchmark

| Property | Value |
|---|---|
| Samples | 150 (balanced) |
| Classes | 3 — Setosa, Versicolor, Virginica |
| Features | 4 — sepal length, sepal width, petal length, petal width (cm) |
| Source | `sklearn.datasets.load_iris()` |

Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (`sklearn`)

Pipeline

1. **Load Data** — `load_iris()` loaded into a pandas DataFrame
2. **Explore Data (EDA)** — `head()`, `shape`, `isnull().sum()`, `describe()`, `info()`
3. **Train-Test Split** — 80/20 split via `train_test_split()`, stratified to preserve class balance
4. **Feature Scaling** — `StandardScaler` (mean = 0, std = 1) so distance-based KNN is not biased by feature scale
5. **Model Training** — `KNeighborsClassifier`
6. **Prediction and Evaluation** — Accuracy, Confusion Matrix, Precision/Recall/F1 via `classification_report`
7. **Hyperparameter Tuning** — Tested `k = 1` to `10` to find the optimal number of neighbors

 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py
```

Results

**Accuracy (k=5): 93.3%**

**Confusion Matrix:**
```
[[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]
```
- Setosa: 10/10 correct
- Versicolor: 10/10 correct
- Virginica: 8/10 correct (2 misclassified as Versicolor)

**Classification Report:**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Setosa (0) | 1.00 | 1.00 | 1.00 |
| Versicolor (1) | 0.83 | 1.00 | 0.91 |
| Virginica (2) | 1.00 | 0.80 | 0.89 |

**K-Value Tuning:**

| k | Accuracy |
|---|---|
| 1 | 0.967 |
| 2 | 0.933 |
| 3 | 0.933 |
| 4 | 0.933 |
| 5 | 0.933 |
| 6 | 0.933 |
| 7 | 0.967 |
| 8 | 0.933 |
| 9 | 0.967 |
| 10 | 0.967 |

Best-performing values: k = 1, 7, 9, and 10 (96.7% accuracy).

Key Learnings

- Accuracy alone can be misleading at the class level. The confusion matrix showed that all errors came from confusing Virginica with Versicolor, two species known to overlap in petal measurements.
- Feature scaling is important for distance-based algorithms like KNN, since unscaled features with larger ranges can dominate distance calculations.
- Choosing k involves a trade-off: very low values risk overfitting to noise, while very high values risk underfitting. Testing a range of values helps identify the best fit for a given dataset.

## Author

Prepared as part of DecodeLabs' AI Industrial Training Kit, Project 2.