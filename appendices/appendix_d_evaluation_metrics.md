# Appendix D: Evaluation Metrics Reference

This appendix provides a comprehensive reference for all evaluation metrics used in machine learning. Understanding when and how to use each metric is crucial for properly assessing model performance and making informed decisions about model selection and optimization.

---

## D.1 Classification Metrics Summary

Classification metrics evaluate how well a model predicts categorical outcomes. The choice of metric depends on the problem type, class distribution, and business objectives.

### D.1.1 Basic Classification Metrics

#### Confusion Matrix Foundation
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, classes=None, title='Confusion Matrix'):
    """
    Plot confusion matrix with proper formatting
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm

# For binary classification confusion matrix:
# [[TN, FP],
#  [FN, TP]]
```

#### Accuracy
- **Definition:** Ratio of correctly predicted observations to total observations
- **Formula:** `(TP + TN) / (TP + TN + FP + FN)`
- **Range:** [0, 1] (higher is better)

```python
from sklearn.metrics import accuracy_score

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# When to use:
# ✓ Balanced datasets
# ✓ Equal misclassification costs
# ✗ Imbalanced datasets
# ✗ When specific class performance matters
```

#### Precision
- **Definition:** Ratio of correctly predicted positive observations to total predicted positive
- **Formula:** `TP / (TP + FP)`
- **Range:** [0, 1] (higher is better)

```python
from sklearn.metrics import precision_score

def calculate_precision(y_true, y_pred, average='binary'):
    return precision_score(y_true, y_pred, average=average)

# When to use:
# ✓ When false positives are costly
# ✓ Spam detection (don't want to mark good emails as spam)
# ✓ Medical diagnosis (don't want false alarms)
```

#### Recall (Sensitivity, True Positive Rate)
- **Definition:** Ratio of correctly predicted positive observations to actual positive class
- **Formula:** `TP / (TP + FN)`
- **Range:** [0, 1] (higher is better)

```python
from sklearn.metrics import recall_score

def calculate_recall(y_true, y_pred, average='binary'):
    return recall_score(y_true, y_pred, average=average)

# When to use:
# ✓ When false negatives are costly
# ✓ Disease detection (don't want to miss cases)
# ✓ Fraud detection (don't want to miss fraudulent transactions)
```

#### Specificity (True Negative Rate)
- **Definition:** Ratio of correctly predicted negative observations to actual negative class
- **Formula:** `TN / (TN + FP)`
- **Range:** [0, 1] (higher is better)

```python
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

# When to use:
# ✓ When correctly identifying negatives is important
# ✓ Screening tests (identifying healthy individuals)
```

#### F1-Score
- **Definition:** Harmonic mean of precision and recall
- **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
- **Range:** [0, 1] (higher is better)

```python
from sklearn.metrics import f1_score

def calculate_f1(y_true, y_pred, average='binary'):
    return f1_score(y_true, y_pred, average=average)

# When to use:
# ✓ Imbalanced datasets
# ✓ When you need balance between precision and recall
# ✓ Single metric for model comparison
```

### D.1.2 Advanced Classification Metrics

#### F-Beta Score
- **Definition:** Weighted harmonic mean of precision and recall
- **Formula:** `(1 + β²) * (Precision * Recall) / (β² * Precision + Recall)`

```python
from sklearn.metrics import fbeta_score

def calculate_fbeta(y_true, y_pred, beta=1.0):
    return fbeta_score(y_true, y_pred, beta=beta)

# Beta values:
# β < 1: Emphasizes precision
# β > 1: Emphasizes recall
# β = 1: Equal weight (F1-score)

# Example usage:
f05_score = calculate_fbeta(y_true, y_pred, beta=0.5)  # Favor precision
f2_score = calculate_fbeta(y_true, y_pred, beta=2.0)   # Favor recall
```

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Definition:** Area under the ROC curve (TPR vs FPR)
- **Range:** [0, 1] (higher is better, 0.5 = random)

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return auc_score

# When to use:
# ✓ Binary classification
# ✓ When you want threshold-independent metric
# ✓ Balanced datasets
# ✗ Highly imbalanced datasets
```

#### Precision-Recall AUC
- **Definition:** Area under the Precision-Recall curve
- **Range:** [0, 1] (higher is better)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return ap_score

# When to use:
# ✓ Imbalanced datasets
# ✓ When positive class is more important
# ✓ Information retrieval tasks
```

#### Log Loss (Cross-Entropy Loss)
- **Definition:** Negative log-likelihood of true labels given probabilistic predictions
- **Formula:** `-Σ(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))`
- **Range:** [0, ∞] (lower is better)

```python
from sklearn.metrics import log_loss

def calculate_log_loss(y_true, y_pred_proba):
    return log_loss(y_true, y_pred_proba)

# When to use:
# ✓ When you have probability predictions
# ✓ Calibrated probability estimates are important
# ✓ Multi-class classification
```

#### Cohen's Kappa
- **Definition:** Agreement between predicted and actual classifications, accounting for chance
- **Formula:** `(p_o - p_e) / (1 - p_e)`
- **Range:** [-1, 1] (higher is better, 0 = random)

```python
from sklearn.metrics import cohen_kappa_score

def calculate_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

# Interpretation:
# < 0: Less than chance agreement
# 0.01-0.20: Slight agreement
# 0.21-0.40: Fair agreement
# 0.41-0.60: Moderate agreement
# 0.61-0.80: Substantial agreement
# 0.81-1.00: Almost perfect agreement
```

#### Matthews Correlation Coefficient (MCC)
- **Definition:** Correlation coefficient between observed and predicted classifications
- **Formula:** `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`
- **Range:** [-1, 1] (higher is better, 0 = random)

```python
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

# When to use:
# ✓ Imbalanced datasets
# ✓ Binary classification
# ✓ When all confusion matrix elements are important
```

### D.1.3 Multi-class Classification Metrics

```python
from sklearn.metrics import classification_report

def comprehensive_classification_report(y_true, y_pred, target_names=None):
    """
    Generate comprehensive classification report
    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    print("="*60)
    print("COMPREHENSIVE CLASSIFICATION REPORT")
    print("="*60)
    
    # Overall metrics
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Average F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Average F1: {report['weighted avg']['f1-score']:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{class_name:<15} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} "
                  f"{int(metrics['support']):<10}")
    
    return report

# Averaging strategies for multi-class:
# - 'micro': Calculate globally (good for imbalanced datasets)
# - 'macro': Calculate per class then average (treats all classes equally)
# - 'weighted': Calculate per class, weighted by support
```

---

## D.2 Regression Metrics Summary

Regression metrics evaluate how well a model predicts continuous numerical values. The choice depends on the scale of your target variable, presence of outliers, and interpretability requirements.

### D.2.1 Basic Regression Metrics

#### Mean Absolute Error (MAE)
- **Definition:** Average of absolute differences between predicted and actual values
- **Formula:** `Σ|y_true - y_pred| / n`
- **Range:** [0, ∞] (lower is better)
- **Units:** Same as target variable

```python
from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# When to use:
# ✓ Robust to outliers
# ✓ Interpretable (same units as target)
# ✓ When all errors are equally important
```

#### Mean Squared Error (MSE)
- **Definition:** Average of squared differences between predicted and actual values
- **Formula:** `Σ(y_true - y_pred)² / n`
- **Range:** [0, ∞] (lower is better)
- **Units:** Squared units of target variable

```python
from sklearn.metrics import mean_squared_error

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# When to use:
# ✓ Penalizes large errors more heavily
# ✓ Differentiable (good for optimization)
# ✗ Sensitive to outliers
# ✗ Units are squared
```

#### Root Mean Squared Error (RMSE)
- **Definition:** Square root of MSE
- **Formula:** `√(Σ(y_true - y_pred)² / n)`
- **Range:** [0, ∞] (lower is better)
- **Units:** Same as target variable

```python
import numpy as np

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# When to use:
# ✓ Most common regression metric
# ✓ Interpretable units
# ✓ Penalizes large errors
# ✗ Sensitive to outliers
```

#### R-squared (Coefficient of Determination)
- **Definition:** Proportion of variance in target variable explained by the model
- **Formula:** `1 - (SS_res / SS_tot)`
- **Range:** (-∞, 1] (higher is better, 1 = perfect fit)

```python
from sklearn.metrics import r2_score

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Interpretation:
# R² = 1: Perfect predictions
# R² = 0: Model performs as well as mean baseline
# R² < 0: Model performs worse than mean baseline

# When to use:
# ✓ Model comparison
# ✓ Proportion of variance explained
# ✗ Doesn't indicate absolute quality
# ✗ Can be misleading with non-linear relationships
```

### D.2.2 Advanced Regression Metrics

#### Adjusted R-squared
- **Definition:** R-squared adjusted for number of features
- **Formula:** `1 - ((1-R²)(n-1)/(n-k-1))`

```python
def calculate_adjusted_r2(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - n_features - 1)
    return adjusted_r2

# When to use:
# ✓ Comparing models with different numbers of features
# ✓ Prevents overfitting due to too many features
```

#### Mean Absolute Percentage Error (MAPE)
- **Definition:** Average of absolute percentage differences
- **Formula:** `(100/n) * Σ|((y_true - y_pred) / y_true)|`
- **Range:** [0, ∞] (lower is better)
- **Units:** Percentage

```python
def calculate_mape(y_true, y_pred, epsilon=1e-8):
    # Add epsilon to avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# When to use:
# ✓ Scale-independent comparison
# ✓ Easy to interpret (percentage)
# ✗ Problematic when y_true contains zeros or small values
# ✗ Asymmetric (over-prediction penalized less)
```

#### Symmetric Mean Absolute Percentage Error (SMAPE)
- **Definition:** Symmetric version of MAPE
- **Formula:** `(100/n) * Σ(|y_pred - y_true| / ((|y_true| + |y_pred|)/2))`

```python
def calculate_smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100

# When to use:
# ✓ Symmetric penalty for over and under-prediction
# ✓ Scale-independent
# ✓ Bounded between 0 and 200%
```

#### Mean Absolute Scaled Error (MASE)
- **Definition:** MAE scaled by naive forecast MAE
- **Formula:** `MAE / MAE_naive`

```python
def calculate_mase(y_true, y_pred, y_train):
    # Calculate naive forecast error (seasonal naive for time series)
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_naive = mean_absolute_error(y_train[1:], y_train[:-1])  # Simple naive
    return mae_model / mae_naive

# Interpretation:
# MASE < 1: Better than naive forecast
# MASE = 1: Same as naive forecast  
# MASE > 1: Worse than naive forecast
```

#### Huber Loss
- **Definition:** Combines MSE and MAE properties
- **Formula:** Quadratic for small errors, linear for large errors

```python
from sklearn.metrics import mean_squared_error

def calculate_huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    condition = residual <= delta
    
    squared_loss = 0.5 * (residual ** 2)
    linear_loss = delta * residual - 0.5 * (delta ** 2)
    
    return np.mean(np.where(condition, squared_loss, linear_loss))

# When to use:
# ✓ Robust to outliers
# ✓ Differentiable everywhere
# ✓ Good balance between MSE and MAE
```

### D.2.3 Comprehensive Regression Evaluation

```python
import pandas as pd
import matplotlib.pyplot as plt

def comprehensive_regression_report(y_true, y_pred, model_name="Model"):
    """
    Generate comprehensive regression evaluation report
    """
    print("="*60)
    print(f"COMPREHENSIVE REGRESSION REPORT - {model_name}")
    print("="*60)
    
    # Calculate all metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    # Display metrics
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape
    }
    
    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.6f}")
    
    # Residual analysis
    residuals = y_true - y_pred
    
    print(f"\nResidual Analysis:")
    print(f"Mean Residual     : {np.mean(residuals):.6f}")
    print(f"Std Residual      : {np.std(residuals):.6f}")
    print(f"Min Residual      : {np.min(residuals):.6f}")
    print(f"Max Residual      : {np.max(residuals):.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual Distribution
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    
    # Q-Q Plot for residual normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Residual Normality)')
    
    plt.tight_layout()
    plt.show()
    
    return metrics
```

---

## D.3 Clustering Evaluation Methods

Clustering evaluation is more challenging than supervised learning because there's no ground truth. We use both internal measures (based on the data itself) and external measures (when true labels are available).

### D.3.1 Internal Evaluation Metrics

#### Silhouette Score
- **Definition:** Measures how well-separated clusters are
- **Range:** [-1, 1] (higher is better)

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

def calculate_silhouette_analysis(X, cluster_labels, n_clusters):
    """
    Comprehensive silhouette analysis
    """
    # Overall silhouette score
    avg_silhouette = silhouette_score(X, cluster_labels)
    
    # Per-sample silhouette scores
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Create silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette plot
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.axvline(x=avg_silhouette, color="red", linestyle="--")
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster label')
    ax1.set_title('Silhouette Plot')
    
    # Scatter plot of clusters
    colors = plt.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
    ax2.set_title('Clustered Data')
    
    plt.show()
    
    print(f"Average Silhouette Score: {avg_silhouette:.4f}")
    
    # Interpretation:
    if avg_silhouette > 0.7:
        print("Strong cluster structure")
    elif avg_silhouette > 0.5:
        print("Reasonable cluster structure")
    elif avg_silhouette > 0.25:
        print("Weak cluster structure")
    else:
        print("No substantial cluster structure")
    
    return avg_silhouette

# When to use:
# ✓ Comparing different clustering algorithms
# ✓ Determining optimal number of clusters
# ✓ Dense, well-separated clusters
# ✗ Different cluster densities or sizes
```

#### Calinski-Harabasz Index (Variance Ratio Criterion)
- **Definition:** Ratio of between-cluster dispersion to within-cluster dispersion
- **Range:** [0, ∞] (higher is better)

```python
from sklearn.metrics import calinski_harabasz_score

def calculate_calinski_harabasz(X, cluster_labels):
    score = calinski_harabasz_score(X, cluster_labels)
    print(f"Calinski-Harabasz Score: {score:.4f}")
    return score

# When to use:
# ✓ Convex clusters
# ✓ Similar cluster sizes
# ✓ Fast computation
```

#### Davies-Bouldin Index
- **Definition:** Average similarity ratio of each cluster with its most similar cluster
- **Range:** [0, ∞] (lower is better)

```python
from sklearn.metrics import davies_bouldin_score

def calculate_davies_bouldin(X, cluster_labels):
    score = davies_bouldin_score(X, cluster_labels)
    print(f"Davies-Bouldin Score: {score:.4f}")
    return score

# When to use:
# ✓ Convex clusters
# ✓ Similar cluster sizes
# ✓ When lower values indicate better clustering
```

#### Inertia (Within-Cluster Sum of Squares)
- **Definition:** Sum of squared distances of samples to cluster centers
- **Range:** [0, ∞] (lower is better)

```python
def calculate_inertia(X, cluster_labels, centroids):
    """
    Calculate within-cluster sum of squares
    """
    inertia = 0
    for i in range(len(centroids)):
        cluster_points = X[cluster_labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[i]) ** 2)
    
    return inertia

def elbow_method_analysis(X, max_k=10):
    """
    Perform elbow method analysis for optimal k
    """
    from sklearn.cluster import KMeans
    
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    return k_range, inertias
```

### D.3.2 External Evaluation Metrics

These metrics are used when true cluster labels are available (for validation purposes).

#### Adjusted Rand Index (ARI)
- **Definition:** Similarity measure between two clusterings, adjusted for chance
- **Range:** [-1, 1] (higher is better, 1 = perfect match)

```python
from sklearn.metrics import adjusted_rand_score

def calculate_ari(true_labels, pred_labels):
    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Interpretation
    if ari > 0.9:
        print("Excellent clustering")
    elif ari > 0.7:
        print("Good clustering")
    elif ari > 0.5:
        print("Moderate clustering")
    else:
        print("Poor clustering")
    
    return ari
```

#### Normalized Mutual Information (NMI)
- **Definition:** Normalized measure of mutual dependence between clusterings
- **Range:** [0, 1] (higher is better)

```python
from sklearn.metrics import normalized_mutual_info_score

def calculate_nmi(true_labels, pred_labels):
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"Normalized Mutual Information: {nmi:.4f}")
    return nmi
```

#### Homogeneity, Completeness, and V-measure
- **Homogeneity:** Each cluster contains only members of a single class
- **Completeness:** All members of a given class are assigned to the same cluster
- **V-measure:** Harmonic mean of homogeneity and completeness

```python
from sklearn.metrics import homogeneity_completeness_v_measure

def calculate_clustering_metrics(true_labels, pred_labels):
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )
    
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"V-measure: {v_measure:.4f}")
    
    return homogeneity, completeness, v_measure
```

### D.3.3 Comprehensive Clustering Evaluation

```python
def comprehensive_clustering_evaluation(X, cluster_labels, true_labels=None, 
                                      centroids=None, n_clusters=None):
    """
    Comprehensive clustering evaluation with all relevant metrics
    """
    print("="*60)
    print("COMPREHENSIVE CLUSTERING EVALUATION")
    print("="*60)
    
    # Internal metrics
    print("Internal Metrics (no ground truth needed):")
    print("-" * 40)
    
    silhouette = silhouette_score(X, cluster_labels)
    calinski = calinski_harabasz_score(X, cluster_labels)
    davies = davies_bouldin_score(X, cluster_labels)
    
    print(f"Silhouette Score      : {silhouette:.4f}")
    print(f"Calinski-Harabasz     : {calinski:.4f}")
    print(f"Davies-Bouldin        : {davies:.4f}")
    
    if centroids is not None:
        inertia = calculate_inertia(X, cluster_labels, centroids)
        print(f"Inertia (WCSS)        : {inertia:.4f}")
    
    # External metrics (if ground truth available)
    if true_labels is not None:
        print(f"\nExternal Metrics (with ground truth):")
        print("-" * 40)
        
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            true_labels, cluster_labels
        )
        
        print(f"Adjusted Rand Index   : {ari:.4f}")
        print(f"Normalized Mutual Info: {nmi:.4f}")
        print(f"Homogeneity           : {homogeneity:.4f}")
        print(f"Completeness          : {completeness:.4f}")
        print(f"V-measure             : {v_measure:.4f}")
    
    # Cluster statistics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\nCluster Statistics:")
    print("-" * 20)
    print(f"Number of clusters    : {len(unique_labels)}")
    print(f"Cluster sizes         : {dict(zip(unique_labels, counts))}")
    print(f"Largest cluster       : {counts.max()} samples")
    print(f"Smallest cluster      : {counts.min()} samples")
    print(f"Avg cluster size      : {counts.mean():.1f} samples")
    
    return {
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'ari': ari if true_labels is not None else None,
        'nmi': nmi if true_labels is not None else None,
        'v_measure': v_measure if true_labels is not None else None
    }
```

---

## D.4 When to Use Each Metric

### D.4.1 Classification Metric Selection Guide

| Scenario | Recommended Metrics | Reasoning |
|----------|-------------------|-----------|
| **Balanced Dataset** | Accuracy, F1-Score | All classes equally important |
| **Imbalanced Dataset** | Precision, Recall, F1, AUC-PR | Focus on minority class performance |
| **Medical Diagnosis** | Recall (Sensitivity), Specificity | Minimize false negatives and false positives |
| **Spam Detection** | Precision, Specificity | Minimize false positives (good emails marked as spam) |
| **Fraud Detection** | Recall, F1-Score | Don't miss fraudulent transactions |
| **Information Retrieval** | Precision@K, Recall@K, MAP | Relevant results at top positions |
| **Multi-class Problems** | Macro/Micro F1, Cohen's Kappa | Handle class imbalances appropriately |
| **Probability Calibration** | Log Loss, Brier Score | Well-calibrated probability estimates |

### D.4.2 Regression Metric Selection Guide

| Scenario | Recommended Metrics | Reasoning |
|----------|-------------------|-----------|
| **General Regression** | RMSE, MAE, R² | Standard metrics for most cases |
| **Outliers Present** | MAE, Huber Loss | Robust to extreme values |
| **Different Scales** | MAPE, SMAPE | Scale-independent comparison |
| **Time Series** | MASE, sMAPE | Account for seasonal patterns |
| **Model Comparison** | Adjusted R², AIC, BIC | Penalize model complexity |
| **Business Impact** | Domain-specific metrics | Custom metrics aligned with business goals |
| **Interpretability** | MAE, MAPE | Easy to explain to stakeholders |

### D.4.3 Clustering Metric Selection Guide

| Scenario | Recommended Metrics | Reasoning |
|----------|-------------------|-----------|
| **No Ground Truth** | Silhouette, Calinski-Harabasz | Internal validation only |
| **Known True Clusters** | ARI, NMI, V-measure | External validation available |
| **Spherical Clusters** | Inertia, K-means metrics | Assumes convex cluster shapes |
| **Arbitrary Shapes** | Silhouette, DBSCAN metrics | Handle non-convex clusters |
| **Different Sizes** | Silhouette analysis | Individual cluster quality |
| **Hierarchical Clustering** | Cophenetic correlation | Preserve hierarchical structure |

### D.4.4 Business Context Considerations

#### Cost-Sensitive Metrics
```python
def cost_sensitive_evaluation(y_true, y_pred, cost_matrix):
    """
    Evaluate model performance considering business costs
    
    cost_matrix: [[TN_cost, FP_cost],
                  [FN_cost, TP_cost]]
    """
    cm = confusion_matrix(y_true, y_pred)
    total_cost = np.sum(cm * cost_matrix)
    
    print(f"Confusion Matrix:")
    print(cm)
    print(f"Cost Matrix:")
    print(cost_matrix)
    print(f"Total Cost: {total_cost}")
    
    return total_cost

# Example: Medical diagnosis where false negatives are 10x more costly
medical_cost_matrix = np.array([[0, 1],    # TN=0, FP=1
                               [10, 0]])   # FN=10, TP=0

# Example: Marketing where false positives waste money
marketing_cost_matrix = np.array([[0, 5],    # TN=0, FP=5  
                                 [1, -2]])   # FN=1, TP=-2 (profit)
```

#### Custom Metrics for Domain-Specific Problems
```python
def custom_metric_example(y_true, y_pred):
    """
    Example: Revenue-based metric for recommendation systems
    """
    # Assume higher predicted values lead to higher revenue
    revenue_per_unit = 10
    cost_per_prediction = 0.1
    
    # Calculate revenue from correct high-value predictions
    high_value_threshold = 0.7
    high_value_correct = ((y_pred > high_value_threshold) & (y_true > high_value_threshold)).sum()
    
    total_revenue = high_value_correct * revenue_per_unit
    total_cost = len(y_pred) * cost_per_prediction
    
    net_profit = total_revenue - total_cost
    
    return {
        'total_revenue': total_revenue,
        'total_cost': total_cost,
        'net_profit': net_profit,
        'roi': net_profit / total_cost if total_cost > 0 else 0
    }
```

---

## D.5 Metric Limitations and Pitfalls

### D.5.1 Common Metric Pitfalls

#### Accuracy Paradox
```python
def demonstrate_accuracy_paradox():
    """
    Show how accuracy can be misleading with imbalanced data
    """
    # Highly imbalanced dataset (1% positive class)
    y_true = np.concatenate([np.ones(10), np.zeros(990)])
    
    # Naive classifier that always predicts negative
    y_pred_naive = np.zeros(1000)
    
    # Slightly better classifier
    y_pred_better = np.concatenate([np.ones(8), np.zeros(992)])
    
    print("Accuracy Paradox Demonstration:")
    print(f"Naive classifier accuracy: {accuracy_score(y_true, y_pred_naive):.3f}")
    print(f"Better classifier accuracy: {accuracy_score(y_true, y_pred_better):.3f}")
    
    print(f"Naive classifier F1: {f1_score(y_true, y_pred_naive):.3f}")
    print(f"Better classifier F1: {f1_score(y_true, y_pred_better):.3f}")

# The naive classifier has high accuracy but zero F1-score!
```

#### Simpson's Paradox in Metrics
```python
def demonstrate_simpsons_paradox():
    """
    Show how aggregate metrics can be misleading
    """
    # Two groups with different characteristics
    group1_true = np.array([1, 1, 1, 0, 0])
    group1_pred = np.array([1, 1, 0, 0, 0])
    
    group2_true = np.array([1, 0, 0, 0, 0])  
    group2_pred = np.array([1, 0, 0, 0, 0])
    
    # Individual group performance
    g1_acc = accuracy_score(group1_true, group1_pred)
    g2_acc = accuracy_score(group2_true, group2_pred)
    
    # Combined performance
    combined_true = np.concatenate([group1_true, group2_true])
    combined_pred = np.concatenate([group1_pred, group2_pred])
    combined_acc = accuracy_score(combined_true, combined_pred)
    
    print("Simpson's Paradox in Metrics:")
    print(f"Group 1 accuracy: {g1_acc:.3f}")
    print(f"Group 2 accuracy: {g2_acc:.3f}")
    print(f"Combined accuracy: {combined_acc:.3f}")
    print(f"Average of group accuracies: {(g1_acc + g2_acc)/2:.3f}")
```

### D.5.2 Best Practices

1. **Use Multiple Metrics:** Never rely on a single metric
2. **Consider Domain Context:** Business objectives should drive metric selection
3. **Validate on Multiple Datasets:** Ensure consistency across different data
4. **Statistical Significance:** Use confidence intervals and hypothesis tests
5. **Bias Analysis:** Check for performance differences across subgroups
6. **Temporal Validation:** Ensure metrics hold over time

```python
def robust_model_evaluation(models, X_test, y_test, cv_folds=5):
    """
    Robust evaluation using multiple metrics and statistical testing
    """
    from scipy import stats
    
    results = {}
    
    for model_name, model in models.items():
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds)
        
        # Bootstrap confidence intervals
        bootstrap_scores = []
        for _ in range(100):
            indices = np.random.choice(len(y_test), len(y_test), replace=True)
            X_boot, y_boot = X_test[indices], y_test[indices]
            y_pred_boot = model.predict(X_boot)
            bootstrap_scores.append(accuracy_score(y_boot, y_pred_boot))
        
        results[model_name] = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'bootstrap_ci': np.percentile(bootstrap_scores, [2.5, 97.5])
        }
    
    return results
```

---

This comprehensive metrics reference provides you with the tools to properly evaluate machine learning models across all domains. Remember that the choice of metrics should always align with your specific problem context, business objectives, and the characteristics of your data.

**Key Takeaway:** There is no single "best" metric. The art of machine learning evaluation lies in selecting the right combination of metrics that tell the complete story of your model's performance.