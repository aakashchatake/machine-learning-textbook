# Chapter 4: Classification Algorithms

> "The goal is to turn data into information, and information into insight."
> 
> — Carly Fiorina

## Learning Objectives

By the end of this chapter, you will be able to:
- **Understand** the fundamentals of classification algorithms
- **Implement** decision trees, KNN, SVM, and logistic regression
- **Evaluate** classification model performance using appropriate metrics
- **Apply** feature engineering techniques for classification problems
- **Compare** different algorithms and select the best for specific problems
- **Build** end-to-end classification pipelines

---

## Statistical Learning Theory of Classification

Classification represents one of the fundamental problems in statistical learning theory, where we seek to learn a mapping from input features to discrete output categories. The mathematical foundations draw from probability theory, decision theory, and statistical inference.

**The Classification Learning Problem**

Given a training dataset D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} where xᵢ ∈ ℝᵈ are feature vectors and yᵢ ∈ {1, 2, ..., K} are class labels, we want to learn a function:

**f: ℝᵈ → {1, 2, ..., K}**

That minimizes the expected prediction error on future, unseen data.

**Bayes Optimal Classifier**

The theoretically optimal classifier is the Bayes classifier, which assigns each point to the class with highest posterior probability:

**f*(x) = arg max_k P(Y = k | X = x)**

Using Bayes' theorem:
**P(Y = k | X = x) = P(X = x | Y = k) × P(Y = k) / P(X = x)**

The Bayes error rate represents the lowest achievable error rate for any classifier:

**L* = 1 - E[max_k P(Y = k | X = x)]**

**Decision Boundaries and Complexity**

Different classification algorithms make different assumptions about the decision boundary:
- **Linear classifiers**: Assume linear decision boundaries
- **Nonlinear classifiers**: Can learn complex, curved boundaries
- **Non-parametric methods**: Make minimal distributional assumptions

Classification is a supervised learning task where we predict discrete class labels rather than continuous values, requiring specialized algorithms and evaluation metrics.

### 4.1.1 Types of Classification Problems

#### Binary Classification
Predicting one of two possible outcomes:
- **Email Spam Detection**: Spam or Not Spam
- **Medical Diagnosis**: Disease Present or Absent  
- **Credit Approval**: Approved or Rejected

#### Multi-class Classification
Predicting one of multiple possible classes:
- **Image Recognition**: Cat, Dog, Bird, etc.
- **Text Classification**: Sports, Politics, Technology, etc.
- **Product Categorization**: Electronics, Clothing, Books, etc.

#### Multi-label Classification
Predicting multiple labels simultaneously:
- **Movie Genre**: Action AND Comedy AND Drama
- **Medical Symptoms**: Multiple conditions present
- **Document Tags**: Multiple relevant topics

### 4.1.2 Classification vs. Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression

# Generate sample data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Classification example
X_class, y_class = make_classification(n_samples=200, n_features=2, 
                                      n_redundant=0, n_informative=2,
                                      n_clusters_per_class=1, random_state=42)

ax1.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='viridis')
ax1.set_title('Classification Problem\n(Discrete Classes)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Regression example  
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)

ax2.scatter(X_reg, y_reg, alpha=0.6)
ax2.set_title('Regression Problem\n(Continuous Values)')
ax2.set_xlabel('Feature')
ax2.set_ylabel('Target Value')

plt.tight_layout()
plt.show()

print("Classification Output: Discrete classes (0, 1, 2, ...)")
print("Regression Output: Continuous values (1.5, 2.7, 10.3, ...)")
```

---

## Decision Trees: Information Theory and Recursive Partitioning

Decision trees represent one of the most interpretable machine learning algorithms, grounded in information theory and recursive optimization. They construct hierarchical decision rules that partition the feature space into regions of high class purity.

**Mathematical Foundation: Recursive Binary Partitioning**

A decision tree recursively partitions the feature space X ⊆ ℝᵈ into disjoint regions R₁, R₂, ..., Rₘ such that:

**X = ⋃ᵢ₌₁ᵐ Rᵢ and Rᵢ ∩ Rⱼ = ∅ for i ≠ j**

Each region Rᵢ is associated with a class prediction ŷᵢ, typically the majority class within that region.

**The Greedy Splitting Algorithm**

At each node, the algorithm chooses the split that maximally reduces impurity:

**(j*, s*) = arg max_{j,s} [N_parent × I(parent) - N_left × I(left) - N_right × I(right)]**

Where:
- **j** is the feature index, **s** is the split threshold
- **I(·)** is an impurity measure (entropy, Gini, etc.)
- **N** represents the number of samples in each node

**Information-Theoretic Splitting Criteria**

The choice of impurity measure determines the tree's behavior:

1. **Entropy (Information Gain)**: H(S) = -Σᵢ pᵢ log₂(pᵢ)
2. **Gini Impurity**: G(S) = 1 - Σᵢ pᵢ²  
3. **Classification Error**: E(S) = 1 - max_i(pᵢ)

Each criterion represents different ways of measuring node "purity" or class homogeneity.

### Tree Construction Algorithm

Decision trees construct hierarchical rules through recursive feature space partitioning, choosing splits that maximize information gain or minimize impurity measures.

#### Example: Should I Play Tennis?

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create a simple tennis dataset
tennis_data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 
                'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 
                'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True',
             'False', 'False', 'False', 'True', 'True', 'False', 'True'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 
             'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df_tennis = pd.DataFrame(tennis_data)
print("Tennis Dataset:")
print(df_tennis.head(10))

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

# Create encoders for each categorical column
encoders = {}
df_encoded = df_tennis.copy()

for column in df_tennis.columns:
    if df_tennis[column].dtype == 'object':
        encoders[column] = LabelEncoder()
        df_encoded[column] = encoders[column].fit_transform(df_tennis[column])

X = df_encoded.drop('Play', axis=1)
y = df_encoded['Play']

print("\nEncoded Dataset:")
print(df_encoded.head())
```

### Information Theory Foundations of Tree Splitting

Decision tree splitting criteria are grounded in information theory and statistical measures of uncertainty. Understanding these mathematical foundations is crucial for algorithm selection and hyperparameter tuning.

**Entropy: Measuring Information Content**

Entropy H(S) quantifies the expected information content (uncertainty) in a dataset S:

**H(S) = -Σᵢ₌₁ᶜ pᵢ log₂(pᵢ)**

**Mathematical Properties**:
- **Maximum entropy**: H(S) = log₂(c) when all classes are equally likely
- **Minimum entropy**: H(S) = 0 when all samples belong to one class  
- **Concavity**: Entropy is a concave function, ensuring unique maxima

**Information Gain: Quantifying Split Quality**

Information gain measures the reduction in entropy achieved by a split:

**IG(S, A) = H(S) - Σᵥ∈Values(A) (|Sᵥ|/|S|) × H(Sᵥ)**

Where Sᵥ is the subset of S where attribute A has value v.

**Gini Impurity: Alternative Measure**

Gini impurity provides a computationally efficient alternative:

**Gini(S) = 1 - Σᵢ₌₁ᶜ pᵢ²**

**Mathematical comparison**:
- **Entropy**: More theoretically principled (information-theoretic foundation)
- **Gini**: Computationally faster (no logarithms)  
- **Both**: Concave functions that favor balanced splits

**Gain Ratio: Addressing Split Bias**

Information gain is biased toward attributes with many values. Gain ratio normalizes by split information:

**GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)**

Where **SplitInfo(S, A) = -Σᵥ (|Sᵥ|/|S|) log₂(|Sᵥ|/|S|)**

This prevents overfitting to high-cardinality categorical features.

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $A$ is the attribute/feature
- $S_v$ is the subset of $S$ where attribute $A$ has value $v$

#### Implementation of Information Gain

```python
import numpy as np
from collections import Counter

def calculate_entropy(y):
    """Calculate entropy of a dataset"""
    if len(y) == 0:
        return 0
    
    # Count occurrences of each class
    counts = Counter(y)
    total = len(y)
    
    # Calculate entropy
    entropy = 0
    for count in counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)
    
    return entropy

def calculate_information_gain(X, y, feature_index):
    """Calculate information gain for a specific feature"""
    # Calculate entropy of the original dataset
    parent_entropy = calculate_entropy(y)
    
    # Get unique values of the feature
    feature_values = np.unique(X[:, feature_index])
    weighted_entropy = 0
    
    # Calculate weighted entropy after the split
    for value in feature_values:
        # Create subset where feature equals value
        subset_indices = X[:, feature_index] == value
        subset_y = y[subset_indices]
        
        # Calculate weight and entropy of subset
        weight = len(subset_y) / len(y)
        subset_entropy = calculate_entropy(subset_y)
        weighted_entropy += weight * subset_entropy
    
    # Information gain = parent entropy - weighted entropy
    information_gain = parent_entropy - weighted_entropy
    return information_gain

# Calculate information gain for each feature
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Windy']
X_array = X.values
y_array = y.values

print("Information Gain Analysis:")
print("-" * 40)
for i, feature in enumerate(feature_names):
    ig = calculate_information_gain(X_array, y_array, i)
    print(f"{feature:<12}: {ig:.4f}")

# Calculate base entropy
base_entropy = calculate_entropy(y_array)
print(f"\nBase Entropy: {base_entropy:.4f}")
```

### 4.2.3 Building Decision Trees with Scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Train a decision tree
dt_classifier = DecisionTreeClassifier(
    criterion='entropy',  # Use information gain
    random_state=42,
    max_depth=3  # Limit depth to prevent overfitting
)

# Fit the model
dt_classifier.fit(X, y)

# Make predictions
predictions = dt_classifier.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"Training Accuracy: {accuracy:.4f}")

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, 
          feature_names=feature_names,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title('Decision Tree for Tennis Playing Decision')
plt.show()

# Feature importance
feature_importance = dt_classifier.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance in Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4.2.4 Real-World Example: Iris Species Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

print("Iris Dataset Information:")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
print(f"Dataset shape: {X_iris.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

# Train decision tree
dt_iris = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42,
    max_depth=3
)

dt_iris.fit(X_train, y_train)

# Make predictions
y_pred = dt_iris.predict(X_test)

# Evaluate performance
train_accuracy = dt_iris.score(X_train, y_train)
test_accuracy = dt_iris.score(X_test, y_test)

print(f"\nModel Performance:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20, 12))
plot_tree(dt_iris,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Iris Species Classification')
plt.show()
```

### 4.2.5 Overfitting and Pruning

Decision trees can easily overfit, especially when they grow too deep. Let's explore this concept:

#### Demonstrating Overfitting

```python
from sklearn.model_selection import validation_curve

# Test different max_depth values
max_depths = range(1, 21)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(criterion='entropy', random_state=42),
    X_iris, y_iris,
    param_name='max_depth',
    param_range=max_depths,
    cv=5,
    scoring='accuracy'
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_mean, 'o-', color='blue', label='Training Accuracy')
plt.fill_between(max_depths, train_mean - train_std, train_mean + train_std, 
                 color='blue', alpha=0.1)

plt.plot(max_depths, val_mean, 'o-', color='red', label='Validation Accuracy')
plt.fill_between(max_depths, val_mean - val_std, val_mean + val_std, 
                 color='red', alpha=0.1)

plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Validation Curve - Decision Tree Max Depth')
plt.legend()
plt.grid(True)
plt.show()

# Find optimal depth
optimal_depth = max_depths[np.argmax(val_mean)]
print(f"Optimal max_depth: {optimal_depth}")
print(f"Best validation accuracy: {val_mean[np.argmax(val_mean)]:.4f}")
```

#### Pruning Parameters

```python
# Compare different pruning strategies
pruning_params = {
    'No Pruning': {},
    'Max Depth = 3': {'max_depth': 3},
    'Min Samples Split = 20': {'min_samples_split': 20},
    'Min Samples Leaf = 5': {'min_samples_leaf': 5},
    'Max Features = 2': {'max_features': 2}
}

results = {}

for name, params in pruning_params.items():
    # Create and train model
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42, **params)
    dt.fit(X_train, y_train)
    
    # Evaluate
    train_acc = dt.score(X_train, y_train)
    test_acc = dt.score(X_test, y_test)
    
    results[name] = {
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Tree Depth': dt.get_depth(),
        'Number of Leaves': dt.get_n_leaves()
    }

# Display results
results_df = pd.DataFrame(results).T
print("Pruning Strategy Comparison:")
print(results_df.round(4))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training vs Test Accuracy
axes[0,0].bar(results_df.index, results_df['Train Accuracy'], 
              alpha=0.7, label='Train')
axes[0,0].bar(results_df.index, results_df['Test Accuracy'], 
              alpha=0.7, label='Test')
axes[0,0].set_title('Training vs Test Accuracy')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend()
axes[0,0].tick_params(axis='x', rotation=45)

# Tree Depth
axes[0,1].bar(results_df.index, results_df['Tree Depth'])
axes[0,1].set_title('Tree Depth')
axes[0,1].set_ylabel('Depth')
axes[0,1].tick_params(axis='x', rotation=45)

# Number of Leaves
axes[1,0].bar(results_df.index, results_df['Number of Leaves'])
axes[1,0].set_title('Number of Leaves')
axes[1,0].set_ylabel('Leaves')
axes[1,0].tick_params(axis='x', rotation=45)

# Overfitting indicator (Train - Test accuracy)
overfitting = results_df['Train Accuracy'] - results_df['Test Accuracy']
axes[1,1].bar(results_df.index, overfitting)
axes[1,1].set_title('Overfitting Indicator\n(Train - Test Accuracy)')
axes[1,1].set_ylabel('Accuracy Difference')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### 4.2.6 Random Forest: Ensemble of Decision Trees

Random Forest improves upon single decision trees by combining multiple trees, reducing overfitting and improving generalization.

#### Key Concepts

1. **Bootstrap Aggregating (Bagging)**: Train each tree on a different bootstrap sample
2. **Feature Randomness**: Each tree uses a random subset of features
3. **Voting**: Final prediction is the majority vote of all trees

#### Mathematical Foundation

For a Random Forest with $T$ trees, the prediction is:

$$\hat{y} = \text{mode}\{h_1(x), h_2(x), ..., h_T(x)\}$$

Where $h_t(x)$ is the prediction of the $t$-th tree.

#### Implementation and Comparison

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time

# Compare single decision tree vs random forest
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest (10 trees)': RandomForestClassifier(n_estimators=10, random_state=42),
    'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42),
    'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42)
}

comparison_results = {}

for name, model in models.items():
    # Measure training time
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_iris, y_iris, cv=5, scoring='accuracy')
    
    # Fit model for additional metrics
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Store results
    comparison_results[name] = {
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Training Time': train_time,
        'Train Accuracy': model.score(X_train, y_train),
        'Test Accuracy': model.score(X_test, y_test)
    }

# Display comparison
comparison_df = pd.DataFrame(comparison_results).T
print("Decision Tree vs Random Forest Comparison:")
print(comparison_df.round(4))

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Cross-validation scores
axes[0].bar(comparison_df.index, comparison_df['CV Mean'])
axes[0].errorbar(range(len(comparison_df)), comparison_df['CV Mean'], 
                yerr=comparison_df['CV Std'], fmt='none', color='black', capsize=5)
axes[0].set_title('Cross-Validation Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=45)

# Training time
axes[1].bar(comparison_df.index, comparison_df['Training Time'])
axes[1].set_title('Training Time')
axes[1].set_ylabel('Time (seconds)')
axes[1].tick_params(axis='x', rotation=45)

# Train vs Test accuracy
x_pos = np.arange(len(comparison_df))
width = 0.35
axes[2].bar(x_pos - width/2, comparison_df['Train Accuracy'], 
           width, label='Train', alpha=0.7)
axes[2].bar(x_pos + width/2, comparison_df['Test Accuracy'], 
           width, label='Test', alpha=0.7)
axes[2].set_title('Training vs Test Accuracy')
axes[2].set_ylabel('Accuracy')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(comparison_df.index, rotation=45)
axes[2].legend()

plt.tight_layout()
plt.show()
```

#### Feature Importance in Random Forest

```python
# Train a Random Forest and analyze feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = rf_model.feature_importances_
feature_names = iris.feature_names

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("Random Forest Feature Importance:")
print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(importance_df)), importance_df['Importance'])
plt.yticks(range(len(importance_df)), importance_df['Feature'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance - Iris Dataset')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Compare individual tree predictions
n_trees_to_show = 5
individual_predictions = []

for i in range(n_trees_to_show):
    tree_pred = rf_model.estimators_[i].predict(X_test)
    individual_predictions.append(tree_pred)

# Show first few predictions
print(f"\nFirst 10 test samples - Individual tree predictions:")
print("Sample | Tree1 | Tree2 | Tree3 | Tree4 | Tree5 | RF Pred | Actual")
print("-" * 70)

rf_pred = rf_model.predict(X_test)
for i in range(10):
    tree_preds = " | ".join([f"  {pred[i]}  " for pred in individual_predictions])
    print(f"  {i:2d}   | {tree_preds} |   {rf_pred[i]}   |   {y_test[i]}")
```

### 4.2.7 Advantages and Disadvantages

#### Decision Trees

**Advantages:**
- Easy to understand and interpret
- Requires little data preparation
- Handles both numerical and categorical data
- Can capture non-linear relationships
- Feature selection happens automatically

**Disadvantages:**
- Prone to overfitting
- Unstable (small data changes can result in different trees)
- Biased toward features with more levels
- Can create overly complex trees

#### Random Forest

**Advantages:**
- Reduces overfitting compared to decision trees
- More stable and robust
- Provides feature importance
- Handles missing values well
- Works well with default parameters

**Disadvantages:**
- Less interpretable than single trees
- Can still overfit with very noisy data
- Memory intensive for large datasets
- Slower prediction than single trees

---

## K-Nearest Neighbors: Non-Parametric Learning Theory

K-Nearest Neighbors represents a fundamental non-parametric approach to classification, based on the assumption of local smoothness in the data distribution. It embodies the principle that nearby points in feature space are likely to share the same class label.

**Mathematical Foundation: Non-Parametric Density Estimation**

KNN implicitly estimates the class conditional densities P(X|Y = k) using a non-parametric approach. For a query point x, the posterior probability is estimated as:

**P̂(Y = k | X = x) = (1/K) Σᵢ∈N_K(x) I(yᵢ = k)**

Where:
- **N_K(x)** is the set of K nearest neighbors to x
- **I(yᵢ = k)** is the indicator function (1 if yᵢ = k, 0 otherwise)

**The Local Smoothness Assumption**

KNN assumes that the target function f: X → Y is locally smooth, meaning:

**If ||x₁ - x₂|| is small, then P(Y | X = x₁) ≈ P(Y | X = x₂)**

This assumption allows local interpolation to approximate the true posterior probabilities.

**Consistency and Convergence Properties**

Under mild regularity conditions, KNN is universally consistent:

**As n → ∞ and K → ∞ such that K/n → 0, then P̂ → P***

Where P* is the Bayes optimal classifier. This theoretical guarantee makes KNN a valuable baseline algorithm.

### Non-Parametric Classification Algorithm

KNN is a "lazy learning" algorithm that defers computation until prediction time, storing the entire training dataset rather than learning parameters.

#### How KNN Works:
1. **Choose K**: Decide how many neighbors to consider
2. **Calculate Distance**: Measure distance from query point to all training points
3. **Find Neighbors**: Identify the K closest points
4. **Vote**: Assign the class based on majority vote of neighbors

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a simple 2D dataset for visualization
X_demo, y_demo = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                    n_informative=2, n_clusters_per_class=1, 
                                    random_state=42)

# Visualize the concept
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
k_values = [1, 3, 7]

for idx, k in enumerate(k_values):
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_demo, y_demo)
    
    # Create decision boundary
    h = 0.02
    x_min, x_max = X_demo[:, 0].min() - 1, X_demo[:, 0].max() + 1
    y_min, y_max = X_demo[:, 1].min() - 1, X_demo[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = axes[idx].scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, cmap='viridis')
    axes[idx].set_title(f'KNN with K={k}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### Distance Metrics: Mathematical Foundations of Similarity

The choice of distance metric fundamentally determines the notion of "similarity" in KNN, directly affecting the algorithm's inductive bias and performance characteristics.

**Mathematical Requirements for Distance Metrics**

A valid distance metric d(x, y) must satisfy the metric axioms:

1. **Non-negativity**: d(x, y) ≥ 0
2. **Identity**: d(x, y) = 0 ⟺ x = y  
3. **Symmetry**: d(x, y) = d(y, x)
4. **Triangle Inequality**: d(x, z) ≤ d(x, y) + d(y, z)

**Lp Norm Family**

Most common distance metrics belong to the Lp norm family:

**Lp(x, y) = (Σᵢ₌₁ᵈ |xᵢ - yᵢ|ᵖ)^(1/p)**

**Special Cases**:
- **L₁ (Manhattan)**: d(x,y) = Σᵢ|xᵢ - yᵢ| (robust to outliers)
- **L₂ (Euclidean)**: d(x,y) = √(Σᵢ(xᵢ - yᵢ)²) (most common)
- **L∞ (Chebyshev)**: d(x,y) = maxᵢ|xᵢ - yᵢ| (worst-case distance)

**Mahalanobis Distance: Covariance-Aware Metric**

Standard metrics assume feature independence and equal importance. Mahalanobis distance accounts for covariance:

**d_M(x, y) = √((x - y)ᵀ Σ⁻¹ (x - y))**

Where Σ is the covariance matrix. This metric:
- **Normalizes by variance** (automatic scaling)
- **Accounts for correlation** between features
- **Reduces to Euclidean** when features are independent and unit variance

#### Distance Metric Selection Considerations

```python
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

# Sample points for distance calculation
point1 = np.array([1, 2])
point2 = np.array([4, 6])
point3 = np.array([2, 1])

points = np.array([point1, point2, point3])

print("Distance Calculations:")
print("Points:", points)
print()

# Euclidean Distance
euclidean_dist = euclidean_distances(points)
print("Euclidean Distance Matrix:")
print(euclidean_dist.round(3))

# Manhattan Distance  
manhattan_dist = manhattan_distances(points)
print("\nManhattan Distance Matrix:")
print(manhattan_dist.round(3))

# Cosine Distance
cosine_dist = cosine_distances(points)
print("\nCosine Distance Matrix:")
print(cosine_dist.round(3))

# Manual calculation for understanding
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def manhattan_distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def cosine_distance(p1, p2):
    dot_product = np.dot(p1, p2)
    norms = np.linalg.norm(p1) * np.linalg.norm(p2)
    return 1 - (dot_product / norms)

print(f"\nManual Euclidean distance between point1 and point2: {euclidean_distance(point1, point2):.3f}")
print(f"Manual Manhattan distance between point1 and point2: {manhattan_distance(point1, point2):.3f}")
print(f"Manual Cosine distance between point1 and point2: {cosine_distance(point1, point2):.3f}")
```

#### Visual Comparison of Distance Metrics

```python
# Visualize different distance metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
distance_metrics = ['euclidean', 'manhattan', 'cosine']

for idx, metric in enumerate(distance_metrics):
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_demo, y_demo)
    
    # Create decision boundary
    h = 0.02
    x_min, x_max = X_demo[:, 0].min() - 1, X_demo[:, 0].max() + 1
    y_min, y_max = X_demo[:, 1].min() - 1, X_demo[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    axes[idx].scatter(X_demo[:, 0], X_demo[:, 1], c=y_demo, cmap='viridis')
    axes[idx].set_title(f'KNN with {metric.capitalize()} Distance')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### 4.3.3 Choosing the Optimal K

The choice of K is crucial for KNN performance:

```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Use Iris dataset for K optimization
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

# Scale features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_iris)
X_test_scaled = scaler.transform(X_test_iris)

# Test different K values
k_values = range(1, 31)
cv_scores = []
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Cross-validation score
    cv_score = cross_val_score(knn, X_train_scaled, y_train_iris, cv=5).mean()
    cv_scores.append(cv_score)
    
    # Fit model for train/test scores
    knn.fit(X_train_scaled, y_train_iris)
    train_score = knn.score(X_train_scaled, y_train_iris)
    test_score = knn.score(X_test_scaled, y_test_iris)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, cv_scores, 'o-', label='Cross-Validation Score')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Score vs K')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_values, train_scores, 'o-', label='Training Score', alpha=0.7)
plt.plot(k_values, test_scores, 'o-', label='Testing Score', alpha=0.7)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Find optimal K
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal K value: {optimal_k}")
print(f"Best cross-validation score: {max(cv_scores):.4f}")

# Compare odd vs even K values
odd_k_scores = [cv_scores[i] for i in range(len(k_values)) if k_values[i] % 2 == 1]
even_k_scores = [cv_scores[i] for i in range(len(k_values)) if k_values[i] % 2 == 0]

print(f"Average score for odd K: {np.mean(odd_k_scores):.4f}")
print(f"Average score for even K: {np.mean(even_k_scores):.4f}")
```

### 4.3.4 KNN Implementation from Scratch

Let's implement KNN from scratch to understand the algorithm better:

```python
class KNNFromScratch:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
        
    def _calculate_distance(self, x1, x2):
        """Calculate distance between two points"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")
    
    def predict(self, X):
        """Make predictions for test data"""
        predictions = []
        
        for test_point in X:
            # Calculate distances to all training points
            distances = []
            for train_point in self.X_train:
                dist = self._calculate_distance(test_point, train_point)
                distances.append(dist)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Majority vote
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Test our implementation
knn_scratch = KNNFromScratch(k=3, distance_metric='euclidean')
knn_scratch.fit(X_train_scaled, y_train_iris)

# Compare with sklearn
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train_scaled, y_train_iris)

scratch_accuracy = knn_scratch.score(X_test_scaled, y_test_iris)
sklearn_accuracy = knn_sklearn.score(X_test_scaled, y_test_iris)

print("KNN Implementation Comparison:")
print(f"From Scratch Accuracy: {scratch_accuracy:.4f}")
print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")
print(f"Difference: {abs(scratch_accuracy - sklearn_accuracy):.6f}")
```

### 4.3.5 KNN Advantages and Disadvantages

#### Advantages:
- Simple to understand and implement
- No assumptions about data distribution
- Works well with small datasets
- Can be used for both classification and regression
- Adapts to new data easily (just add to training set)

#### Disadvantages:
- Computationally expensive for large datasets
- Sensitive to irrelevant features and feature scaling
- Sensitive to local structure of data
- Memory intensive (stores all training data)
- Poor performance with high-dimensional data (curse of dimensionality)

## Support Vector Machines: Optimal Margin Theory

Support Vector Machines represent one of the most theoretically principled approaches to classification, grounded in statistical learning theory and convex optimization. SVMs find the optimal separating hyperplane that maximizes the margin between classes.

**Geometric Intuition: Maximum Margin Principle**

Given linearly separable data, infinitely many hyperplanes can separate the classes. SVM chooses the hyperplane that maximizes the **margin** - the distance to the nearest training examples.

For a hyperplane defined by **wᵀx + b = 0**, the margin is **2/||w||**.

**Primal Optimization Problem**

SVM solves the constrained optimization problem:

**minimize_{w,b}** (1/2)||w||² 

**subject to:** yᵢ(wᵀxᵢ + b) ≥ 1, ∀i ∈ {1,...,n}

This ensures all points are correctly classified with margin at least 1/||w||.

**Soft-Margin SVM: Handling Non-Separable Data**

For non-linearly separable data, we introduce slack variables ξᵢ:

**minimize_{w,b,ξ}** (1/2)||w||² + C Σᵢ ξᵢ

**subject to:** 
- yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ∀i
- ξᵢ ≥ 0, ∀i

The parameter C controls the bias-variance trade-off:
- **Large C**: Low bias, high variance (hard margin)
- **Small C**: High bias, low variance (soft margin)

**Dual Formulation and Support Vectors**

Using Lagrange multipliers, the dual problem becomes:

**maximize_α** Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼ⟨xᵢ,xⱼ⟩

**subject to:** 
- Σᵢ αᵢyᵢ = 0
- 0 ≤ αᵢ ≤ C, ∀i

Points with αᵢ > 0 are **support vectors** - they define the decision boundary.

### The Kernel Trick: Infinite-Dimensional Feature Spaces

The kernel trick enables SVMs to efficiently work in high-dimensional (even infinite-dimensional) feature spaces without explicitly computing the transformations.

**Mathematical Foundation of Kernels**

A kernel function K(x, z) implicitly defines a mapping φ: X → H to a Hilbert space H:

**K(x, z) = ⟨φ(x), φ(z)⟩_H**

**Mercer's Theorem** provides conditions for valid kernels: K must be positive semi-definite.

**Common Kernel Functions and Their Properties**

1. **Linear Kernel**: K(x, z) = xᵀz
   - **Feature space**: Original space (no transformation)
   - **Use case**: Linearly separable data

2. **Polynomial Kernel**: K(x, z) = (γxᵀz + r)ᵈ
   - **Feature space**: All monomials up to degree d
   - **Dimensionality**: (n+d choose d) features
   - **Use case**: Polynomial decision boundaries

3. **RBF (Gaussian) Kernel**: K(x, z) = exp(-γ||x - z||²)
   - **Feature space**: Infinite-dimensional
   - **Properties**: Universal approximator, smooth boundaries
   - **Parameter γ**: Controls smoothness (large γ → overfitting)

4. **Sigmoid Kernel**: K(x, z) = tanh(γxᵀz + r)
   - **Feature space**: Neural network-like transformation
   - **Note**: Not always positive semi-definite

**Kernel Selection Principles**

- **Linear**: When #features >> #samples
- **RBF**: Default choice, good for most problems  
- **Polynomial**: When domain knowledge suggests polynomial relationships

### 4.4.3 SVM Implementation

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline with feature scaling and SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Train the SVM
svm_pipeline.fit(X_train_iris, y_train_iris)

# Make predictions
y_pred_svm = svm_pipeline.predict(X_test_iris)

# Evaluate performance
svm_accuracy = accuracy_score(y_test_iris, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Detailed classification report
print("\nSVM Classification Report:")
print(classification_report(y_test_iris, y_pred_svm, target_names=iris.target_names))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm_svm = confusion_matrix(y_test_iris, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - SVM Iris Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### 4.4.4 SVM with Different Kernels

```python
# Compare different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_accuracies = {}

for kernel in kernels:
    svm_model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_iris, y_train_iris)
    y_pred = svm_model.predict(X_test_iris)
    accuracy = accuracy_score(y_test_iris, y_pred)
    svm_accuracies[kernel] = accuracy
    print(f"{kernel.capitalize()} Kernel SVM Accuracy: {accuracy:.4f}")

# Visualize kernel performance
plt.figure(figsize=(10, 6))
plt.bar(svm_accuracies.keys(), svm_accuracies.values())
plt.title('SVM Accuracy with Different Kernels')
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.grid(axis='y')
plt.show()
```

### 4.4.5 SVM Advantages and Disadvantages

#### Advantages:
- Effective in high-dimensional spaces
- Robust to overfitting in high dimensions
- Versatile (different kernels for different data types)
- Works well with clear margin of separation

#### Disadvantages:
- Not very effective on very large datasets
- Less effective on noisy data
- Requires careful tuning of parameters
- Can be memory intensive

---

## Logistic Regression: Probabilistic Classification Theory

Logistic regression represents a fundamental probabilistic approach to classification, modeling the posterior class probabilities using the logistic function. It's grounded in maximum likelihood estimation and provides well-calibrated probability estimates.

**Probabilistic Foundation: Generalized Linear Models**

Logistic regression belongs to the family of Generalized Linear Models (GLMs), where we model the relationship between features and target through a link function.

For binary classification, we model the log-odds (logit):

**log(P(y=1|x) / P(y=0|x)) = wᵀx + b**

Solving for P(y=1|x) gives the logistic function:

**P(y=1|x) = 1 / (1 + e^{-(wᵀx + b)}) = σ(wᵀx + b)**

**Statistical Interpretation**

The linear combination wᵀx + b represents the **log-odds ratio**:
- When wᵀx + b = 0, P(y=1|x) = 0.5 (equal probability)
- When wᵀx + b > 0, P(y=1|x) > 0.5 (favors class 1)  
- When wᵀx + b < 0, P(y=1|x) < 0.5 (favors class 0)

**Decision Boundary**

The decision boundary is the hyperplane where P(y=1|x) = 0.5:

**wᵀx + b = 0**

This is linear in the original feature space, making logistic regression a linear classifier.

#### The Sigmoid Function

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow

# Plot sigmoid function
z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, 'b-', linewidth=2, label='Sigmoid Function')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('z = w^T x + b')
plt.ylabel('P(y=1|x)')
plt.title('Sigmoid Function for Logistic Regression')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 1)
plt.show()

print("Sigmoid Function Properties:")
print(f"sigmoid(0) = {sigmoid(0):.3f}")
print(f"sigmoid(-∞) ≈ {sigmoid(-10):.6f}")
print(f"sigmoid(+∞) ≈ {sigmoid(10):.6f}")
```

### Maximum Likelihood Estimation and Cost Function

Logistic regression parameters are estimated using Maximum Likelihood Estimation (MLE), which provides strong statistical foundations for the learning algorithm.

**Likelihood Function**

Assuming independence, the likelihood of observing the data is:

**L(w) = ∏ᵢ₌₁ⁿ P(yᵢ|xᵢ)^yᵢ × (1 - P(yᵢ|xᵢ))^{1-yᵢ}**

**Log-Likelihood (Easier to Optimize)**

Taking the logarithm (monotonic transformation):

**ℓ(w) = Σᵢ₌₁ⁿ [yᵢ log P(yᵢ|xᵢ) + (1-yᵢ) log(1 - P(yᵢ|xᵢ))]**

**Cost Function (Negative Log-Likelihood)**

To convert to a minimization problem:

**J(w) = -1/n × ℓ(w) = -1/n Σᵢ₌₁ⁿ [yᵢ log σ(wᵀxᵢ) + (1-yᵢ) log(1 - σ(wᵀxᵢ))]**

**Convexity and Global Optimization**

The logistic regression cost function is **strictly convex**, guaranteeing:
1. **Unique global minimum**: No local minima
2. **Gradient descent convergence**: Always reaches the optimal solution
3. **Well-defined MLE**: Under mild regularity conditions

#### Implementation from Scratch

```python
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def fit(self, X, y):
        """Train the logistic regression model"""
        # Add bias term
        m, n = X.shape
        self.weights = np.zeros(n + 1)
        X_with_bias = np.column_stack([np.ones(m), X])
        
        # Store cost history
        self.cost_history = []
        
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X_with_bias, self.weights)
            predictions = sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Backward pass (gradient descent)
            gradient = np.dot(X_with_bias.T, (predictions - y)) / m
            self.weights -= self.learning_rate * gradient
            
    def _compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost"""
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
        
    def predict_proba(self, X):
        """Predict class probabilities"""
        m = X.shape[0]
        X_with_bias = np.column_stack([np.ones(m), X])
        return sigmoid(np.dot(X_with_bias, self.weights))
        
    def predict(self, X):
        """Make predictions"""
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Test implementation with binary classification
from sklearn.datasets import make_classification

X_binary, y_binary = make_classification(n_samples=1000, n_features=2, 
                                        n_redundant=0, n_informative=2,
                                        n_clusters_per_class=1, random_state=42)

# Split data
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# Train custom implementation
lr_scratch = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
lr_scratch.fit(X_train_lr, y_train_lr)

# Train sklearn implementation
from sklearn.linear_model import LogisticRegression
lr_sklearn = LogisticRegression(random_state=42)
lr_sklearn.fit(X_train_lr, y_train_lr)

# Compare results
scratch_acc = lr_scratch.score(X_test_lr, y_test_lr)
sklearn_acc = lr_sklearn.score(X_test_lr, y_test_lr)

print("Logistic Regression Comparison:")
print(f"From Scratch Accuracy: {scratch_acc:.4f}")
print(f"Scikit-learn Accuracy: {sklearn_acc:.4f}")
print(f"Difference: {abs(scratch_acc - sklearn_acc):.6f}")

# Plot cost function convergence
plt.figure(figsize=(10, 6))
plt.plot(lr_scratch.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Logistic Regression Cost Function Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

### 4.5.3 Multi-class Logistic Regression

For multi-class problems, logistic regression uses strategies like One-vs-Rest or softmax regression.

#### Softmax Function

For K classes, the softmax function is:

$$P(y=k|x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^{K} e^{w_j^T x + b_j}}$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Multi-class comparison on Iris dataset
multiclass_strategies = {
    'Multinomial': LogisticRegression(multi_class='multinomial', random_state=42),
    'One-vs-Rest': LogisticRegression(multi_class='ovr', random_state=42),
    'OneVsRestClassifier': OneVsRestClassifier(LogisticRegression(random_state=42))
}

multiclass_results = {}
for name, classifier in multiclass_strategies.items():
    classifier.fit(X_train_scaled, y_train_iris)
    
    train_acc = classifier.score(X_train_scaled, y_train_iris)
    test_acc = classifier.score(X_test_scaled, y_test_iris)
    
    multiclass_results[name] = {
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc
    }

multiclass_df = pd.DataFrame(multiclass_results).T
print("Multi-class Logistic Regression Comparison:")
print(multiclass_df.round(4))
```

### 4.5.4 Regularization in Logistic Regression

Regularization prevents overfitting by adding penalty terms to the cost function.

#### L1 and L2 Regularization

**L1 (Lasso):** $J(w) = J_0(w) + \lambda \sum_{i=1}^{n} |w_i|$

**L2 (Ridge):** $J(w) = J_0(w) + \lambda \sum_{i=1}^{n} w_i^2$

```python
from sklearn.model_selection import validation_curve

# Compare different regularization strengths
C_values = np.logspace(-3, 3, 7)  # C is inverse of regularization strength

# L1 Regularization
train_scores_l1, val_scores_l1 = validation_curve(
    LogisticRegression(penalty='l1', solver='liblinear'),
    X_train_scaled, y_train_iris,
    param_name='C', param_range=C_values, cv=5
)

# L2 Regularization  
train_scores_l2, val_scores_l2 = validation_curve(
    LogisticRegression(penalty='l2'),
    X_train_scaled, y_train_iris,
    param_name='C', param_range=C_values, cv=5
)

# Plot validation curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# L1 Regularization
train_mean_l1 = np.mean(train_scores_l1, axis=1)
val_mean_l1 = np.mean(val_scores_l1, axis=1)

ax1.semilogx(C_values, train_mean_l1, 'o-', color='blue', label='Training')
ax1.semilogx(C_values, val_mean_l1, 'o-', color='red', label='Validation')
ax1.set_xlabel('C Parameter')
ax1.set_ylabel('Accuracy')
ax1.set_title('L1 Regularization (Lasso)')
ax1.legend()
ax1.grid(True)

# L2 Regularization
train_mean_l2 = np.mean(train_scores_l2, axis=1)
val_mean_l2 = np.mean(val_scores_l2, axis=1)

ax2.semilogx(C_values, train_mean_l2, 'o-', color='blue', label='Training')
ax2.semilogx(C_values, val_mean_l2, 'o-', color='red', label='Validation')
ax2.set_xlabel('C Parameter')
ax2.set_ylabel('Accuracy')
ax2.set_title('L2 Regularization (Ridge)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Find optimal C values
optimal_C_l1 = C_values[np.argmax(val_mean_l1)]
optimal_C_l2 = C_values[np.argmax(val_mean_l2)]

print(f"Optimal C for L1: {optimal_C_l1:.3f}")
print(f"Optimal C for L2: {optimal_C_l2:.3f}")
```

## Statistical Foundations of Classification Evaluation

Model evaluation in classification requires understanding the statistical properties of performance metrics and their interpretations. Each metric captures different aspects of model behavior, and the choice depends on the problem context and class distribution.

**The Fundamental Evaluation Framework**

Classification evaluation is based on the **confusion matrix**, which cross-tabulates predicted versus actual labels. For binary classification:

```
                Predicted
              0      1
Actual    0  TN     FP
          1  FN     TP
```

Where:
- **TP (True Positives)**: Correctly predicted positive cases
- **TN (True Negatives)**: Correctly predicted negative cases  
- **FP (False Positives)**: Incorrectly predicted as positive (Type I error)
- **FN (False Negatives)**: Incorrectly predicted as negative (Type II error)

**Statistical Interpretation of Errors**

- **Type I Error (α)**: P(Predict Positive | Actual Negative) = FP/(FP + TN)
- **Type II Error (β)**: P(Predict Negative | Actual Positive) = FN/(FN + TP)

The confusion matrix forms the mathematical foundation for all classification metrics.

```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns

# Train multiple models for comparison
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Train all models and collect predictions
model_predictions = {}
model_probabilities = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_iris)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)
    
    model_predictions[name] = predictions
    model_probabilities[name] = probabilities

# Create confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, predictions) in enumerate(model_predictions.items()):
    cm = confusion_matrix(y_test_iris, predictions)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names,
                yticklabels=iris.target_names,
                ax=axes[idx])
    axes[idx].set_title(f'{name} - Confusion Matrix')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.show()
```

### Mathematical Definitions of Performance Metrics

**Accuracy: Overall Correctness**

**Accuracy = (TP + TN) / (TP + TN + FP + FN)**

Accuracy measures the proportion of correct predictions. However, it can be misleading with imbalanced classes due to the **accuracy paradox**.

**Precision: Positive Predictive Value**  

**Precision = TP / (TP + FP)**

Precision answers: "Of all positive predictions, how many were actually correct?" High precision minimizes false alarms.

**Recall (Sensitivity): True Positive Rate**

**Recall = TP / (TP + FN)**

Recall answers: "Of all actual positives, how many did we correctly identify?" High recall minimizes missed detections.

**F1-Score: Harmonic Mean of Precision and Recall**

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**

F1-score provides a balanced measure when precision and recall are both important. The harmonic mean penalizes extreme values more than arithmetic mean.

**Specificity: True Negative Rate**

**Specificity = TN / (TN + FP)**

Specificity measures the ability to correctly identify negative cases.

**Statistical Trade-offs**

There exists a fundamental **precision-recall trade-off**: improving one often decreases the other. The optimal balance depends on the relative costs of Type I and Type II errors in your application domain.

```python
# Calculate comprehensive metrics for all models
evaluation_results = {}

for name, predictions in model_predictions.items():
    metrics = {
        'Accuracy': accuracy_score(y_test_iris, predictions),
        'Precision (macro)': precision_score(y_test_iris, predictions, average='macro'),
        'Recall (macro)': recall_score(y_test_iris, predictions, average='macro'),
        'F1-Score (macro)': f1_score(y_test_iris, predictions, average='macro')
    }
    evaluation_results[name] = metrics

# Create comparison DataFrame
eval_df = pd.DataFrame(evaluation_results).T
print("Classification Metrics Comparison:")
print(eval_df.round(4))

# Visualize metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']

for idx, metric in enumerate(metrics):
    ax = axes[idx//2, idx%2]
    values = eval_df[metric]
    bars = ax.bar(range(len(values)), values, alpha=0.7)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(eval_df.index, rotation=45)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

#### Detailed Per-Class Metrics

```python
# Detailed classification reports
print("Detailed Classification Reports:")
print("=" * 60)

for name, predictions in model_predictions.items():
    print(f"\n{name}:")
    print("-" * 40)
    report = classification_report(y_test_iris, predictions, 
                                 target_names=iris.target_names)
    print(report)
```

### 4.6.3 ROC Curves and AUC

For binary and multi-class ROC analysis:

```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Binarize the output for multi-class ROC
y_test_binarized = label_binarize(y_test_iris, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Plot ROC curves for each model
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for idx, (name, probabilities) in enumerate(model_probabilities.items()):
    ax = axes[idx]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves for each class
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {iris.target_names[i]} (area = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} - ROC Curves')
    ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.show()

# Calculate average AUC for each model
avg_auc_scores = {}
for name, probabilities in model_probabilities.items():
    auc_scores = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probabilities[:, i])
        auc_scores.append(auc(fpr, tpr))
    avg_auc_scores[name] = np.mean(auc_scores)

print("Average AUC Scores:")
for name, score in sorted(avg_auc_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<20}: {score:.4f}")
```

### 4.6.4 Cross-Validation and Statistical Significance

```python
from sklearn.model_selection import cross_validate
from scipy import stats

# Perform comprehensive cross-validation
cv_results = {}

for name, model in models.items():
    cv_result = cross_validate(model, X_iris, y_iris, cv=10, 
                              scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                              return_train_score=True)
    cv_results[name] = cv_result

# Extract and display results
cv_summary = {}
for name, results in cv_results.items():
    cv_summary[name] = {
        'CV Accuracy': results['test_accuracy'].mean(),
        'CV Accuracy Std': results['test_accuracy'].std(),
        'CV Precision': results['test_precision_macro'].mean(),
        'CV Recall': results['test_recall_macro'].mean(),
        'CV F1-Score': results['test_f1_macro'].mean()
    }

cv_df = pd.DataFrame(cv_summary).T
print("Cross-Validation Results (10-fold):")
print(cv_df.round(4))

# Statistical significance test (paired t-test)
print("\nPairwise Model Comparison (Accuracy):")
print("=" * 50)

model_names = list(cv_results.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        model1, model2 = model_names[i], model_names[j]
        scores1 = cv_results[model1]['test_accuracy']
        scores2 = cv_results[model2]['test_accuracy']
        
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        print(f"{model1} vs {model2}:")
        print(f"  Mean diff: {np.mean(scores1 - scores2):.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print()
```

### 4.6.5 Learning Curves

```python
from sklearn.model_selection import learning_curve

# Generate learning curves for best performing model
best_model_name = max(avg_auc_scores, key=avg_auc_scores.get)
best_model = models[best_model_name]

print(f"Generating learning curve for best model: {best_model_name}")

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_iris, y_iris, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                 color='blue', alpha=0.1)

plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                 color='red', alpha=0.1)

plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title(f'Learning Curve - {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check for overfitting/underfitting
final_gap = train_mean[-1] - val_mean[-1]
print(f"Final training-validation gap: {final_gap:.4f}")
if final_gap > 0.05:
    print("⚠️  Possible overfitting detected")
elif val_mean[-1] < 0.8:
    print("⚠️  Possible underfitting detected")
else:
    print("✅ Model appears well-fitted")
```
## 4.7 Algorithm Comparison and Selection

### 4.7.1 Comprehensive Algorithm Comparison

```python
# Final comprehensive comparison of all algorithms
final_comparison = {}

algorithms = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Evaluate on multiple datasets
from sklearn.datasets import load_breast_cancer, load_wine

datasets = {
    'Iris': (X_iris, y_iris),
    'Breast Cancer': load_breast_cancer(return_X_y=True),
    'Wine': load_wine(return_X_y=True)
}

comparison_results = {}

for dataset_name, (X, y) in datasets.items():
    print(f"Evaluating on {dataset_name} dataset...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    dataset_results = {}
    
    for alg_name, algorithm in algorithms.items():
        # Cross-validation
        cv_scores = cross_val_score(algorithm, X_scaled, y, cv=5, scoring='accuracy')
        
        dataset_results[alg_name] = {
            'Mean CV Accuracy': cv_scores.mean(),
            'Std CV Accuracy': cv_scores.std()
        }
    
    comparison_results[dataset_name] = dataset_results

# Display results
print("\nAlgorithm Performance Across Datasets:")
print("=" * 60)

for dataset_name, results in comparison_results.items():
    print(f"\n{dataset_name} Dataset:")
    df_temp = pd.DataFrame(results).T
    print(df_temp.round(4))
    
    # Find best algorithm for this dataset
    best_alg = df_temp['Mean CV Accuracy'].idxmax()
    best_score = df_temp.loc[best_alg, 'Mean CV Accuracy']
    print(f"Best Algorithm: {best_alg} ({best_score:.4f})")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (dataset_name, results) in enumerate(comparison_results.items()):
    df_plot = pd.DataFrame(results).T
    
    bars = axes[idx].bar(range(len(df_plot)), df_plot['Mean CV Accuracy'])
    axes[idx].errorbar(range(len(df_plot)), df_plot['Mean CV Accuracy'], 
                      yerr=df_plot['Std CV Accuracy'], fmt='none', 
                      color='black', capsize=5)
    
    axes[idx].set_title(f'{dataset_name} Dataset')
    axes[idx].set_ylabel('Cross-Validation Accuracy')
    axes[idx].set_xticks(range(len(df_plot)))
    axes[idx].set_xticklabels(df_plot.index, rotation=45)
    axes[idx].set_ylim(0, 1.1)
    
    # Highlight best performer
    best_idx = df_plot['Mean CV Accuracy'].argmax()
    bars[best_idx].set_color('gold')

plt.tight_layout()
plt.show()
```

### 4.7.2 Algorithm Selection Guidelines

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Decision Tree** | Interpretable models, mixed data types | Easy to understand, handles missing values | Prone to overfitting, unstable |
| **Random Forest** | General-purpose, feature importance | Reduces overfitting, robust | Less interpretable, memory intensive |
| **KNN** | Small datasets, local patterns | Simple, no assumptions | Computationally expensive, sensitive to scaling |
| **SVM** | High-dimensional data, non-linear patterns | Effective in high dimensions, memory efficient | Slow on large datasets, requires scaling |
| **Logistic Regression** | Linear relationships, probability estimates | Fast, interpretable, probabilistic | Assumes linear relationships |

```python
# Decision tree for algorithm selection
def recommend_algorithm(dataset_size, interpretability_needed, data_linearity, computational_budget):
    """
    Simple algorithm recommendation system
    """
    recommendations = []
    
    if interpretability_needed == "high":
        if dataset_size == "small":
            recommendations.append("Decision Tree")
        else:
            recommendations.append("Logistic Regression")
    
    if data_linearity == "linear":
        recommendations.extend(["Logistic Regression", "SVM (linear)"])
    else:
        recommendations.extend(["Random Forest", "SVM (RBF)", "KNN"])
    
    if computational_budget == "low":
        recommendations = [alg for alg in recommendations if alg not in ["KNN", "SVM (RBF)"]]
        recommendations.append("Logistic Regression")
    
    if dataset_size == "large":
        recommendations = [alg for alg in recommendations if alg != "KNN"]
    
    return list(set(recommendations))

# Example recommendations
print("Algorithm Recommendation Examples:")
print("=" * 40)

scenarios = [
    {"size": "small", "interpretability": "high", "linearity": "linear", "budget": "high"},
    {"size": "large", "interpretability": "low", "linearity": "non-linear", "budget": "medium"},
    {"size": "medium", "interpretability": "medium", "linearity": "unknown", "budget": "low"}
]

for i, scenario in enumerate(scenarios, 1):
    recommendations = recommend_algorithm(
        scenario["size"], scenario["interpretability"], 
        scenario["linearity"], scenario["budget"]
    )
    print(f"\nScenario {i}: {scenario}")
    print(f"Recommended: {', '.join(recommendations)}")
```

## 4.8 Best Practices

### 4.8.1 Data Preparation Checklist

```python
def classification_preprocessing_checklist():
    checklist = [
        "☐ Handle missing values appropriately",
        "☐ Encode categorical variables (one-hot, label encoding)",
        "☐ Scale/normalize features (especially for KNN, SVM, LogReg)",
        "☐ Check for class imbalance",
        "☐ Remove or handle outliers",
        "☐ Feature selection/engineering",
        "☐ Split data properly (train/validation/test)",
        "☐ Ensure no data leakage"
    ]
    
    print("Classification Data Preparation Checklist:")
    print("=" * 45)
    for item in checklist:
        print(item)

classification_preprocessing_checklist()
```

### 4.8.2 Model Selection Process

```python
def model_selection_workflow():
    steps = [
        "1. Start with simple baselines (Logistic Regression, Decision Tree)",
        "2. Try ensemble methods (Random Forest)",
        "3. Experiment with different algorithms (SVM, KNN)",
        "4. Tune hyperparameters using cross-validation", 
        "5. Evaluate using multiple metrics",
        "6. Check for overfitting/underfitting",
        "7. Test final model on held-out test set",
        "8. Consider business constraints (interpretability, speed)"
    ]
    
    print("Model Selection Workflow:")
    print("=" * 30)
    for step in steps:
        print(step)

model_selection_workflow()
```

## Theoretical and Practical Synthesis of Classification

**1. Statistical Learning Foundation**: Classification algorithms approximate the Bayes optimal classifier P(Y|X), each making different assumptions about the data generating process and decision boundaries.

**2. Algorithm-Specific Theoretical Strengths**:
   - **Decision Trees**: Information-theoretic splitting using entropy/Gini, naturally handle feature interactions
   - **KNN**: Non-parametric with universal consistency guarantees, assumes local smoothness
   - **SVM**: Maximum margin principle with optimal separating hyperplane, kernel trick for non-linearity
   - **Logistic Regression**: Probabilistic GLM with convex optimization and well-calibrated probabilities

**3. Mathematical Evaluation Framework**: Performance metrics derive from the confusion matrix, each capturing different aspects of Type I/II errors with statistical interpretation.

**4. Bias-Variance Considerations**: Different algorithms exhibit different bias-variance trade-offs - understanding these helps with algorithm selection and hyperparameter tuning.

**5. Computational Complexity**: Training complexities vary dramatically (O(n log n) for trees, O(n²) for KNN, O(n³) for SVM), affecting scalability decisions.

**6. No Free Lunch Theorem**: No universally best algorithm exists - optimal choice depends on data distribution, noise level, sample size, and interpretability requirements.

## 4.10 Exercises

### Exercise 4.1: Decision Tree Analysis
Using the Titanic dataset:
1. Build a decision tree to predict survival
2. Visualize the tree and interpret the rules
3. Compare different pruning strategies
4. Analyze feature importance

### Exercise 4.2: KNN Optimization
With the Wine dataset:
1. Find the optimal K using cross-validation
2. Compare different distance metrics
3. Analyze the effect of feature scaling
4. Implement weighted KNN from scratch

### Exercise 4.3: SVM Kernel Comparison
Using a synthetic non-linear dataset:
1. Create data that's not linearly separable
2. Compare linear, polynomial, and RBF kernels
3. Tune hyperparameters using grid search
4. Visualize decision boundaries

### Exercise 4.4: Comprehensive Comparison
Choose a real-world classification problem:
1. Apply all algorithms covered in this chapter
2. Perform proper preprocessing and feature engineering
3. Use appropriate evaluation metrics
4. Create a detailed comparison report
5. Justify your final algorithm choice

### Exercise 4.5: Imbalanced Classification
Using a highly imbalanced dataset:
1. Identify the class imbalance problem
2. Try different sampling techniques
3. Use appropriate evaluation metrics
4. Compare algorithm performance before and after handling imbalance

---

*This completes Chapter 4: Classification Algorithms. The next chapter will cover Regression Algorithms, exploring continuous prediction problems and their evaluation methods.*
