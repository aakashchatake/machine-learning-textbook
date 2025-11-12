# Chapter 7: Dimensionality Reduction

## Learning Outcomes
**CO5 - Apply unsupervised learning models**

By the end of this chapter, students will be able to:
- Understand the curse of dimensionality and its impact on machine learning
- Apply Principal Component Analysis (PCA) for dimensionality reduction
- Implement t-SNE for high-dimensional data visualization
- Use Linear Discriminant Analysis (LDA) for supervised dimensionality reduction
- Select appropriate dimensionality reduction techniques for different scenarios
- Integrate dimensionality reduction with clustering and classification pipelines

---

## Chapter Overview: The Art of Seeing in Higher Dimensions

*"The most beautiful thing we can experience is the mysterious. It is the source of all true art and science."* — Albert Einstein

Imagine standing in a vast, invisible cathedral where each pillar represents a dimension of your data. In this sacred space of machine learning, we often find ourselves overwhelmed by thousands, sometimes millions of these pillars—each feature a voice in a complex symphony of information. Yet, like a master conductor who can hear the essential melody beneath the orchestral complexity, dimensionality reduction allows us to distill this cacophony into pure, meaningful harmony.

This chapter is your journey into the profound art of **seeing patterns in the unseen**. We'll explore how mathematical elegance meets computational necessity, where ancient geometric principles guide modern algorithms, and where the reduction of complexity reveals hidden beauty in data.

### The Mathematical Poetry of Dimensionality Reduction

**What awaits you in this chapter:**
- **The Philosophical Foundation**: Understanding why "more" isn't always "better" in the mathematical universe
- **PCA as Mathematical Archeology**: Uncovering the principal stories hidden in your data's covariance structure  
- **t-SNE as Digital Artistry**: Painting high-dimensional landscapes on two-dimensional canvases
- **LDA as Supervised Wisdom**: Learning to see differences that matter most
- **The Future Landscape**: Emerging techniques that push the boundaries of dimensional understanding

### Where This Journey Takes You
- **Data Whispering**: Learning to hear what your data is really saying beneath the noise
- **Computational Alchemy**: Transforming complex, unwieldy datasets into actionable insights
- **Visual Storytelling**: Creating compelling narratives through dimensional projection
- **Pattern Recognition Mastery**: Developing intuition for what matters in high-dimensional spaces
- **Future-Ready Skills**: Preparing for the next evolution in unsupervised learning

*In this chapter, we don't just learn algorithms—we develop the artistic intuition of a data scientist who sees beyond dimensions.*

---

## 7.1 The Curse of Dimensionality: A Mathematical Paradox

### 7.1.1 The Beautiful Tragedy of High-Dimensional Spaces

*"In higher dimensions, intuition goes to die, but mathematics comes alive."* — Anonymous Data Scientist

Picture this: You're an explorer in a mathematical universe where each step forward adds another dimension to your world. At first, moving from 1D to 2D to 3D feels natural—we can visualize, touch, and understand these spaces. But as you venture into the 10th dimension, then the 100th, then the 1000th, something magical and terrifying happens: the very fabric of space begins to betray your intuition.

This is the **curse of dimensionality**—not merely a technical challenge, but a profound philosophical statement about the nature of space, distance, and meaning in mathematics. It's a phenomenon so counterintuitive that it forced mathematicians to rebuild their understanding of geometry itself.

### The Paradox That Changed Everything

In our three-dimensional world, if you double the radius of a sphere, its volume increases by a factor of 8 (2³). Intuitive, right? But in higher dimensions, something almost mystical occurs: **most of a hypersphere's volume concentrates in a thin shell near its surface**. The interior becomes increasingly empty as dimensions grow.

This isn't just mathematical curiosity—it's the reason why your machine learning algorithms sometimes seem to lose their way in high-dimensional space, why distances become meaningless, and why the very concept of "similarity" requires redefinition.

#### **Key Problems with High-Dimensional Data:**

**1. Exponential Growth of Space**
```python
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_volume_growth():
    """Demonstrate how volume grows with dimensions"""
    dimensions = range(1, 21)
    volumes = []
    
    # Calculate volume of unit hypersphere in d dimensions
    for d in dimensions:
        if d == 1:
            volume = 2  # Line segment [-1, 1]
        elif d == 2:
            volume = np.pi  # Circle with radius 1
        else:
            # Hypersphere volume formula
            from math import gamma
            volume = (np.pi**(d/2)) / gamma(d/2 + 1)
        volumes.append(volume)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, volumes, 'bo-')
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Unit Hypersphere Volume')
    plt.title('Volume Growth in High Dimensions')
    plt.grid(True)
    
    # Demonstrate distance distribution
    np.random.seed(42)
    dimensions_to_test = [2, 10, 50, 100]
    
    plt.subplot(1, 2, 2)
    for d in dimensions_to_test:
        # Generate random points and calculate pairwise distances
        points = np.random.normal(0, 1, (1000, d))
        distances = []
        
        for i in range(100):  # Sample pairs
            dist = np.linalg.norm(points[i] - points[i+1])
            distances.append(dist)
        
        plt.hist(distances, bins=20, alpha=0.6, label=f'D={d}', density=True)
    
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('Distance Distribution in Different Dimensions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

demonstrate_volume_growth()
```

**2. Distance Concentration**
In high dimensions, all points become approximately equidistant from each other.

```python
def analyze_distance_concentration():
    """Analyze how distances concentrate in high dimensions"""
    
    np.random.seed(42)
    dimensions = [2, 5, 10, 20, 50, 100]
    results = []
    
    for d in dimensions:
        # Generate random points
        n_points = 1000
        points = np.random.normal(0, 1, (n_points, d))
        
        # Calculate all pairwise distances
        distances = []
        for i in range(min(100, n_points-1)):  # Sample for efficiency
            for j in range(i+1, min(i+11, n_points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        # Calculate concentration metrics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        coefficient_variation = std_dist / mean_dist
        
        results.append({
            'dimension': d,
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'coefficient_variation': coefficient_variation
        })
    
    # Plot results
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(df['dimension'], df['mean_distance'], 'bo-')
    axes[0].set_xlabel('Dimensions')
    axes[0].set_ylabel('Mean Distance')
    axes[0].set_title('Mean Distance vs Dimensions')
    axes[0].grid(True)
    
    axes[1].plot(df['dimension'], df['std_distance'], 'ro-')
    axes[1].set_xlabel('Dimensions')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('Distance Variation vs Dimensions')
    axes[1].grid(True)
    
    axes[2].plot(df['dimension'], df['coefficient_variation'], 'go-')
    axes[2].set_xlabel('Dimensions')
    axes[2].set_ylabel('Coefficient of Variation')
    axes[2].set_title('Distance Concentration (Lower = More Concentrated)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Distance Concentration Analysis:")
    print(df.round(4))
    
    return df

concentration_results = analyze_distance_concentration()
```

### 7.1.2 Impact on Machine Learning Algorithms

#### **1. K-Nearest Neighbors (KNN) Degradation**
```python
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def demonstrate_knn_degradation():
    """Show how KNN performance degrades with increasing dimensions"""
    
    np.random.seed(42)
    n_samples = 1000
    dimensions_to_test = [2, 5, 10, 20, 50, 100, 200]
    
    results = []
    
    for n_features in dimensions_to_test:
        print(f"Testing {n_features} dimensions...")
        
        # Generate classification dataset
        X, y = make_classification(n_samples=n_samples, 
                                 n_features=n_features,
                                 n_informative=min(n_features, 10),
                                 n_redundant=0,
                                 n_clusters_per_class=1,
                                 random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test KNN performance
        knn = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        
        results.append({
            'dimensions': n_features,
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std()
        })
    
    # Plot results
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(df_results['dimensions'], df_results['mean_accuracy'],
                yerr=df_results['std_accuracy'], marker='o', capsize=5)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('KNN Accuracy')
    plt.title('KNN Performance Degradation with Increasing Dimensions')
    plt.grid(True)
    plt.show()
    
    print("\nKNN Performance vs Dimensions:")
    print(df_results.round(4))
    
    return df_results

knn_results = demonstrate_knn_degradation()
```

#### **2. Computational Complexity Issues**
```python
import time
from sklearn.cluster import KMeans

def analyze_computational_complexity():
    """Analyze computational complexity with increasing dimensions"""
    
    np.random.seed(42)
    dimensions = [5, 10, 20, 50, 100, 200]
    n_samples = 1000
    
    computation_times = []
    memory_usage = []
    
    for d in dimensions:
        print(f"Processing {d} dimensions...")
        
        # Generate data
        X = np.random.normal(0, 1, (n_samples, d))
        
        # Measure KMeans computation time
        start_time = time.time()
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(X)
        computation_time = time.time() - start_time
        
        # Estimate memory usage (rough approximation)
        memory_mb = X.nbytes / (1024 * 1024)
        
        computation_times.append(computation_time)
        memory_usage.append(memory_mb)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(dimensions, computation_times, 'bo-')
    axes[0].set_xlabel('Dimensions')
    axes[0].set_ylabel('Computation Time (seconds)')
    axes[0].set_title('KMeans Computation Time vs Dimensions')
    axes[0].grid(True)
    
    axes[1].plot(dimensions, memory_usage, 'ro-')
    axes[1].set_xlabel('Dimensions')
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].set_title('Memory Usage vs Dimensions')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create summary
    summary_df = pd.DataFrame({
        'dimensions': dimensions,
        'computation_time': computation_times,
        'memory_mb': memory_usage
    })
    
    print("\nComputational Complexity Analysis:")
    print(summary_df.round(4))
    
    return summary_df

complexity_results = analyze_computational_complexity()
```

### 7.1.3 When Dimensionality Reduction is Needed

#### **Indicators for Dimensionality Reduction:**

**1. High-Dimensional Data Symptoms**
```python
def diagnose_high_dimensional_data(X, feature_names=None):
    """Diagnose if dataset suffers from high-dimensional problems"""
    
    n_samples, n_features = X.shape
    
    print(f"=== High-Dimensional Data Diagnosis ===")
    print(f"Dataset shape: {X.shape}")
    print(f"Samples to features ratio: {n_samples/n_features:.2f}")
    
    diagnoses = []
    recommendations = []
    
    # 1. Check samples-to-features ratio
    if n_samples < n_features:
        diagnoses.append("⚠️  More features than samples (n < p problem)")
        recommendations.append("Apply dimensionality reduction or feature selection")
    elif n_samples < 10 * n_features:
        diagnoses.append("⚠️  Low samples-to-features ratio")
        recommendations.append("Consider dimensionality reduction for better generalization")
    
    # 2. Check for high sparsity
    sparsity = np.mean(X == 0)
    if sparsity > 0.8:
        diagnoses.append(f"⚠️  High sparsity ({sparsity:.1%} zeros)")
        recommendations.append("Apply sparse-aware dimensionality reduction")
    
    # 3. Check correlation structure
    correlation_matrix = np.corrcoef(X.T)
    high_correlations = np.sum(np.abs(correlation_matrix) > 0.8) - n_features  # Exclude diagonal
    if high_correlations > n_features:
        diagnoses.append(f"⚠️  Many highly correlated features ({high_correlations} pairs)")
        recommendations.append("PCA can remove redundant information")
    
    # 4. Check memory usage
    memory_mb = X.nbytes / (1024 * 1024)
    if memory_mb > 1000:  # > 1GB
        diagnoses.append(f"⚠️  Large memory footprint ({memory_mb:.1f} MB)")
        recommendations.append("Dimensionality reduction can reduce memory usage")
    
    # 5. Estimate computation time for common algorithms
    if n_features > 100:
        diagnoses.append("⚠️  High computational complexity expected")
        recommendations.append("Reduce dimensions before applying ML algorithms")
    
    print(f"\nDiagnoses:")
    for diagnosis in diagnoses:
        print(f"  {diagnosis}")
    
    print(f"\nRecommendations:")
    for recommendation in recommendations:
        print(f"  • {recommendation}")
    
    # Calculate some useful statistics
    stats = {
        'n_samples': n_samples,
        'n_features': n_features,
        'ratio': n_samples / n_features,
        'sparsity': sparsity,
        'memory_mb': memory_mb,
        'high_correlations': high_correlations
    }
    
    return stats, diagnoses, recommendations

# Example usage with different datasets
datasets = [
    ('Low-dimensional', np.random.normal(0, 1, (1000, 10))),
    ('Balanced', np.random.normal(0, 1, (1000, 50))),
    ('High-dimensional', np.random.normal(0, 1, (100, 500))),
    ('Very high-dimensional', np.random.normal(0, 1, (50, 2000)))
]

for name, X in datasets:
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    stats, diagnoses, recommendations = diagnose_high_dimensional_data(X)
```

### 7.1.4 Benefits and Trade-offs of Dimensionality Reduction

#### **Benefits:**
✅ **Computational Efficiency**: Faster training and prediction  
✅ **Memory Reduction**: Lower storage requirements  
✅ **Visualization**: Enable 2D/3D plotting of high-dimensional data  
✅ **Noise Reduction**: Remove irrelevant features and noise  
✅ **Overfitting Prevention**: Reduce model complexity  
✅ **Feature Engineering**: Create meaningful composite features  

#### **Trade-offs:**
❌ **Information Loss**: Some data variance is discarded  
❌ **Interpretability**: Transformed features may be harder to interpret  
❌ **Additional Preprocessing**: Extra computational step required  
❌ **Parameter Tuning**: Need to select number of components/dimensions  
❌ **Algorithm Selection**: Different methods suit different data types  

#### **Quantitative Analysis of Trade-offs**
```python
def analyze_dimensionality_tradeoffs(X, y=None, max_components=None):
    """Analyze trade-offs of different dimensionality reduction levels"""
    
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import time
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples, n_features = X_scaled.shape
    if max_components is None:
        max_components = min(n_features, n_samples) - 1
    
    # Test different numbers of components
    n_components_list = [2, 5, 10, 20, 50, min(100, max_components), max_components]
    n_components_list = [n for n in n_components_list if n <= max_components]
    
    results = []
    
    for n_comp in n_components_list:
        print(f"Testing {n_comp} components...")
        
        # Apply PCA
        pca = PCA(n_components=n_comp, random_state=42)
        
        start_time = time.time()
        X_reduced = pca.fit_transform(X_scaled)
        transform_time = time.time() - start_time
        
        # Calculate information retention
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        # Calculate compression ratio
        original_size = X_scaled.nbytes
        reduced_size = X_reduced.nbytes
        compression_ratio = original_size / reduced_size
        
        # If labels provided, test classification performance
        classification_score = None
        if y is not None:
            try:
                clf = LogisticRegression(random_state=42, max_iter=1000)
                scores = cross_val_score(clf, X_reduced, y, cv=3)
                classification_score = scores.mean()
            except:
                classification_score = None
        
        results.append({
            'n_components': n_comp,
            'variance_explained': variance_explained,
            'compression_ratio': compression_ratio,
            'transform_time': transform_time,
            'classification_score': classification_score,
            'memory_reduction': 1 - (reduced_size / original_size)
        })
    
    # Create visualization
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Variance explained
    axes[0, 0].plot(df_results['n_components'], df_results['variance_explained'], 'bo-')
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Variance Explained')
    axes[0, 0].set_title('Information Retention')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    axes[0, 0].legend()
    
    # Compression ratio
    axes[0, 1].plot(df_results['n_components'], df_results['compression_ratio'], 'ro-')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Compression Ratio')
    axes[0, 1].set_title('Memory Compression')
    axes[0, 1].grid(True)
    
    # Transform time
    axes[1, 0].plot(df_results['n_components'], df_results['transform_time'], 'go-')
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Transform Time (seconds)')
    axes[1, 0].set_title('Computational Efficiency')
    axes[1, 0].grid(True)
    
    # Classification performance (if available)
    if any(df_results['classification_score'].notna()):
        valid_results = df_results.dropna(subset=['classification_score'])
        axes[1, 1].plot(valid_results['n_components'], valid_results['classification_score'], 'mo-')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Classification Accuracy')
        axes[1, 1].set_title('Predictive Performance')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No classification\ntarget provided', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Classification Performance')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDimensionality Reduction Trade-off Analysis:")
    print(df_results.round(4))
    
    # Find optimal number of components
    if any(df_results['classification_score'].notna()):
        # If classification available, optimize for 95% variance + good performance
        candidates = df_results[df_results['variance_explained'] >= 0.95]
        if not candidates.empty:
            optimal = candidates.loc[candidates['classification_score'].idxmax()]
            print(f"\nRecommended components: {optimal['n_components']}")
            print(f"  Variance explained: {optimal['variance_explained']:.1%}")
            print(f"  Classification score: {optimal['classification_score']:.3f}")
            print(f"  Compression ratio: {optimal['compression_ratio']:.1f}x")
    else:
        # Optimize for elbow in variance explained curve
        # Find point where marginal gain drops significantly
        variance_gains = np.diff(df_results['variance_explained'])
        elbow_idx = np.argmax(variance_gains < 0.01) if any(variance_gains < 0.01) else len(variance_gains)
        optimal_components = df_results.iloc[elbow_idx]['n_components']
        
        print(f"\nRecommended components: {optimal_components}")
        print(f"  Variance explained: {df_results.iloc[elbow_idx]['variance_explained']:.1%}")
        print(f"  Compression ratio: {df_results.iloc[elbow_idx]['compression_ratio']:.1f}x")
    
    return df_results

# Example usage
X_example, y_example = make_classification(n_samples=1000, n_features=100, 
                                         n_informative=20, random_state=42)
tradeoff_analysis = analyze_dimensionality_tradeoffs(X_example, y_example)
```

### 7.2 Principal Component Analysis: The Art of Seeing Through Mathematical Eyes

### 7.2.1 The Dance of Variance and Dimensional Wisdom

*"In the theater of high-dimensional space, PCA is both the choreographer and the audience—it knows exactly where to look to see the most beautiful movements."*

Imagine you're a photographer trying to capture the essence of a complex, swirling dance performance. From your position, you see bodies moving in seemingly chaotic patterns, but you know that somewhere in this three-dimensional choreography lies a simpler, more beautiful story. **PCA is your magical lens**—it reveals the fundamental movements, the core rhythms that define the dance.

**Principal Component Analysis isn't just a dimensionality reduction technique—it's mathematical poetry in motion.** It whispers to us the deepest secret of high-dimensional data: that beneath apparent complexity often lies elegant simplicity, waiting to be discovered by those who know how to look.

### The Philosophy of Maximum Variance

When PCA seeks directions of maximum variance, it's not just performing a mathematical optimization—it's **asking the data to reveal its most important stories**. Variance is the language of difference, the vocabulary of variation. Where there is high variance, there are patterns, relationships, and insights waiting to be unlocked.

Think of it this way: If all your data points were identical, they would tell you nothing. It's precisely in their differences—their variance—that information lives. PCA is the master detective who can spot these differences and organize them in order of importance.

#### **Core Concepts:**

**1. Variance Maximization**
PCA seeks directions in which data varies the most. The first principal component captures maximum variance, the second captures maximum remaining variance (orthogonal to the first), and so on.

**2. Covariance Matrix**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def understand_covariance_matrix():
    """Understand covariance matrix and its role in PCA"""
    
    # Generate 2D correlated data
    np.random.seed(42)
    mean = [2, 3]
    cov = [[2, 1.5], [1.5, 1]]  # Covariance matrix
    data = np.random.multivariate_normal(mean, cov, 300)
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    print("Original Covariance Matrix:")
    print(cov_matrix)
    
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    print(f"\nEigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Data')
    plt.scatter(mean[0], mean[1], c='red', s=100, marker='x', label='Mean')
    
    # Draw eigenvectors from mean
    for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
        plt.arrow(mean[0], mean[1], vec[0]*np.sqrt(val)*2, vec[1]*np.sqrt(val)*2,
                 head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}',
                 label=f'PC{i+1} (λ={val:.2f})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data with Principal Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Show centered data
    plt.subplot(1, 2, 2)
    plt.scatter(data_centered[:, 0], data_centered[:, 1], alpha=0.6, label='Centered Data')
    
    # Draw eigenvectors from origin
    for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
        plt.arrow(0, 0, vec[0]*np.sqrt(val)*2, vec[1]*np.sqrt(val)*2,
                 head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}',
                 label=f'PC{i+1} (λ={val:.2f})')
    
    plt.xlabel('Feature 1 (centered)')
    plt.ylabel('Feature 2 (centered)')
    plt.title('Centered Data with Principal Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return data, cov_matrix, eigenvals, eigenvecs

data, cov_matrix, eigenvals, eigenvecs = understand_covariance_matrix()
```

#### **3. Eigendecomposition**
The principal components are the eigenvectors of the covariance matrix, and the eigenvalues represent the variance along each component.

```python
def step_by_step_pca_math():
    """Step-by-step mathematical derivation of PCA"""
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.multivariate_normal([0, 0], [[3, 2], [2, 2]], 100)
    
    print("Step-by-Step PCA Mathematical Process:")
    print("="*50)
    
    # Step 1: Center the data
    print("Step 1: Center the data")
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    print(f"Original mean: {X_mean}")
    print(f"Centered mean: {np.mean(X_centered, axis=0)}")
    
    # Step 2: Compute covariance matrix
    print("\nStep 2: Compute covariance matrix")
    n_samples = X_centered.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # Step 3: Eigendecomposition
    print("\nStep 3: Eigendecomposition")
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Step 4: Calculate explained variance
    print("\nStep 4: Calculate explained variance")
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    print(f"Explained variance ratio: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {np.cumsum(explained_variance_ratio)}")
    
    # Step 5: Transform data
    print("\nStep 5: Transform data to PC space")
    X_pca = X_centered @ eigenvectors
    
    print(f"Original data shape: {X.shape}")
    print(f"Transformed data shape: {X_pca.shape}")
    print(f"Variance in PC space: {np.var(X_pca, axis=0)}")
    
    # Verify: variance in PC space should equal eigenvalues
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Verification: variances match eigenvalues: {np.allclose(np.var(X_pca, axis=0, ddof=1), eigenvalues)}")
    
    # Step 6: Reconstruction
    print("\nStep 6: Data reconstruction")
    X_reconstructed = X_pca @ eigenvectors.T + X_mean
    reconstruction_error = np.mean((X - X_reconstructed)**2)
    print(f"Reconstruction error (full components): {reconstruction_error:.10f}")
    
    # Partial reconstruction (using only first component)
    X_pca_1d = X_pca[:, :1]  # Only first component
    X_reconstructed_1d = X_pca_1d @ eigenvectors[:1, :].T + X_mean
    reconstruction_error_1d = np.mean((X - X_reconstructed_1d)**2)
    print(f"Reconstruction error (1 component): {reconstruction_error_1d:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Centered data with principal components
    axes[0, 1].scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6)
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        axes[0, 1].arrow(0, 0, vec[0]*np.sqrt(val)*2, vec[1]*np.sqrt(val)*2,
                        head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}',
                        label=f'PC{i+1}')
    axes[0, 1].set_title('Centered Data with PCs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Data in PC space
    axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    axes[1, 0].set_xlabel('First Principal Component')
    axes[1, 0].set_ylabel('Second Principal Component')
    axes[1, 0].set_title('Data in Principal Component Space')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reconstruction comparison
    axes[1, 1].scatter(X[:, 0], X[:, 1], alpha=0.6, label='Original')
    axes[1, 1].scatter(X_reconstructed_1d[:, 0], X_reconstructed_1d[:, 1], 
                      alpha=0.6, label='Reconstructed (1 PC)')
    axes[1, 1].set_title('Original vs Reconstructed (1 PC)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return X, X_pca, eigenvectors, eigenvalues

X, X_pca, eigenvectors, eigenvalues = step_by_step_pca_math()
```

### 7.2.2 PCA Algorithm Implementation

#### **From Scratch Implementation**
```python
class PCAFromScratch:
    """Principal Component Analysis implementation from scratch"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, X):
        """Fit PCA to data"""
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        if self.n_components is None:
            self.n_components = len(eigenvalues)
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        """Transform data to principal component space"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct original data from transformed data"""
        return X_transformed @ self.components_ + self.mean_
    
    def get_covariance(self):
        """Get the covariance matrix of the original data"""
        return (self.components_.T * self.explained_variance_) @ self.components_

# Test custom PCA implementation
def test_custom_pca():
    """Test our custom PCA implementation"""
    
    # Generate test data
    np.random.seed(42)
    X_test = np.random.multivariate_normal([1, 2], [[2, 1.5], [1.5, 1]], 200)
    
    # Apply custom PCA
    pca_custom = PCAFromScratch(n_components=2)
    X_transformed_custom = pca_custom.fit_transform(X_test)
    
    # Apply scikit-learn PCA for comparison
    from sklearn.decomposition import PCA
    pca_sklearn = PCA(n_components=2)
    X_transformed_sklearn = pca_sklearn.fit_transform(X_test)
    
    print("Custom PCA vs Scikit-learn PCA Comparison:")
    print("="*50)
    
    print(f"Explained variance ratio (Custom): {pca_custom.explained_variance_ratio_}")
    print(f"Explained variance ratio (Sklearn): {pca_sklearn.explained_variance_ratio_}")
    
    print(f"Components shape (Custom): {pca_custom.components_.shape}")
    print(f"Components shape (Sklearn): {pca_sklearn.components_.shape}")
    
    # Check if components are the same (allowing for sign flip)
    components_match = np.allclose(np.abs(pca_custom.components_), 
                                  np.abs(pca_sklearn.components_), atol=1e-10)
    print(f"Components match: {components_match}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(X_test[:, 0], X_test[:, 1], alpha=0.6)
    axes[0].set_title('Original Data')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_transformed_custom[:, 0], X_transformed_custom[:, 1], alpha=0.6)
    axes[1].set_title('Custom PCA')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(X_transformed_sklearn[:, 0], X_transformed_sklearn[:, 1], alpha=0.6)
    axes[2].set_title('Scikit-learn PCA')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pca_custom, pca_sklearn

pca_custom, pca_sklearn = test_custom_pca()
```

#### **Efficient Implementation for Large Datasets**
```python
def efficient_pca_methods():
    """Compare different PCA computation methods for efficiency"""
    
    from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
    import time
    
    # Generate large dataset
    np.random.seed(42)
    n_samples, n_features = 5000, 1000
    X_large = np.random.normal(0, 1, (n_samples, n_features))
    
    methods = {
        'Standard PCA': PCA(n_components=50, random_state=42),
        'Incremental PCA': IncrementalPCA(n_components=50, batch_size=500),
        'Truncated SVD': TruncatedSVD(n_components=50, random_state=42)
    }
    
    results = {}
    
    print("Efficiency Comparison for Large Dataset:")
    print(f"Dataset shape: {X_large.shape}")
    print("="*50)
    
    for method_name, method in methods.items():
        print(f"\nTesting {method_name}...")
        
        # Measure fitting time
        start_time = time.time()
        X_transformed = method.fit_transform(X_large)
        fit_time = time.time() - start_time
        
        # Measure memory usage (approximate)
        memory_usage = X_transformed.nbytes / (1024**2)  # MB
        
        results[method_name] = {
            'fit_time': fit_time,
            'memory_usage': memory_usage,
            'explained_variance': getattr(method, 'explained_variance_ratio_', None)
        }
        
        print(f"  Fit time: {fit_time:.3f} seconds")
        print(f"  Memory usage: {memory_usage:.2f} MB")
        if hasattr(method, 'explained_variance_ratio_'):
            print(f"  Total variance explained: {np.sum(method.explained_variance_ratio_):.3f}")
    
    # Visualize comparison
    methods_list = list(results.keys())
    fit_times = [results[m]['fit_time'] for m in methods_list]
    memory_usage = [results[m]['memory_usage'] for m in methods_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(methods_list, fit_times, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0].set_ylabel('Fit Time (seconds)')
    axes[0].set_title('Computation Time Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(methods_list, memory_usage, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].set_title('Memory Usage Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

efficiency_results = efficient_pca_methods()
```

### 7.2.3 Selecting Number of Components

#### **1. Explained Variance Method**
```python
def analyze_explained_variance(X, max_components=None):
    """Analyze explained variance to select optimal number of components"""
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine maximum components
    n_samples, n_features = X_scaled.shape
    if max_components is None:
        max_components = min(n_samples, n_features)
    
    # Fit PCA with all components
    pca_full = PCA(n_components=max_components)
    pca_full.fit(X_scaled)
    
    # Calculate cumulative explained variance
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find components needed for different variance thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    threshold_components = []
    
    for threshold in thresholds:
        n_comp = np.argmax(cumulative_variance >= threshold) + 1
        threshold_components.append(n_comp)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Individual explained variance
    axes[0, 0].bar(range(1, min(21, len(explained_variance_ratio)+1)), 
                   explained_variance_ratio[:20], alpha=0.7)
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('Individual Component Variance (First 20)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative explained variance
    components_range = range(1, len(cumulative_variance) + 1)
    axes[0, 1].plot(components_range, cumulative_variance, 'b-o', markersize=3)
    
    # Add threshold lines
    colors = ['red', 'orange', 'green', 'purple']
    for threshold, n_comp, color in zip(thresholds, threshold_components, colors):
        axes[0, 1].axhline(y=threshold, color=color, linestyle='--', alpha=0.7, 
                          label=f'{threshold:.0%} ({n_comp} comp)')
        axes[0, 1].axvline(x=n_comp, color=color, linestyle='--', alpha=0.7)
    
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scree plot (eigenvalues)
    eigenvalues = pca_full.explained_variance_
    axes[1, 0].plot(range(1, min(21, len(eigenvalues)+1)), eigenvalues[:20], 'ro-')
    axes[1, 0].set_xlabel('Principal Component')
    axes[1, 0].set_ylabel('Eigenvalue')
    axes[1, 0].set_title('Scree Plot (First 20 Components)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Elbow detection
    if len(eigenvalues) > 3:
        # Calculate second derivative to find elbow
        second_derivative = np.gradient(np.gradient(eigenvalues[:20]))
        elbow_idx = np.argmax(second_derivative) + 1 if any(second_derivative > 0) else len(second_derivative)
        optimal_components = df_results.iloc[elbow_idx]['n_components']
        
        axes[1, 0].axvline(x=elbow_idx, color='green', linestyle='--', 
                          label=f'Elbow at {elbow_idx}')
        axes[1, 0].legend()
    
    # Component selection summary
    axes[1, 1].axis('off')
    summary_text = "Component Selection Summary:\n\n"
    for threshold, n_comp in zip(thresholds, threshold_components):
        summary_text += f"{threshold:.0%} variance: {n_comp} components\n"
    
    if len(eigenvalues) > 3:
        summary_text += f"\nElbow method suggests: {elbow_idx} components\n"
    
    # Add practical recommendations
    summary_text += "\nRecommendations:\n"
    summary_text += f"• For visualization: 2-3 components\n"
    summary_text += f"• For preprocessing: {threshold_components[1]} components (90%)\n"
    summary_text += f"• For high accuracy: {threshold_components[2]} components (95%)\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Return analysis results
    analysis_results = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'threshold_components': dict(zip(thresholds, threshold_components)),
        'eigenvalues': eigenvalues
    }
    
    if len(eigenvalues) > 3:
        analysis_results['elbow_point'] = elbow_idx
    
    return analysis_results

# Example usage with different datasets
datasets = {
    'Random Data': np.random.normal(0, 1, (500, 50)),
    'Correlated Data': None  # Will generate correlated data
}

# Generate correlated data
np.random.seed(42)
base_data = np.random.normal(0, 1, (500, 10))
noise = np.random.normal(0, 0.1, (500, 40))
correlated_data = np.column_stack([
    base_data,
    base_data[:, :5] + noise[:, :5],  # Correlated features
    base_data[:, :10] * 0.5 + noise[:, 5:15],  # Partially correlated
    noise[:, 15:]  # Pure noise
])
datasets['Correlated Data'] = correlated_data

for name, X in datasets.items():
    print(f"\n{'='*60}")
    print(f"Analysis for {name}")
    print(f"{'='*60}")
    
    if X is not None:
        variance_analysis = analyze_explained_variance(X, max_components=30)
```

#### **2. Cross-Validation Approach**
```python
def pca_cross_validation_selection(X, y, max_components=20):
    """Select optimal number of PCA components using cross-validation"""
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of components
    n_components_range = range(1, min(max_components + 1, X_scaled.shape[1]))
    cv_scores = []
    cv_stds = []
    
    print("Cross-Validation Component Selection:")
    print("="*40)
    
    for n_comp in n_components_range:
        # Create pipeline
        pipeline = Pipeline([
            ('pca', PCA(n_components=n_comp, random_state=42)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Cross-validation
        scores = cross_val_score(pipeline, X_scaled, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
        cv_stds.append(scores.std())
        
        if n_comp <= 10 or n_comp % 5 == 0:  # Print selected results
            print(f"  {n_comp:2d} components: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Find optimal number of components
    optimal_idx = np.argmax(cv_scores)
    optimal_components = n_components_range[optimal_idx]
    optimal_score = cv_scores[optimal_idx]
    
    print(f"\nOptimal components: {optimal_components}")
    print(f"Best CV score: {optimal_score:.4f} ± {cv_stds[optimal_idx]:.4f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(n_components_range, cv_scores, yerr=cv_stds, 
                marker='o', capsize=5, capthick=2)
    plt.axvline(x=optimal_components, color='red', linestyle='--', 
                label=f'Optimal: {optimal_components}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('PCA Component Selection via Cross-Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare with baseline (no PCA)
    baseline_pipeline = Pipeline([
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    baseline_scores = cross_val_score(baseline_pipeline, X_scaled, y, cv=5)
    baseline_mean = baseline_scores.mean()
    
    plt.subplot(1, 2, 2)
    performance_comparison = [baseline_mean, optimal_score]
    labels = ['No PCA\n(All Features)', f'PCA\n({optimal_components} Components)']
    colors = ['lightcoral', 'lightgreen']
    
    bars = plt.bar(labels, performance_comparison, color=colors)
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Performance Comparison')
    
    # Add value labels on bars
    for bar, value in zip(bars, performance_comparison):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.ylim(min(performance_comparison) - 0.05, max(performance_comparison) + 0.05)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'n_components_range': list(n_components_range),
        'cv_scores': cv_scores,
        'cv_stds': cv_stds,
        'optimal_components': optimal_components,
        'optimal_score': optimal_score,
        'baseline_score': baseline_mean
    }

# Example usage
X_example, y_example = make_classification(n_samples=1000, n_features=50, 
                                         n_informative=15, n_redundant=10,
                                         random_state=42)
cv_results = pca_cross_validation_selection(X_example, y_example, max_components=30)
```

### 7.2.4 PCA Applications and Interpretation

#### **1. Data Visualization**
```python
def pca_visualization_techniques():
    """Demonstrate PCA for data visualization"""
    
    from sklearn.datasets import load_digits, load_wine, load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Load different datasets
    datasets = {
        'Digits': load_digits(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer()
    }
    
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 5 * len(datasets)))
    
    for i, (name, dataset) in enumerate(datasets.items()):
        X, y = dataset.data, dataset.target
        
        print(f"\n{name} Dataset:")
        print(f"  Original shape: {X.shape}")
        print(f"  Number of classes: {len(np.unique(y))}")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca_2d = PCA(n_components=2, random_state=42)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        pca_3d = PCA(n_components=3, random_state=42)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        # 2D visualization
        scatter = axes[i, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[i, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
        axes[i, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
        axes[i, 0].set_title(f'{name} - 2D PCA')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Explained variance plot
        pca_full = PCA().fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        axes[i, 1].plot(range(1, len(cumvar[:20]) + 1), cumvar[:20], 'bo-')
        axes[i, 1].axhline(y=0.95, color='red', linestyle='--', label='95%')
        axes[i, 1].set_xlabel('Number of Components')
        axes[i, 1].set_ylabel('Cumulative Explained Variance')
        axes[i, 1].set_title(f'{name} - Explained Variance')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Component interpretation (feature importance)
        n_features_show = min(10, X.shape[1])
        component_importance = np.abs(pca_2d.components_[0, :n_features_show])
        feature_names = getattr(dataset, 'feature_names', 
                               [f'Feature_{j}' for j in range(X.shape[1])])
        
        axes[i, 2].barh(range(10), component_importance[top_features_pca])
        axes[i, 2].set_yticks(range(10))
        axes[i, 2].set_yticklabels([feature_names[j][:15] for j in range(n_features_show)])
        axes[i, 2].set_xlabel('Absolute Component Weight')
        axes[i, 2].set_title(f'{name} - PC1 Feature Importance')
        axes[i, 2].grid(True, alpha=0.3)
        
        print(f"  2D PCA variance explained: {np.sum(pca_2d.explained_variance_ratio_):.1%}")
        print(f"  3D PCA variance explained: {np.sum(pca_3d.explained_variance_ratio_):.1%}")
    
    plt.tight_layout()
    plt.show()

pca_visualization_techniques()
```

#### **2. Noise Reduction and Data Compression**
```python
def pca_noise_reduction_demo():
    """Demonstrate PCA for noise reduction and data compression"""
    
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Load digit data
    digits = load_digits()
    X_digits = digits.data  # 400 faces, each 64×64 pixels (4096 features)
    
    print("PCA for Noise Reduction and Compression Demo:")
    print("="*50)
    
    # Add noise to simulate real-world conditions
    np.random.seed(42)
    noise = np.random.normal(0, 2, X_digits.shape)
    X_noisy = X_digits + noise
    
    # Standardize data
    scaler = StandardScaler()
    X_noisy_scaled = scaler.fit_transform(X_noisy)
    
    # Test different numbers of components
    n_components_list = [10, 20, 50, 100, 200, 500]
    
    # Apply PCA to entire dataset
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_digits)
    
    # Analyze compression results
    compression_results = []
    
    fig, axes = plt.subplots(2, len(n_components_list), figsize=(20, 8))
    
    for i, n_comp in enumerate(n_components_list):
        # Reconstruct using n components
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_noisy_scaled)
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Calculate metrics
        mse = np.mean((X_digits - X_reconstructed) ** 2)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        compression_ratio = 4096 / (n_comp + n_comp * 4096 / 400)  # Approximate
        
        compression_results.append({
            'n_components': n_comp,
            'mse': mse,
            'variance_explained': variance_explained,
            'compression_ratio': compression_ratio
        })
        
        # Display original and reconstructed
        axes[0, i].imshow(original_face, cmap='gray')
        axes[0, i].set_title(f'Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed_face, cmap='gray')
        axes[1, i].set_title(f'{n_comp} comp\\nMSE: {mse:.4f}\\nVar: {variance_explained:.1%}')
        axes[1, i].axis('off')
        
        print(f"{n_comp:3d} components: MSE={mse:.4f}, Variance={variance_explained:.1%}, Compression={compression_ratio:.1f}x")
    
    plt.tight_layout()
    plt.show()
    
    # Analysis plots
    df_compression = pd.DataFrame(compression_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE vs Components
    axes[0].plot(df_compression['n_components'], df_compression['mse'], 'ro-')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Reconstruction Error vs Components')
    axes[0].grid(True, alpha=0.3)
    
    # Variance Explained
    axes[1].plot(df_compression['n_components'], df_compression['variance_explained'], 'bo-')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Variance Explained')
    axes[1].set_title('Information Retention vs Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Compression Trade-off
    axes[2].scatter(df_compression['compression_ratio'], df_compression['mse'], 
                   c=df_compression['n_components'], cmap='viridis', s=100)
    axes[2].set_xlabel('Compression Ratio')
    axes[2].set_ylabel('Reconstruction Error (MSE)')
    axes[2].set_title('Compression vs Quality Trade-off')
    
    # Add component labels
    for _, row in df_compression.iterrows():
        axes[2].annotate(f"{row['n_components']}", 
                        (row['compression_ratio'], row['mse']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Components')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_compression

compression_lab_results = pca_noise_reduction_demo()
```

### 7.3 Advanced Dimensionality Reduction Techniques

### 7.3.1 Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a supervised dimensionality reduction technique that finds the directions that best separate different classes. Unlike PCA, which maximizes variance, LDA maximizes class separability.

#### **Mathematical Foundation**

LDA seeks to find a projection that maximizes the ratio of between-class variance to within-class variance:

```
J(w) = (w^T S_B w) / (w^T S_W w)
```

Where:
- S_B = between-class scatter matrix
- S_W = within-class scatter matrix
- w = projection vector

#### **Implementation and Comparison with PCA**
```python
def lda_vs_pca_comparison():
    """Compare LDA and PCA for dimensionality reduction"""
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.decomposition import PCA
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    
    # Generate classification dataset with overlapping classes
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                             n_redundant=5, n_clusters_per_class=2, 
                             class_sep=0.8, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("LDA vs PCA Comparison:")
    print("="*30)
    print(f"Original data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Apply PCA and LDA
    pca = PCA(n_components=2, random_state=42)
    lda = LDA(n_components=2)
    
    X_pca = pca.fit_transform(X_scaled)
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Evaluate classification performance
    classifier = SVC(random_state=42)
    
    # Original data performance
    scores_original = cross_val_score(classifier, X_scaled, y, cv=5)
    
    # PCA performance
    scores_pca = cross_val_score(classifier, X_pca, y, cv=5)
    
    # LDA performance
    scores_lda = cross_val_score(classifier, X_lda, y, cv=5)
    
    print(f"\nClassification Performance:")
    print(f"Original (20D): {scores_original.mean():.4f} ± {scores_original.std():.4f}")
    print(f"PCA (2D):       {scores_pca.mean():.4f} ± {scores_pca.std():.4f}")
    print(f"LDA (2D):       {scores_lda.mean():.4f} ± {scores_lda.std():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PCA visualization
    scatter_pca = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0, 0].set_title('PCA Projection')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter_pca, ax=axes[0, 0])
    
    # LDA visualization
    scatter_lda = axes[0, 1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0, 1].set_xlabel('LD1')
    axes[0, 1].set_ylabel('LD2')
    axes[0, 1].set_title('LDA Projection')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter_lda, ax=axes[0, 1])
    
    # Performance comparison
    methods = ['Original\n(20D)', 'PCA\n(2D)', 'LDA\n(2D)']
    performances = [scores_original.mean(), scores_pca.mean(), scores_lda.mean()]
    errors = [scores_original.std(), scores_pca.std(), scores_lda.std()]
    
    bars = axes[0, 2].bar(methods, performances, yerr=errors, capsize=5,
                         color=['lightblue', 'lightgreen', 'lightcoral'])
    axes[0, 2].set_ylabel('Cross-Validation Accuracy')
    axes[0, 2].set_title('Performance Comparison')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, performances):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # Component analysis
    # PCA components (feature importance)
    pca_components = pca.components_
    rf_importance = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_pca, y).feature_importances_
    
    # Calculate original feature importance through PCA
    original_importance = np.abs(pca_components.T @ rf_importance)
    
    # Show top 15 features
    top_features = np.argsort(original_importance)[-15:]
    
    axes[1, 0].barh(range(15), original_importance[top_features])
    axes[1, 0].set_yticks(range(15))
    axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_features])
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].set_title('PCA Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # LDA components (discriminant weights)
    lda_importance = np.abs(lda.scalings_[:, 0])  # First discriminant
    top_features_lda = np.argsort(lda_importance)[-10:]
    
    axes[1, 1].barh(range(10), lda_importance[top_features_lda])
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_features_lda])
    axes[1, 1].set_xlabel('Absolute Discriminant Weight')
    axes[1, 1].set_title('LDA Feature Importance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Class separability analysis
    # Calculate Fisher's ratio for both methods
    def fishers_ratio(X_proj, y):
        """Calculate Fisher's ratio (between-class / within-class variance)"""
        classes = np.unique(y)
        class_means = [X_proj[y == c].mean(axis=0) for c in classes]
        overall_mean = X_proj.mean(axis=0)
        
        # Between-class variance
        between_var = sum(len(X_proj[y == c]) * np.sum((class_means[i] - overall_mean)**2) 
                         for i, c in enumerate(classes))
        
        # Within-class variance
        within_var = sum(np.sum((X_proj[y == c] - class_means[i])**2) 
                        for i, c in enumerate(classes))
        
        return between_var / within_var if within_var > 0 else 0
    
    fisher_pca = fishers_ratio(X_pca, y)
    fisher_lda = fishers_ratio(X_lda, y)
    
    fisher_ratios = [fisher_pca, fisher_lda]
    method_names = ['PCA', 'LDA']
    
    bars = axes[1, 2].bar(method_names, fisher_ratios, color=['lightgreen', 'lightcoral'])
    axes[1, 2].set_ylabel("Fisher's Ratio")
    axes[1, 2].set_title('Class Separability Comparison')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars, fisher_ratios):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{ratio:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'pca_performance': scores_pca.mean(),
        'lda_performance': scores_lda.mean(),
        'fisher_pca': fisher_pca,
        'fisher_lda': fisher_lda
    }

lda_pca_results = lda_vs_pca_comparison()
```

#### **LDA Implementation from Scratch**
```python
class LDAFromScratch:
    """Linear Discriminant Analysis implementation from scratch"""
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings_ = None
        self.means_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit LDA to training data"""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # If n_components not specified, use maximum possible
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        
        # Calculate class means
        class_means = []
        for c in self.classes_:
            class_means.append(X[y == c].mean(axis=0))
        class_means = np.array(class_means)
        
        # Overall mean
        overall_mean = X.mean(axis=0)
        
        # Between-class scatter matrix (S_B)
        S_B = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            n_c = np.sum(y == c)
            mean_diff = (class_means[i] - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)
        
        # Within-class scatter matrix (S_W)
        S_W = np.zeros((n_features, n_features))
        for c in self.classes_:
            class_data = X[y == c]
            class_mean = class_data.mean(axis=0)
            for sample in class_data:
                diff = (sample - class_mean).reshape(-1, 1)
                S_W += diff @ diff.T
        
        # Solve generalized eigenvalue problem: S_B * v = λ * S_W * v
        # This is equivalent to: S_W^(-1) * S_B * v = λ * v
        try:
            # Add small regularization to avoid singular matrix
            S_W_reg = S_W + np.eye(n_features) * 1e-6
            eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(S_W_reg) @ S_B)
            
            # Sort by eigenvalues (descending)
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Select top components
            self.scalings_ = eigenvecs[:, :self.n_components]
            self.eigenvalues_ = eigenvals[:self.n_components]
            
        except np.linalg.LinAlgError:
            # Fallback to SVD if matrix is singular
            U, s, Vt = np.linalg.svd(S_B)
            self.scalings_ = U[:, :self.n_components]
            self.eigenvalues_ = s[:self.n_components]
        
        self.means_ = class_means
        return self
    
    def transform(self, X):
        """Transform data to discriminant space"""
        return X @ self.scalings_
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

# Test custom LDA implementation
def test_custom_lda():
    """Test our custom LDA implementation"""
    
    # Generate test data
    np.random.seed(42)
    X_test, y_test = make_classification(n_samples=300, n_features=10, 
                                        n_classes=3, n_informative=8,
                                        random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    
    # Apply custom LDA
    lda_custom = LDAFromScratch(n_components=2)
    X_custom = lda_custom.fit_transform(X_scaled, y_test)
    
    # Apply sklearn LDA
    lda_sklearn = LDA(n_components=2)
    X_sklearn = lda_sklearn.fit_transform(X_scaled, y_test)
    
    print("Custom LDA vs Scikit-learn LDA:")
    print("="*35)
    print(f"Custom LDA shape: {X_custom.shape}")
    print(f"Sklearn LDA shape: {X_sklearn.shape}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data (first 2 features)
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    axes[0].set_title('Original Data (First 2 Features)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_custom[:, 0], X_custom[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    axes[1].set_title('Custom LDA')
    axes[1].set_xlabel('LD1')
    axes[1].set_ylabel('LD2')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(X_sklearn[:, 0], X_sklearn[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    axes[2].set_title('Scikit-learn LDA')
    axes[2].set_xlabel('LD1')
    axes[2].set_ylabel('LD2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return lda_custom, lda_sklearn

custom_lda, sklearn_lda = test_custom_lda()
```

### 7.3.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a non-linear dimensionality reduction technique primarily used for data visualization. It preserves local neighborhood structure and can reveal clusters that linear methods might miss.

#### **Key Concepts:**

**1. Probability Distributions**: t-SNE converts distances to probabilities
**2. Kullback-Leibler Divergence**: Minimizes difference between high-D and low-D distributions
**3. Perplexity**: Controls the effective number of neighbors

#### **t-SNE Implementation and Analysis**
```python
def tsne_comprehensive_analysis():
    """Comprehensive t-SNE analysis with parameter exploration"""
    
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits, make_swiss_roll
    from sklearn.preprocessing import StandardScaler
    
    # Load datasets
    datasets = {
        'Digits': load_digits(),
        'Swiss Roll': make_swiss_roll(n_samples=1000, random_state=42)
    }
    
    # t-SNE parameters to test
    perplexity_values = [5, 15, 30, 50]
    learning_rates = [10, 200, 1000]
    
    for dataset_name, dataset in datasets.items():
        if dataset_name == 'Swiss Roll':
            X, color = dataset
            y = color  # Use color for visualization
        else:
            X, y = dataset.data, dataset.target
        
        print(f"\n{'='*50}")
        print(f"t-SNE Analysis: {dataset_name}")
        print(f"{'='*50}")
        print(f"Data shape: {X.shape}")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test different perplexity values
        fig, axes = plt.subplots(2, len(perplexity_values), figsize=(20, 10))
        
        for i, perplexity in enumerate(perplexity_values):
            print(f"Processing perplexity = {perplexity}...")
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                       learning_rate=200, n_iter=1000)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Plot result
            scatter = axes[0, i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                                       cmap='tab10' if dataset_name == 'Digits' else 'viridis',
                                       alpha=0.7, s=20)
            axes[0, i].set_title(f'Perplexity = {perplexity}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Calculate and plot KL divergence over iterations (if available)
            # Note: sklearn doesn't expose KL divergence history, so we'll show final result
            axes[1, i].text(0.5, 0.5, f'Perplexity: {perplexity}\\nFinal KL Divergence: {tsne.kl_divergence_:.2f}',
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, i].set_title('Optimization Info')
        
        plt.suptitle(f't-SNE Perplexity Comparison - {dataset_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Compare with PCA
        print("\nComparing t-SNE with PCA...")
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Best t-SNE (perplexity=30 is usually good default)
        tsne_best = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate=200)
        X_tsne_best = tsne_best.fit_transform(X_scaled)
        
        # Visualization comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original data (first 2 features)
        if X.shape[1] >= 2:
            axes[0].scatter(X[:, 0], X[:, 1], c=y, 
                          cmap='tab10' if dataset_name == 'Digits' else 'viridis',
                          alpha=0.7)
            axes[0].set_title('Original Data (First 2 Features)')
            feature_names = getattr(dataset, 'feature_names', ['Feature 0', 'Feature 1'])
            axes[0].set_xlabel(feature_names[0][:20])
            axes[0].set_ylabel(feature_names[1][:20])
        else:
            axes[0].text(0.5, 0.5, 'Not applicable', ha='center', va='center',
                        transform=axes[0].transAxes)
            axes[0].set_title('Original Data')
        
        # PCA
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[1].set_title(f'PCA ({pca.explained_variance_ratio_.sum():.1%} var)')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        # t-SNE
        scatter = axes[2].scatter(X_tsne_best[:, 0], X_tsne_best[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[2].set_title('t-SNE (Perplexity=30)')
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results

tsne_comprehensive_analysis()
```

#### **t-SNE Parameter Optimization**
```python
def tsne_parameter_optimization():
    """Optimize t-SNE parameters for best visualization"""
    
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Parameter grid
    perplexity_range = [5, 10, 15, 20, 30, 40, 50]
    learning_rate_range = [10, 50, 100, 200, 500, 1000]
    
    print("t-SNE Parameter Optimization:")
    print("="*30)
    
    # Store results
    results = []
    
    # Test subset of combinations for efficiency
    param_combinations = [
        (5, 200), (15, 200), (30, 200), (50, 200),  # Different perplexities
        (30, 10), (30, 100), (30, 500), (30, 1000)   # Different learning rates
    ]
    
    for perplexity, learning_rate in param_combinations:
        print(f"Testing perplexity={perplexity}, learning_rate={learning_rate}")
        
        try:
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       learning_rate=learning_rate, random_state=42, 
                       n_iter=1000, verbose=0)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Evaluate clustering quality using silhouette score
            # Apply KMeans to t-SNE result
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_tsne)
            
            # Calculate silhouette score
            sil_score = silhouette_score(X_tsne, cluster_labels)
            
            # Calculate how well clusters match true labels (ARI)
            from sklearn.metrics import adjusted_rand_score
            ari_score = adjusted_rand_score(y, cluster_labels)
            
            results.append({
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'silhouette_score': sil_score,
                'ari_score': ari_score,
                'kl_divergence': tsne.kl_divergence_,
                'X_tsne': X_tsne
            })
            
            print(f"  Silhouette: {sil_score:.3f}, ARI: {ari_score:.3f}, KL: {tsne.kl_divergence_:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['silhouette_score'])
    
    print(f"\nBest parameters:")
    print(f"  Perplexity: {best_result['perplexity']}")
    print(f"  Learning Rate: {best_result['learning_rate']}")
    print(f"  Silhouette Score: {best_result['silhouette_score']:.3f}")
    print(f"  ARI Score: {best_result['ari_score']:.3f}")
    
    # Visualize parameter effects
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Perplexity effect (fixed learning rate = 200)
    perp_results = df_results[df_results['learning_rate'] == 200]
    if not perp_results.empty:
        axes[0, 0].plot(perp_results['perplexity'], perp_results['silhouette_score'], 'bo-')
        axes[0, 0].set_xlabel('Perplexity')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Perplexity Effect (LR=200)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate effect (fixed perplexity = 30)
    lr_results = df_results[df_results['perplexity'] == 30]
    if not lr_results.empty:
        axes[0, 1].plot(lr_results['learning_rate'], lr_results['silhouette_score'], 'ro-')
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Learning Rate Effect (Perp=30)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence vs quality
    axes[0, 2].scatter(df_results['kl_divergence'], df_results['silhouette_score'], 
                      c=df_results['perplexity'], cmap='viridis', s=60)
    axes[0, 2].set_xlabel('KL Divergence')
    axes[0, 2].set_ylabel('Silhouette Score')
    axes[0, 2].set_title('Convergence vs Quality')
    plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2], label='Perplexity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Show best visualizations
    # Best result
    axes[1, 0].scatter(best_result['X_tsne'][:, 0], best_result['X_tsne'][:, 1], 
                      c=y, cmap='tab10', alpha=0.7, s=20)
    axes[1, 0].set_title(f'Best Result\\n(Perp={best_result["perplexity"]}, LR={best_result["learning_rate"]})')
    
    # Comparison with different parameters
    worst_result = min(results, key=lambda x: x['silhouette_score'])
    axes[1, 1].scatter(worst_result['X_tsne'][:, 0], worst_result['X_tsne'][:, 1], 
                      c=y, cmap='tab10', alpha=0.7, s=20)
    axes[1, 1].set_title(f'Worst Result\\n(Perp={worst_result["perplexity"]}, LR={worst_result["learning_rate"]})')
    
    # Parameter space heatmap
    pivot_data = df_results.pivot_table(values='silhouette_score', 
                                       index='perplexity', 
                                       columns='learning_rate', 
                                       fill_value=np.nan)
    
    im = axes[1, 2].imshow(pivot_data.values, cmap='viridis', aspect='auto')
    axes[1, 2].set_xticks(range(len(pivot_data.columns)))
    axes[1, 2].set_xticklabels(pivot_data.columns)
    axes[1, 2].set_yticks(range(len(pivot_data.index)))
    axes[1, 2].set_yticklabels(pivot_data.index)
    axes[1, 2].set_xlabel('Learning Rate')
    axes[1, 2].set_ylabel('Perplexity')
    axes[1, 2].set_title('Parameter Heatmap')
    plt.colorbar(im, ax=axes[1, 2], label='Silhouette Score')
    
    plt.tight_layout()
    plt.show()
    
    return results, best_result
```

### 7.3.3 Feature Selection vs Feature Extraction

#### **Comparison of Approaches**
```python
def feature_selection_vs_extraction():
    """Compare feature selection and feature extraction methods"""
    
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import cross_val_score
    
    # Generate dataset with mix of informative and noisy features
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=50, 
                             n_informative=15, n_redundant=10, 
                             n_clusters_per_class=1, random_state=42)
    
    print("Feature Selection vs Feature Extraction Comparison:")
    print("="*55)
    print(f"Original dataset shape: {X.shape}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define methods
    methods = {
        'Original': None,
        'Filter Selection (f_classif)': 'f_classif',
        'Wrapper Selection (RFE)': 'rfe',
        'PCA (15 components)': 'pca_15',
        'LDA (2 components)': 'lda_2'
    }
    
    results = {}
    
    for method_name, method_type in methods.items():
        print(f"\nTesting {method_name}...")
        
        if method_type is None:
            # No reduction
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            X_train_processed = X_train
            X_test_processed = X_test
            
        elif method_type == 'f_classif':
            # Filter selection
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=10)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'rfe':
            # Wrapper selection
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', RFE(LogisticRegression(random_state=42, max_iter=1000), 
                                 n_features_to_select=10)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'pca_15':
            # PCA with 15 components
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=15, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'lda_2':
            # LDA with 2 components
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lda', LDA(n_components=2)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get effective dimensionality
        if method_type is None:
            n_features_used = X.shape[1]
        elif 'pca' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['pca'].n_components_
        elif 'lda' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['lda'].scalings_.shape[1]
        elif 'selector' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['selector'].k
        else:
            n_features_used = X.shape[1]
        
        results[method_name] = {
            'accuracy': accuracy,
            'n_features': n_features_used,
            'pipeline': pipeline,
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Features used: {n_features_used}")
        print(f"  Reduction: {(1 - n_features_used/X.shape[1])*100:.1f}%")
    
    # Step 3: Detailed Analysis
    print(f"\nStep 3: Detailed Analysis")
    print("-" * 23)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Performance comparison
    method_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in method_names]
    n_features = [results[m]['n_features'] for m in method_names]
    
    bars = axes[0, 0].bar(range(len(method_names)), accuracies, alpha=0.7)
    axes[0, 0].set_xticks(range(len(method_names)))
    axes[0, 0].set_xticklabels([name.split()[0] for name in method_names], rotation=45)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Method Performance Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Feature count comparison
    bars = axes[0, 1].bar(range(len(method_names)), n_features, alpha=0.7)
    axes[0, 1].set_xticks(range(len(method_names)))
    axes[0, 1].set_xticklabels([name.split()[0] for name in method_names], rotation=45)
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Dimensionality Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA analysis (if available)
    pca_pipeline = results['PCA (95% var)']['pipeline']
    if 'pca' in pca_pipeline.named_steps:
        pca = pca_pipeline.named_steps['pca']
        
        # Explained variance
        axes[0, 2].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                       np.cumsum(pca.explained_variance_ratio_), 'bo-')
        axes[0, 2].axhline(y=0.95, color='red', linestyle='--', label='95%')
        axes[0, 2].set_xlabel('Component')
        axes[0, 2].set_ylabel('Cumulative Variance Explained')
        axes[0, 2].set_title('PCA Variance Analysis')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Feature importance in first 2 PCs
        pc1_importance = np.abs(pca.components_[0])
        pc2_importance = np.abs(pca.components_[1])
        
        top_features_pc1 = np.argsort(pc1_importance)[-10:]
        top_features_pc2 = np.argsort(pc2_importance)[-10:]
        
        axes[1, 0].barh(range(10), pc1_importance[top_features_pc1])
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels([feature_names[i][:15] for i in top_features_pc1], fontsize=8)
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_title('PC1 - Top Feature Contributions')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].barh(range(10), pc2_importance[top_features_pc2])
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels([feature_names[i][:15] for i in top_features_pc2], fontsize=8)
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_title('PC2 - Top Feature Contributions')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy vs Dimensionality trade-off
    axes[1, 2].scatter(n_features, accuracies, s=100, alpha=0.7)
    for i, method in enumerate(method_names):
        axes[1, 2].annotate(method.split()[0], (n_features[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 2].set_xlabel('Number of Features')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy vs Dimensionality Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Method recommendations
    axes[1, 2].axis('off')
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
    most_efficient = min(results.keys(), key=lambda k: results[k]['n_features'] 
                        if results[k]['accuracy'] > 0.9 else float('inf'))
    
    recommendations = f"""
    RECOMMENDATIONS:
    
    Best Accuracy: {best_method}
    • Accuracy: {results[best_method]['accuracy']:.4f}
    • Features: {results[best_method]['n_features']}
    
    Most Efficient: {most_efficient}
    • Accuracy: {results[most_efficient]['accuracy']:.4f}
    • Features: {results[most_efficient]['n_features']}
    • Reduction: {(1-results[most_efficient]['n_features']/X.shape[1])*100:.1f}%
    
    INSIGHTS:
    • {len(high_corr_pairs)} highly correlated features
    • PCA effective for noise reduction
    • LDA good for classification tasks
    • Feature selection preserves interpretability
    """
    
    axes[1, 2].text(0.05, 0.95, recommendations, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification reports
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORTS:")
    print(f"{'='*60}")
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print("-" * len(method_name))
        print(classification_report(y_test, result['predictions'], target_names=data.target_names))
    
    return results
```

### 7.4 Applications and Best Practices

### 7.4.1 Data Visualization and Exploration

#### **Multi-Dataset Visualization Pipeline**
```python
def create_visualization_pipeline():
    """Create comprehensive visualization pipeline for different data types"""
    
    from sklearn.datasets import (load_breast_cancer, load_wine, load_digits, 
                                 fetch_olivetti_faces)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    # Load diverse datasets
    datasets = {
        'Breast Cancer': load_breast_cancer(),
        'Wine': load_wine(),
        'Digits': load_digits(),
        'Faces': fetch_olivetti_faces()
    }
    
    # Create visualization pipeline
    def visualize_dataset(name, dataset, max_samples=1000):
        """Visualize single dataset with multiple methods"""
        
        X, y = dataset.data, dataset.target
        
        # Sample data if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        
        print(f"\nProcessing {name} dataset:")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {len(np.unique(y))}")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply different reduction methods
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # LDA (limit components to n_classes - 1)
        n_components_lda = min(len(np.unique(y)) - 1, 2)
        if n_components_lda > 0:
            lda = LDA(n_components=n_components_lda)
            X_lda = lda.fit_transform(X_scaled, y)
        else:
            X_lda = None
        
        # t-SNE (with PCA preprocessing if high-dimensional)
        if X_scaled.shape[1] > 50:
            pca_pre = PCA(n_components=50, random_state=42)
            X_for_tsne = pca_pre.fit_transform(X_scaled)
        else:
            X_for_tsne = X_scaled
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(X)//4), 
                   random_state=42, verbose=0)
        X_tsne = tsne.fit_transform(X_for_tsne)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original data (first 2 features)
        if X.shape[1] >= 2:
            scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.7)
            axes[0].set_title('Original (First 2 Features)')
            feature_names = getattr(dataset, 'feature_names', ['Feature 0', 'Feature 1'])
            axes[0].set_xlabel(feature_names[0][:20])
            axes[0].set_ylabel(feature_names[1][:20])
        else:
            axes[0].text(0.5, 0.5, 'Not applicable', ha='center', va='center',
                        transform=axes[0].transAxes)
            axes[0].set_title('Original Data')
        
        # PCA
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[1].set_title(f'PCA ({pca.explained_variance_ratio_.sum():.1%} var)')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        
        # LDA
        if X_lda is not None:
            if X_lda.shape[1] >= 2:
                scatter = axes[2].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.7)
                axes[2].set_xlabel('LD1')
                axes[2].set_ylabel('LD2')
            else:
                # 1D LDA
                scatter = axes[2].scatter(X_lda[:, 0], np.random.normal(0, 0.1, len(X_lda)), 
                                        c=y, cmap='tab10', alpha=0.7)
                axes[2].set_xlabel('LD1')
                axes[2].set_ylabel('Random Jitter')
            axes[2].set_title('LDA')
        else:
            axes[2].text(0.5, 0.5, 'Single class', ha='center', va='center',
                        transform=axes[2].transAxes)
            axes[2].set_title('LDA (Not applicable)')
        
        # t-SNE
        scatter = axes[3].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[3].set_title('t-SNE')
        axes[3].set_xlabel('t-SNE 1')
        axes[3].set_ylabel('t-SNE 2')
        
        # Add colorbar to last plot
        plt.colorbar(scatter, ax=axes[3])
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{name} Dataset Visualization Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return {
            'pca_variance': pca.explained_variance_ratio_.sum(),
            'n_components_lda': n_components_lda,
            'tsne_kl': tsne.kl_divergence_
        }
    
    # Process all datasets
    results = {}
    for name, dataset in datasets.items():
        results[name] = visualize_dataset(name, dataset)
    
    return results

visualization_results = create_visualization_pipeline()
```

#### **Interactive Dimensionality Reduction Explorer**
```python
def interactive_dim_reduction_explorer(X, y, feature_names=None):
    """Interactive explorer for different dimensionality reduction techniques"""
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    
    print("Interactive Dimensionality Reduction Explorer:")
    print("="*45)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Methods configuration
    methods_config = {
        'PCA': {
            'method': PCA,
            'params': {'n_components': 2, 'random_state': 42},
            'requires_y': False
        },
        'LDA': {
            'method': LDA,
            'params': {'n_components': min(len(np.unique(y))-1, 2)},
            'requires_y': True
        },
        't-SNE (Perp=5)': {
            'method': TSNE,
            'params': {'n_components': 2, 'perplexity': 5, 'random_state': 42, 'verbose': 0},
            'requires_y': False
        },
        't-SNE (Perp=30)': {
            'method': TSNE,
            'params': {'n_components': 2, 'perplexity': 30, 'random_state': 42, 'verbose': 0},
            'requires_y': False
        },
        't-SNE (Perp=50)': {
            'method': TSNE,
            'params': {'n_components': 2, 'perplexity': 50, 'random_state': 42, 'verbose': 0},
            'requires_y': False
        }
    }
    
    # Apply all methods
    results = {}
    valid_methods = {}
    
    for name, config in methods_config.items():
        try:
            print(f"Applying {name}...")
            
            method_class = config['method']
            params = config['params']
            
            # Skip if invalid configuration
            if name.startswith('LDA') and params['n_components'] <= 0:
                print(f"  Skipped (insufficient classes)")
                continue
            
            if name.startswith('t-SNE') and params['perplexity'] >= len(X) / 3:
                print(f"  Skipped (perplexity too large)")
                continue
            
            method = method_class(**params)
            
            if config['requires_y']:
                X_transformed = method.fit_transform(X_scaled, y)
            else:
                X_transformed = method.fit_transform(X_scaled)
            
            results[name] = X_transformed
            valid_methods[name] = method
            print(f"  ✓ Success")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Create comprehensive visualization
    n_methods = len(results)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for i, (name, X_transformed) in enumerate(results.items()):
        ax = axes_flat[i]
        
        # Create scatter plot
        scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                           c=y, cmap='tab10', alpha=0.7, s=30)
        
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        
        # Set axis labels based on method
        if name.startswith('PCA'):
            method = valid_methods[name]
            ax.set_xlabel(f'PC1 ({method.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({method.explained_variance_ratio_[1]:.1%})')
        elif name.startswith('LDA'):
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
        else:  # t-SNE
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
    
    # Hide unused subplots
    for i in range(n_methods, len(axes_flat)):
        axes_flat[i].axis('off')
    
    # Add colorbar
    if n_methods > 0:
        plt.colorbar(scatter, ax=axes_flat[n_methods-1])
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    print(f"\nMethod Analysis:")
    print("-" * 40)
    
    for name, method in valid_methods.items():
        if hasattr(method, 'explained_variance_ratio_'):
            variance_explained = method.explained_variance_ratio_.sum()
            print(f"{name:20s}: {variance_explained:.1%} variance explained")
        elif hasattr(method, 'kl_divergence_'):
            print(f"{name:20s}: KL divergence = {method.kl_divergence_:.2f}")
        else:
            print(f"{name:20s}: Method applied successfully")
    
    return results, valid_methods
```

### 7.4.2 Preprocessing for Machine Learning Pipelines

#### **Integrated ML Pipeline with Dimensionality Reduction**
```python
def ml_pipeline_with_dim_reduction():
    """Complete ML pipeline integrating dimensionality reduction"""
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Generate high-dimensional dataset
    X, y = make_classification(n_samples=1000, n_features=100, 
                             n_informative=20, n_redundant=30,
                             n_classes=3, random_state=42)
    
    print("ML Pipeline with Dimensionality Reduction:")
    print("="*45)
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Define different pipeline configurations
    pipelines = {
        'Baseline (No Reduction)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        
        'PCA + Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        
        'LDA + SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LDA()),
            ('classifier', SVC(random_state=42))
        ]),
        
        'PCA + Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=42)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    # Parameter grids for hyperparameter tuning
    param_grids = {
        'Baseline (No Reduction)': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, 20, None]
        },
        
        'PCA + Random Forest': {
            'pca__n_components': [10, 20, 50],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [10, 20]
        },
        
        'LDA + SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        },
        
        'PCA + Logistic Regression': {
            'pca__n_components': [10, 20, 50],
            'classifier__C': [0.1, 1, 10]
        }
    }
    
    # Train and evaluate pipelines
    results = {}
    
    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        
        # Grid search with cross-validation
        param_grid = param_grids[name]
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, 
                                  scoring='accuracy', n_jobs=-1, verbose=0)
        
        # Fit and predict
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        
        # Store results
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_score': grid_search.score(X_test, y_test),
            'predictions': y_pred,
            'model': grid_search.best_estimator_
        }
        
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        print(f"  Test score: {grid_search.score(X_test, y_test):.4f}")
        print(f"  Best params: {grid_search.best_params_}")
    
    # Comprehensive comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance comparison
    pipeline_names = list(results.keys())
    cv_scores = [results[name]['best_score'] for name in pipeline_names]
    test_scores = [results[name]['test_score'] for name in pipeline_names]
    errors = [results[name]['std'] for name in pipeline_names]
    
    x = np.arange(len(pipeline_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
    bars2 = axes[0, 0].bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
    
    axes[0, 0].set_xlabel('Pipeline')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([name.split()[0] for name in pipeline_names], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Feature importance analysis (for PCA pipelines)
    pca_pipeline = results['PCA + Random Forest']['model']
    if hasattr(pca_pipeline.named_steps['pca'], 'components_'):
        # Get PCA components and feature importance
        pca_components = pca_pipeline.named_steps['pca'].components_
        rf_importance = pca_pipeline.named_steps['classifier'].feature_importances_
        
        # Calculate original feature importance through PCA
        original_importance = np.abs(pca_components.T @ rf_importance)
        
        # Show top 15 features
        top_features = np.argsort(original_importance)[-15:]
        
        axes[0, 1].barh(range(15), original_importance[top_features])
        axes[0, 1].set_yticks(range(15))
        axes[0, 1].set_yticklabels([f'Feature {i}' for i in top_features])
        axes[0, 1].set_xlabel('Importance Score')
        axes[0, 1].set_title('PCA Feature Importance')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy vs Dimensionality trade-off
    axes[1, 0].scatter(n_features, accuracies, s=100, alpha=0.7)
    for i, method in enumerate(method_names):
        axes[1, 0].annotate(method.split()[0], (n_features[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('Number of Features')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Dimensionality Trade-off')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Method recommendations
    axes[1, 1].axis('off')
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
    most_efficient = min(results.keys(), key=lambda k: results[k]['n_features'] 
                        if results[k]['accuracy'] > 0.9 else float('inf'))
    
    recommendations = f"""
    RECOMMENDATIONS:
    
    Best Accuracy: {best_method}
    • Accuracy: {results[best_method]['accuracy']:.4f}
    • Features: {results[best_method]['n_features']}
    
    Most Efficient: {most_efficient}
    • Accuracy: {results[most_efficient]['accuracy']:.4f}
    • Features: {results[most_efficient]['n_features']}
    • Reduction: {(1-results[most_efficient]['n_features']/X.shape[1])*100:.1f}%
    
    INSIGHTS:
    • {len(high_corr_pairs)} highly correlated features
    • PCA effective for noise reduction
    • LDA good for classification tasks
    • Feature selection preserves interpretability
    """
    
    axes[1, 1].text(0.05, 0.95, recommendations, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification reports
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORTS:")
    print(f"{'='*60}")
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print("-" * len(method_name))
        print(classification_report(y_test, result['predictions'], target_names=data.target_names))
    
    return results
```

### 7.4.3 Performance Considerations and Best Practices

#### **Computational Efficiency Analysis**
```python
def analyze_computational_efficiency():
    """Analyze computational efficiency of different dimensionality reduction methods"""
    
    import time
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import SelectKBest, f_classif
    
    print("Computational Efficiency Analysis:")
    print("="*35)
    
    # Test different dataset sizes
    dataset_sizes = [
        (500, 50),
        (1000, 100),
        (2000, 200),
        (5000, 500)
    ]
    
    methods = {
        'PCA': lambda X, y: PCA(n_components=min(20, X.shape[1]//2)).fit_transform(X),
        'Incremental PCA': lambda X, y: IncrementalPCA(n_components=min(20, X.shape[1]//2), 
                                                       batch_size=100).fit_transform(X),
        'Truncated SVD': lambda X, y: TruncatedSVD(n_components=min(20, X.shape[1]//2)).fit_transform(X),
        'LDA': lambda X, y: LDA(n_components=min(len(np.unique(y))-1, 10)).fit_transform(X, y),
        'SelectKBest': lambda X, y: SelectKBest(f_classif, k=min(20, X.shape[1]//2)).fit_transform(X, y),
        't-SNE (small)': lambda X, y: TSNE(n_components=2, perplexity=min(30, len(X)//4), 
                                          verbose=0).fit_transform(X) if len(X) <= 1000 else None
    }
    
    results = []
    
    for n_samples, n_features in dataset_sizes:
        print(f"\nDataset size: {n_samples} × {n_features}")
        
        # Generate data
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=min(20, n_features//2),
                                 n_classes=5, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for method_name, method_func in methods.items():
            try:
                print(f"  Testing {method_name}...", end=' ')
                
                # Measure time
                start_time = time.time()
                result = method_func(X_scaled, y)
                end_time = time.time()
                
                if result is not None:
                    execution_time = end_time - start_time
                    memory_usage = result.nbytes / (1024**2)  # MB
                    
                    results.append({
                        'method': method_name,
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'time': execution_time,
                        'memory_mb': memory_usage,
                        'output_shape': result.shape
                    })
                    
                    print(f"{execution_time:.3f}s, {memory_usage:.2f}MB")
                else:
                    print("Skipped (too large)")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    # Analyze results
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time vs dataset size
        for method in df_results['method'].unique():
            method_data = df_results[df_results['method'] == method]
            dataset_complexity = method_data['n_samples'] * method_data['n_features']
            axes[0, 0].loglog(dataset_complexity, method_data['time'], 
                             'o-', label=method, alpha=0.7)
        
        axes[0, 0].set_xlabel('Dataset Complexity (samples × features)')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].set_title('Scalability Analysis')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage comparison
        for method in df_results['method'].unique():
            method_data = df_results[df_results['method'] == method]
            axes[0, 1].plot(method_data['n_samples'], method_data['memory_mb'], 
                           'o-', label=method, alpha=0.7)
        
        axes[0, 1].set_xlabel('Number of Samples')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Consumption')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Method comparison for largest dataset
        largest_dataset = df_results[df_results['n_samples'] == max(df_results['n_samples'])]
        if not largest_dataset.empty:
            methods_subset = largest_dataset['method'].tolist()
            times_subset = largest_dataset['time'].tolist()
            
            bars = axes[1, 0].bar(methods_subset, times_subset, alpha=0.7)
            axes[1, 0].set_ylabel('Execution Time (seconds)')
            axes[1, 0].set_title(f'Performance on Largest Dataset ({max(df_results["n_samples"])}×{max(df_results["n_features"])})')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, time_val in zip(bars, times_subset):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
        
        # Efficiency ratio (output features / input features) vs time
        df_results['efficiency_ratio'] = df_results.apply(
            lambda row: row['output_shape'][1] / row['n_features'], axis=1
        )
        
        scatter = axes[1, 1].scatter(df_results['efficiency_ratio'], df_results['time'],
                                    c=df_results['n_samples'], cmap='viridis', 
                                    s=60, alpha=0.7)
        
        axes[1, 1].set_xlabel('Compression Ratio (output/input features)')
        axes[1, 1].set_ylabel('Execution Time (seconds)')
        axes[1, 1].set_title('Compression vs Speed Trade-off')
        plt.colorbar(scatter, ax=axes[1, 1], label='Number of Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Performance recommendations
        print(f"\n{'='*50}")
        print("PERFORMANCE RECOMMENDATIONS:")
        print(f"{'='*50}")
        
        # Find fastest method for each dataset size
        for size in df_results[['n_samples', 'n_features']].drop_duplicates().values:
            n_samples, n_features = size
            size_data = df_results[(df_results['n_samples'] == n_samples) & 
                                 (df_results['n_features'] == n_features)]
            if not size_data.empty:
                fastest_method = size_data.loc[size_data['time'].idxmin(), 'method']
                fastest_time = size_data['time'].min()
                print(f"For {n_samples}×{n_features}: {fastest_method} ({fastest_time:.3f}s)")
        
        # General recommendations
        print(f"\nGeneral Guidelines:")
        avg_times = df_results.groupby('method')['time'].mean().sort_values()
        print(f"• Fastest overall: {avg_times.index[0]}")
        print(f"• Most scalable: Incremental PCA (for very large datasets)")
        print(f"• Best for visualization: t-SNE (but expensive)")
        print(f"• Good balance: PCA or Truncated SVD")
        
    return df_results
```

### 7.4.3 Performance Considerations and Best Practices

#### **Computational Efficiency Analysis**
```python
def analyze_computational_efficiency():
    """Analyze computational efficiency of different dimensionality reduction methods"""
    
    import time
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import SelectKBest, f_classif
    
    print("Computational Efficiency Analysis:")
    print("="*35)
    
    # Test different dataset sizes
    dataset_sizes = [
        (500, 50),
        (1000, 100),
        (2000, 200),
        (5000, 500)
    ]
    
    methods = {
        'PCA': lambda X, y: PCA(n_components=min(20, X.shape[1]//2)).fit_transform(X),
        'Incremental PCA': lambda X, y: IncrementalPCA(n_components=min(20, X.shape[1]//2), 
                                                       batch_size=100).fit_transform(X),
        'Truncated SVD': lambda X, y: TruncatedSVD(n_components=min(20, X.shape[1]//2)).fit_transform(X),
        'LDA': lambda X, y: LDA(n_components=min(len(np.unique(y))-1, 10)).fit_transform(X, y),
        'SelectKBest': lambda X, y: SelectKBest(f_classif, k=min(20, X.shape[1]//2)).fit_transform(X, y),
        't-SNE (small)': lambda X, y: TSNE(n_components=2, perplexity=min(30, len(X)//4), 
                                          verbose=0).fit_transform(X) if len(X) <= 1000 else None
    }
    
    results = []
    
    for n_samples, n_features in dataset_sizes:
        print(f"\nDataset size: {n_samples} × {n_features}")
        
        # Generate data
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                 n_informative=min(20, n_features//2),
                                 n_classes=5, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for method_name, method_func in methods.items():
            try:
                print(f"  Testing {method_name}...", end=' ')
                
                # Measure time
                start_time = time.time()
                result = method_func(X_scaled, y)
                end_time = time.time()
                
                if result is not None:
                    execution_time = end_time - start_time
                    memory_usage = result.nbytes / (1024**2)  # MB
                    
                    results.append({
                        'method': method_name,
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'time': execution_time,
                        'memory_mb': memory_usage,
                        'output_shape': result.shape
                    })
                    
                    print(f"{execution_time:.3f}s, {memory_usage:.2f}MB")
                else:
                    print("Skipped (too large)")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    # Analyze results
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time vs dataset size
        for method in df_results['method'].unique():
            method_data = df_results[df_results['method'] == method]
            dataset_complexity = method_data['n_samples'] * method_data['n_features']
            axes[0, 0].loglog(dataset_complexity, method_data['time'], 
                             'o-', label=method, alpha=0.7)
        
        axes[0, 0].set_xlabel('Dataset Complexity (samples × features)')
        axes[0, 0].set_ylabel('Execution Time (seconds)')
        axes[0, 0].set_title('Scalability Analysis')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage comparison
        for method in df_results['method'].unique():
            method_data = df_results[df_results['method'] == method]
            axes[0, 1].plot(method_data['n_samples'], method_data['memory_mb'], 
                           'o-', label=method, alpha=0.7)
        
        axes[0, 1].set_xlabel('Number of Samples')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Consumption')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Method comparison for largest dataset
        largest_dataset = df_results[df_results['n_samples'] == max(df_results['n_samples'])]
        if not largest_dataset.empty:
            methods_subset = largest_dataset['method'].tolist()
            times_subset = largest_dataset['time'].tolist()
            
            bars = axes[1, 0].bar(methods_subset, times_subset, alpha=0.7)
            axes[1, 0].set_ylabel('Execution Time (seconds)')
            axes[1, 0].set_title(f'Performance on Largest Dataset ({max(df_results["n_samples"])}×{max(df_results["n_features"])})')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, time_val in zip(bars, times_subset):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{time_val:.3f}s', ha='center', va='bottom', fontsize=8)
        
        # Efficiency ratio (output features / input features) vs time
        df_results['efficiency_ratio'] = df_results.apply(
            lambda row: row['output_shape'][1] / row['n_features'], axis=1
        )
        
        scatter = axes[1, 1].scatter(df_results['efficiency_ratio'], df_results['time'],
                                    c=df_results['n_samples'], cmap='viridis', 
                                    s=60, alpha=0.7)
        
        axes[1, 1].set_xlabel('Compression Ratio (output/input features)')
        axes[1, 1].set_ylabel('Execution Time (seconds)')
        axes[1, 1].set_title('Compression vs Speed Trade-off')
        plt.colorbar(scatter, ax=axes[1, 1], label='Number of Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Performance recommendations
        print(f"\n{'='*50}")
        print("PERFORMANCE RECOMMENDATIONS:")
        print(f"{'='*50}")
        
        # Find fastest method for each dataset size
        for size in df_results[['n_samples', 'n_features']].drop_duplicates().values:
            n_samples, n_features = size
            size_data = df_results[(df_results['n_samples'] == n_samples) & 
                                 (df_results['n_features'] == n_features)]
            if not size_data.empty:
                fastest_method = size_data.loc[size_data['time'].idxmin(), 'method']
                fastest_time = size_data['time'].min()
                print(f"For {n_samples}×{n_features}: {fastest_method} ({fastest_time:.3f}s)")
        
        # General recommendations
        print(f"\nGeneral Guidelines:")
        avg_times = df_results.groupby('method')['time'].mean().sort_values()
        print(f"• Fastest overall: {avg_times.index[0]}")
        print(f"• Most scalable: Incremental PCA (for very large datasets)")
        print(f"• Best for visualization: t-SNE (but expensive)")
        print(f"• Good balance: PCA or Truncated SVD")
        
    return df_results
```

### 7.5 Practical Labs

#### **Lab 1: Image Compression with PCA**
```python
def image_compression_lab():
    """Hands-on lab: Image compression using PCA"""
    
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    print("Lab 1: Image Compression with PCA")
    print("="*35)
    
    # Load face dataset
    faces = fetch_olivetti_faces()
    X_faces = faces.data  # 400 faces, each 64×64 pixels (4096 features)
    
    print(f"Dataset shape: {X_faces.shape}")
    print(f"Image resolution: 64×64 pixels")
    
    # Select a single face for compression analysis
    face_idx = 0
    original_face = X_faces[face_idx].reshape(64, 64)
    
    # Test different numbers of components
    n_components_list = [10, 25, 50, 100, 200, 500]
    
    # Apply PCA to entire dataset
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_faces)
    
    # Analyze compression results
    compression_results = []
    
    fig, axes = plt.subplots(2, len(n_components_list), figsize=(20, 8))
    
    for i, n_comp in enumerate(n_components_list):
        # Reconstruct using n components
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_faces)
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Reconstruct the selected face
        reconstructed_face = X_reconstructed[face_idx].reshape(64, 64)
        
        # Calculate metrics
        mse = np.mean((original_face - reconstructed_face) ** 2)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        compression_ratio = 4096 / (n_comp + n_comp * 4096 / 400)  # Approximate
        
        compression_results.append({
            'n_components': n_comp,
            'mse': mse,
            'variance_explained': variance_explained,
            'compression_ratio': compression_ratio
        })
        
        # Display original and reconstructed
        axes[0, i].imshow(original_face, cmap='gray')
        axes[0, i].set_title(f'Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed_face, cmap='gray')
        axes[1, i].set_title(f'{n_comp} comp\\nMSE: {mse:.4f}\\nVar: {variance_explained:.1%}')
        axes[1, i].axis('off')
        
        print(f"{n_comp:3d} components: MSE={mse:.4f}, Variance={variance_explained:.1%}, Compression={compression_ratio:.1f}x")
    
    plt.tight_layout()
    plt.show()
    
    # Analysis plots
    df_compression = pd.DataFrame(compression_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE vs Components
    axes[0].plot(df_compression['n_components'], df_compression['mse'], 'ro-')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Reconstruction Error vs Components')
    axes[0].grid(True, alpha=0.3)
    
    # Variance Explained
    axes[1].plot(df_compression['n_components'], df_compression['variance_explained'], 'bo-')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Variance Explained')
    axes[1].set_title('Information Retention vs Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Compression Trade-off
    axes[2].scatter(df_compression['compression_ratio'], df_compression['mse'], 
                   c=df_compression['n_components'], cmap='viridis', s=100)
    axes[2].set_xlabel('Compression Ratio')
    axes[2].set_ylabel('Reconstruction Error (MSE)')
    axes[2].set_title('Compression vs Quality Trade-off')
    
    # Add component labels
    for _, row in df_compression.iterrows():
        axes[2].annotate(f"{row['n_components']}", 
                        (row['compression_ratio'], row['mse']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Components')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_compression

compression_lab_results = image_compression_lab()
```

#### **Lab 2: Dimensionality Reduction Pipeline**
```python
def dimensionality_reduction_pipeline_lab():
    """Lab 2: Building a complete dimensionality reduction pipeline"""
    
    from sklearn.datasets import make_classification, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    
    print("Lab 2: Complete Dimensionality Reduction Pipeline")
    print("="*50)
    
    # Load real dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset: {data.DESCR.split(':', 1)[0]}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))} ({np.bincount(y)})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        stratify=y, random_state=42)
    
    # Step 1: Exploratory Data Analysis
    print(f"\nStep 1: Exploratory Data Analysis")
    print("-" * 35)
    
    # Analyze feature correlations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    correlation_matrix = np.corrcoef(X_scaled.T)
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if abs(correlation_matrix[i, j]) > 0.8:
                high_corr_pairs.append((i, j, correlation_matrix[i, j]))
    
    print(f"Highly correlated feature pairs (>0.8): {len(high_corr_pairs)}")
    
    # Step 2: Compare Dimensionality Reduction Methods
    print(f"\nStep 2: Method Comparison")
    print("-" * 25)
    
    methods = {
        'No Reduction': None,
        'PCA (95% var)': 'pca_95',
        'PCA (10 comp)': 'pca_10',
        'LDA': 'lda',
        'SelectKBest (10)': 'select_10'
    }
    
    results = {}
    
    for method_name, method_type in methods.items():
        print(f"\nTesting {method_name}...")
        
        if method_type is None:
            # No reduction
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            X_train_processed = X_train
            X_test_processed = X_test
            
        elif method_type == 'pca_95':
            # PCA with 95% variance
            pca = PCA(random_state=42)
            X_scaled_train = StandardScaler().fit_transform(X_train)
            pca.fit(X_scaled_train)
            
            # Find components for 95% variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_comp_95 = np.argmax(cumvar >= 0.95) + 1
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_comp_95, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'pca_10':
            # PCA with 10 components
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=10, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'lda':
            # LDA
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lda', LDA()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'select_10':
            # Feature selection
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=10)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get effective dimensionality
        if method_type is None:
            n_features_used = X.shape[1]
        elif 'pca' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['pca'].n_components_
        elif 'lda' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['lda'].scalings_.shape[1]
        elif 'selector' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['selector'].k
        else:
            n_features_used = X.shape[1]
        
        results[method_name] = {
            'accuracy': accuracy,
            'n_features': n_features_used,
            'pipeline': pipeline,
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Features used: {n_features_used}")
        print(f"  Reduction: {(1 - n_features_used/X.shape[1])*100:.1f}%")
    
    # Step 3: Detailed Analysis
    print(f"\nStep 3: Detailed Analysis")
    print("-" * 23)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Performance comparison
    method_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in method_names]
    n_features = [results[m]['n_features'] for m in method_names]
    
    bars = axes[0, 0].bar(range(len(method_names)), accuracies, alpha=0.7)
    axes[0, 0].set_xticks(range(len(method_names)))
    axes[0, 0].set_xticklabels([name.split()[0] for name in method_names], rotation=45)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Method Performance Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Feature count comparison
    bars = axes[0, 1].bar(range(len(method_names)), n_features, alpha=0.7)
    axes[0, 1].set_xticks(range(len(method_names)))
    axes[0, 1].set_xticklabels([name.split()[0] for name in method_names], rotation=45)
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Dimensionality Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA analysis (if available)
    pca_pipeline = results['PCA (95% var)']['pipeline']
    if 'pca' in pca_pipeline.named_steps:
        pca = pca_pipeline.named_steps['pca']
        
        # Explained variance
        axes[0, 2].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                       np.cumsum(pca.explained_variance_ratio_), 'bo-')
        axes[0, 2].axhline(y=0.95, color='red', linestyle='--', label='95%')
        axes[0, 2].set_xlabel('Component')
        axes[0, 2].set_ylabel('Cumulative Variance Explained')
        axes[0, 2].set_title('PCA Variance Analysis')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Feature importance in first 2 PCs
        pc1_importance = np.abs(pca.components_[0])
        pc2_importance = np.abs(pca.components_[1])
        
        top_features_pc1 = np.argsort(pc1_importance)[-10:]
        top_features_pc2 = np.argsort(pc2_importance)[-10:]
        
        axes[1, 0].barh(range(10), pc1_importance[top_features_pc1])
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels([feature_names[i][:15] for i in top_features_pc1], fontsize=8)
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_title('PC1 - Top Feature Contributions')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].barh(range(10), pc2_importance[top_features_pc2])
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels([feature_names[i][:15] for i in top_features_pc2], fontsize=8)
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_title('PC2 - Top Feature Contributions')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy vs Dimensionality trade-off
    axes[1, 2].scatter(n_features, accuracies, s=100, alpha=0.7)
    for i, method in enumerate(method_names):
        axes[1, 2].annotate(method.split()[0], (n_features[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 2].set_xlabel('Number of Features')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy vs Dimensionality Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Method recommendations
    axes[1, 2].axis('off')
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
    most_efficient = min(results.keys(), key=lambda k: results[k]['n_features'] 
                        if results[k]['accuracy'] > 0.9 else float('inf'))
    
    recommendations = f"""
    RECOMMENDATIONS:
    
    Best Accuracy: {best_method}
    • Accuracy: {results[best_method]['accuracy']:.4f}
    • Features: {results[best_method]['n_features']}
    
    Most Efficient: {most_efficient}
    • Accuracy: {results[most_efficient]['accuracy']:.4f}
    • Features: {results[most_efficient]['n_features']}
    • Reduction: {(1-results[most_efficient]['n_features']/X.shape[1])*100:.1f}%
    
    INSIGHTS:
    • {len(high_corr_pairs)} highly correlated features
    • PCA effective for noise reduction
    • LDA good for classification tasks
    • Feature selection preserves interpretability
    """
    
    axes[1, 2].text(0.05, 0.95, recommendations, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification reports
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORTS:")
    print(f"{'='*60}")
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print("-" * len(method_name))
        print(classification_report(y_test, result['predictions'], target_names=data.target_names))
    
    return results
```

### 7.5 Practical Labs

#### **Lab 1: Image Compression with PCA**
```python
def image_compression_lab():
    """Hands-on lab: Image compression using PCA"""
    
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    print("Lab 1: Image Compression with PCA")
    print("="*35)
    
    # Load face dataset
    faces = fetch_olivetti_faces()
    X_faces = faces.data  # 400 faces, each 64×64 pixels (4096 features)
    
    print(f"Dataset shape: {X_faces.shape}")
    print(f"Image resolution: 64×64 pixels")
    
    # Select a single face for compression analysis
    face_idx = 0
    original_face = X_faces[face_idx].reshape(64, 64)
    
    # Test different numbers of components
    n_components_list = [10, 25, 50, 100, 200, 500]
    
    # Apply PCA to entire dataset
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_faces)
    
    # Analyze compression results
    compression_results = []
    
    fig, axes = plt.subplots(2, len(n_components_list), figsize=(20, 8))
    
    for i, n_comp in enumerate(n_components_list):
        # Reconstruct using n components
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_faces)
        X_reconstructed = pca.inverse_transform(X_pca)
        
        # Reconstruct the selected face
        reconstructed_face = X_reconstructed[face_idx].reshape(64, 64)
        
        # Calculate metrics
        mse = np.mean((original_face - reconstructed_face) ** 2)
        variance_explained = np.sum(pca.explained_variance_ratio_)
        compression_ratio = 4096 / (n_comp + n_comp * 4096 / 400)  # Approximate
        
        compression_results.append({
            'n_components': n_comp,
            'mse': mse,
            'variance_explained': variance_explained,
            'compression_ratio': compression_ratio
        })
        
        # Display original and reconstructed
        axes[0, i].imshow(original_face, cmap='gray')
        axes[0, i].set_title(f'Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed_face, cmap='gray')
        axes[1, i].set_title(f'{n_comp} comp\\nMSE: {mse:.4f}\\nVar: {variance_explained:.1%}')
        axes[1, i].axis('off')
        
        print(f"{n_comp:3d} components: MSE={mse:.4f}, Variance={variance_explained:.1%}, Compression={compression_ratio:.1f}x")
    
    plt.tight_layout()
    plt.show()
    
    # Analysis plots
    df_compression = pd.DataFrame(compression_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE vs Components
    axes[0].plot(df_compression['n_components'], df_compression['mse'], 'ro-')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Reconstruction Error vs Components')
    axes[0].grid(True, alpha=0.3)
    
    # Variance Explained
    axes[1].plot(df_compression['n_components'], df_compression['variance_explained'], 'bo-')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Variance Explained')
    axes[1].set_title('Information Retention vs Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Compression Trade-off
    axes[2].scatter(df_compression['compression_ratio'], df_compression['mse'], 
                   c=df_compression['n_components'], cmap='viridis', s=100)
    axes[2].set_xlabel('Compression Ratio')
    axes[2].set_ylabel('Reconstruction Error (MSE)')
    axes[2].set_title('Compression vs Quality Trade-off')
    
    # Add component labels
    for _, row in df_compression.iterrows():
        axes[2].annotate(f"{row['n_components']}", 
                        (row['compression_ratio'], row['mse']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Components')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_compression

compression_lab_results = image_compression_lab()
```

#### **Lab 2: Dimensionality Reduction Pipeline**
```python
def dimensionality_reduction_pipeline_lab():
    """Lab 2: Building a complete dimensionality reduction pipeline"""
    
    from sklearn.datasets import make_classification, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    
    print("Lab 2: Complete Dimensionality Reduction Pipeline")
    print("="*50)
    
    # Load real dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset: {data.DESCR.split(':', 1)[0]}")
    print(f"Shape: {X.shape}")
    print(f"Classes: {len(np.unique(y))} ({np.bincount(y)})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        stratify=y, random_state=42)
    
    # Step 1: Exploratory Data Analysis
    print(f"\nStep 1: Exploratory Data Analysis")
    print("-" * 35)
    
    # Analyze feature correlations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    correlation_matrix = np.corrcoef(X_scaled.T)
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if abs(correlation_matrix[i, j]) > 0.8:
                high_corr_pairs.append((i, j, correlation_matrix[i, j]))
    
    print(f"Highly correlated feature pairs (>0.8): {len(high_corr_pairs)}")
    
    # Step 2: Compare Dimensionality Reduction Methods
    print(f"\nStep 2: Method Comparison")
    print("-" * 25)
    
    methods = {
        'No Reduction': None,
        'PCA (95% var)': 'pca_95',
        'PCA (10 comp)': 'pca_10',
        'LDA': 'lda',
        'SelectKBest (10)': 'select_10'
    }
    
    results = {}
    
    for method_name, method_type in methods.items():
        print(f"\nTesting {method_name}...")
        
        if method_type is None:
            # No reduction
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            X_train_processed = X_train
            X_test_processed = X_test
            
        elif method_type == 'pca_95':
            # PCA with 95% variance
            pca = PCA(random_state=42)
            X_scaled_train = StandardScaler().fit_transform(X_train)
            pca.fit(X_scaled_train)
            
            # Find components for 95% variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_comp_95 = np.argmax(cumvar >= 0.95) + 1
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_comp_95, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'pca_10':
            # PCA with 10 components
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=10, random_state=42)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'lda':
            # LDA
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lda', LDA()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
        elif method_type == 'select_10':
            # Feature selection
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=10)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get effective dimensionality
        if method_type is None:
            n_features_used = X.shape[1]
        elif 'pca' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['pca'].n_components_
        elif 'lda' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['lda'].scalings_.shape[1]
        elif 'selector' in pipeline.named_steps:
            n_features_used = pipeline.named_steps['selector'].k
        else:
            n_features_used = X.shape[1]
        
        results[method_name] = {
            'accuracy': accuracy,
            'n_features': n_features_used,
            'pipeline': pipeline,
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Features used: {n_features_used}")
        print(f"  Reduction: {(1 - n_features_used/X.shape[1])*100:.1f}%")
    
    # Step 3: Detailed Analysis
    print(f"\nStep 3: Detailed Analysis")
    print("-" * 23)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Performance comparison
    method_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in method_names]
    n_features = [results[m]['n_features'] for m in method_names]
    
    bars = axes[0, 0].bar(range(len(method_names)), accuracies, alpha=0.7)
    axes[0, 0].set_xticks(range(len(method_names)))
    axes[0, 0].set_xticklabels([name.split()[0] for name in method_names], rotation=45)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Method Performance Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Feature count comparison
    bars = axes[0, 1].bar(range(len(method_names)), n_features, alpha=0.7)
    axes[0, 1].set_xticks(range(len(method_names)))
    axes[0, 1].set_xticklabels([name.split()[0] for name in method_names], rotation=45)
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Dimensionality Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA analysis (if available)
    pca_pipeline = results['PCA (95% var)']['pipeline']
    if 'pca' in pca_pipeline.named_steps:
        pca = pca_pipeline.named_steps['pca']
        
        # Explained variance
        axes[0, 2].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                       np.cumsum(pca.explained_variance_ratio_), 'bo-')
        axes[0, 2].axhline(y=0.95, color='red', linestyle='--', label='95%')
        axes[0, 2].set_xlabel('Component')
        axes[0, 2].set_ylabel('Cumulative Variance Explained')
        axes[0, 2].set_title('PCA Variance Analysis')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Feature importance in first 2 PCs
        pc1_importance = np.abs(pca.components_[0])
        pc2_importance = np.abs(pca.components_[1])
        
        top_features_pc1 = np.argsort(pc1_importance)[-10:]
        top_features_pc2 = np.argsort(pc2_importance)[-10:]
        
        axes[1, 0].barh(range(10), pc1_importance[top_features_pc1])
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels([feature_names[i][:15] for i in top_features_pc1], fontsize=8)
        axes[1, 0].set_xlabel('Importance Score')
        axes[1, 0].set_title('PC1 - Top Feature Contributions')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].barh(range(10), pc2_importance[top_features_pc2])
        axes[1, 1].set_yticks(range(10))
        axes[1, 1].set_yticklabels([feature_names[i][:15] for i in top_features_pc2], fontsize=8)
        axes[1, 1].set_xlabel('Importance Score')
        axes[1, 1].set_title('PC2 - Top Feature Contributions')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy vs Dimensionality trade-off
    axes[1, 2].scatter(n_features, accuracies, s=100, alpha=0.7)
    for i, method in enumerate(method_names):
        axes[1, 2].annotate(method.split()[0], (n_features[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 2].set_xlabel('Number of Features')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Accuracy vs Dimensionality Trade-off')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Method recommendations
    axes[1, 2].axis('off')
    
    # Find best method
    best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
    most_efficient = min(results.keys(), key=lambda k: results[k]['n_features'] 
                        if results[k]['accuracy'] > 0.9 else float('inf'))
    
    recommendations = f"""
    RECOMMENDATIONS:
    
    Best Accuracy: {best_method}
    • Accuracy: {results[best_method]['accuracy']:.4f}
    • Features: {results[best_method]['n_features']}
    
    Most Efficient: {most_efficient}
    • Accuracy: {results[most_efficient]['accuracy']:.4f}
    • Features: {results[most_efficient]['n_features']}
    • Reduction: {(1-results[most_efficient]['n_features']/X.shape[1])*100:.1f}%
    
    INSIGHTS:
    • {len(high_corr_pairs)} highly correlated features
    • PCA effective for noise reduction
    • LDA good for classification tasks
    • Feature selection preserves interpretability
    """
    
    axes[1, 2].text(0.05, 0.95, recommendations, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification reports
    print(f"\n{'='*60}")
    print("DETAILED CLASSIFICATION REPORTS:")
    print(f"{'='*60}")
    
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print("-" * len(method_name))
        print(classification_report(y_test, result['predictions'], target_names=data.target_names))
    
    return results
```

### 7.6 Chapter Exercises

#### **Exercise 7.1: PCA Implementation Challenge**
```python
# Exercise 7.1: Implement PCA with different initialization methods
def exercise_pca_implementation():
    """
    Exercise: Implement PCA with SVD and compare with eigendecomposition
    
    Tasks:
    1. Implement PCA using SVD (Singular Value Decomposition)
    2. Compare with eigendecomposition method
    3. Analyze numerical stability and performance
    """
    
    # TODO: Implement SVD-based PCA
    class PCA_SVD:
        def __init__(self, n_components=None):
            self.n_components = n_components
            # TODO: Initialize other necessary attributes
            
        def fit(self, X):
            # TODO: Implement PCA using SVD
            # Hint: Use np.linalg.svd()
            pass
            
        def transform(self, X):
            # TODO: Transform data using learned components
            pass
            
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    # TODO: Compare with eigendecomposition method
    # TODO: Test numerical stability with different datasets
    # TODO: Performance comparison
    
    print("Exercise 7.1: Implement and test your PCA_SVD class")

# Exercise 7.2: t-SNE Parameter Exploration
def exercise_tsne_exploration():
    """
    Exercise: Comprehensive t-SNE parameter exploration
    
    Tasks:
    1. Implement parameter grid search for t-SNE
    2. Create evaluation metrics for visualization quality
    3. Build interactive parameter selector
    """
    
    # TODO: Load different datasets (digits, faces, synthetic)
    # TODO: Implement grid search over perplexity and learning rate
    # TODO: Define visualization quality metrics
    # TODO: Create recommendation system for parameters
    
    print("Exercise 7.2: Explore t-SNE parameters systematically")

# Exercise 7.3: Dimensionality Reduction for Streaming Data
def exercise_streaming_pca():
    """
    Exercise: Implement streaming/online PCA
    
    Tasks:
    1. Implement incremental PCA from scratch
    2. Compare with batch PCA on streaming data
    3. Analyze memory usage and accuracy trade-offs
    """
    
    # TODO: Implement online PCA algorithm
    # TODO: Simulate streaming data scenario
    # TODO: Compare accuracy with batch processing
    # TODO: Analyze computational and memory efficiency
    
    print("Exercise 7.3: Implement streaming PCA algorithm")
```

### 7.7 Chapter Summary

#### **Key Learning Outcomes Achieved**

✅ **Curse of Dimensionality Understanding**
- Identified high-dimensional data problems and their impact
- Analyzed distance concentration and sparsity issues
- Evaluated computational complexity considerations
- Developed diagnostic tools for high-dimensional datasets

✅ **Principal Component Analysis (PCA) Mastery**
- Understood mathematical foundations (eigendecomposition, covariance)
- Implemented PCA from scratch with complete derivation
- Mastered component selection techniques (elbow method, cross-validation)
- Applied PCA for noise reduction, compression, and visualization

✅ **Advanced Dimensionality Reduction Techniques**
- Implemented Linear Discriminant Analysis (LDA) for supervised reduction
- Mastered t-SNE for non-linear visualization with parameter optimization
- Compared feature selection vs feature extraction approaches
- Developed intelligent method selection frameworks

✅ **Practical Applications and Integration**
- Built complete ML pipelines with dimensionality reduction preprocessing
- Analyzed computational efficiency and scalability considerations
- Created comprehensive visualization and exploration tools
- Implemented real-world case studies (image compression, data visualization)

#### **Technical Skills Developed**

💻 **Implementation Expertise**
- From-scratch algorithm implementations with mathematical rigor
- Efficient computation techniques for large datasets
- Parameter optimization and hyperparameter tuning
- Pipeline integration with scikit-learn ecosystem

📊 **Analysis and Evaluation**
- Comprehensive evaluation metrics and quality assessment
- Trade-off analysis (compression vs accuracy, speed vs quality)
- Method selection based on data characteristics and requirements
- Performance benchmarking and scalability analysis

🔧 **Practical Tools**
- Interactive exploration and visualization frameworks
- Automated method recommendation systems
- Computational efficiency analysis tools
- Real-world application pipelines

#### **Industry Applications Covered**

🖼️ **Computer Vision**
- Image compression and noise reduction
- Feature extraction for image classification
- Facial recognition preprocessing

📈 **Data Science and Analytics**
- Exploratory data analysis and visualization
- High-dimensional data preprocessing
- Pattern recognition and cluster analysis

🔬 **Scientific Computing**
- Genomics and bioinformatics applications
- Signal processing and noise reduction
- Experimental data analysis

#### **Best Practices and Guidelines**

**When to Use Each Method:**

| Method | Best For | Avoid When |
|--------|----------|------------|
| **PCA** | General preprocessing, noise reduction, visualization | Need interpretable features |
| **LDA** | Classification tasks, supervised reduction | Single class or unlabeled data |
| **t-SNE** | Exploratory visualization, cluster discovery | Preprocessing for ML, large datasets |
| **Feature Selection** | Model interpretability, regulatory compliance | High noise, redundant features |

**Implementation Recommendations:**

1. **Data Preprocessing**: Always standardize features for distance-based methods
2. **Component Selection**: Use multiple criteria (variance, cross-validation, domain knowledge)
3. **Method Selection**: Consider data size, supervised/unsupervised, linear/non-linear requirements
4. **Performance**: Use incremental methods for large datasets, consider computational constraints
5. **Evaluation**: Validate using both intrinsic quality metrics and downstream task performance
6. **Visualization**: Combine multiple techniques for comprehensive data exploration

#### **Common Pitfalls and Solutions**

❌ **Common Mistakes:**
- Applying PCA to non-scaled data
- Using t-SNE for preprocessing instead of visualization
- Selecting components based on arbitrary thresholds
- Ignoring computational complexity for large datasets

✅ **Best Practices:**
- Scale features appropriately for each method
- Use PCA for preprocessing, t-SNE for visualization
- Select components based on downstream task performance
- Consider incremental/streaming methods for scalability

#### **Integration with Machine Learning Pipeline**

The dimensionality reduction techniques covered integrate seamlessly with:
- **Classification and Regression**: Preprocessing to improve performance and reduce overfitting
- **Clustering**: Visualization and noise reduction for better cluster discovery
- **Feature Engineering**: Automated feature creation and selection
- **Model Deployment**: Efficient models with reduced computational requirements

#### **Preparation for Advanced Topics**

This chapter provides foundation for:
- **Deep Learning**: Understanding of autoencoders and representation learning
- **Manifold Learning**: Advanced non-linear dimensionality reduction
- **Big Data**: Distributed and streaming dimensionality reduction
- **Domain-Specific Applications**: Specialized techniques for images, text, and signals

**End of Chapter 7: Dimensionality Reduction**

---

**Next:** Chapter 8 will focus on **End-to-End Machine Learning Projects**, integrating all learned techniques into complete, real-world applications with proper methodology and deployment considerations.

---
