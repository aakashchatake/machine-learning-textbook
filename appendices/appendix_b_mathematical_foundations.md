# Appendix B: Mathematical Foundations

## B.1 Introduction

This appendix provides a comprehensive review of the mathematical concepts essential for understanding machine learning algorithms. The material is designed to serve as both a refresher for those familiar with these topics and a learning resource for newcomers.

---

## B.2 Linear Algebra Essentials

### B.2.1 Vectors and Vector Operations

#### Vector Definitions
A vector **v** in n-dimensional space is represented as:
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

#### Vector Operations

**Vector Addition:**
$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$

**Scalar Multiplication:**
$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

**Dot Product:**
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1v_1 + u_2v_2 + \cdots + u_nv_n$$

**Vector Magnitude (Norm):**
$$||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Vector operations in Python
def demonstrate_vector_operations():
    """Demonstrate basic vector operations"""
    
    # Define vectors
    u = np.array([3, 4])
    v = np.array([1, 2])
    
    print("Vector Operations Demo")
    print(f"u = {u}")
    print(f"v = {v}")
    
    # Addition
    addition = u + v
    print(f"u + v = {addition}")
    
    # Scalar multiplication
    scalar_mult = 2 * u
    print(f"2 * u = {scalar_mult}")
    
    # Dot product
    dot_product = np.dot(u, v)
    print(f"u · v = {dot_product}")
    
    # Magnitude
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    print(f"||u|| = {magnitude_u:.3f}")
    print(f"||v|| = {magnitude_v:.3f}")
    
    # Visualization
    plt.figure(figsize=(10, 4))
    
    # Plot vectors
    plt.subplot(1, 2, 1)
    plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='blue', label='u')
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red', label='v')
    plt.quiver(0, 0, addition[0], addition[1], angles='xy', scale_units='xy', scale=1, color='green', label='u+v')
    plt.xlim(-1, 6)
    plt.ylim(-1, 7)
    plt.grid(True)
    plt.legend()
    plt.title('Vector Addition')
    plt.axis('equal')
    
    # Plot dot product geometric interpretation
    plt.subplot(1, 2, 2)
    theta = np.arccos(dot_product / (magnitude_u * magnitude_v))
    plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='blue', label=f'u (||u||={magnitude_u:.2f})')
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red', label=f'v (||v||={magnitude_v:.2f})')
    
    # Project v onto u
    proj_v_on_u = (dot_product / (magnitude_u**2)) * u
    plt.quiver(0, 0, proj_v_on_u[0], proj_v_on_u[1], angles='xy', scale_units='xy', scale=1, color='orange', label='proj_u(v)')
    
    plt.xlim(-1, 4)
    plt.ylim(-1, 5)
    plt.grid(True)
    plt.legend()
    plt.title(f'Dot Product: {dot_product:.2f}\nAngle: {np.degrees(theta):.1f}°')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'vectors': {'u': u, 'v': v},
        'operations': {
            'addition': addition,
            'dot_product': dot_product,
            'angle': np.degrees(theta)
        }
    }

# Run demonstration
vector_demo = demonstrate_vector_operations()
```

### B.2.2 Matrices and Matrix Operations

#### Matrix Definitions
An m×n matrix **A** is a rectangular array:
$$\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

#### Matrix Operations

**Matrix Addition:**
$$(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$$

**Matrix Multiplication:**
$$(\mathbf{AB})_{ij} = \sum_{k=1}^{p} a_{ik}b_{kj}$$

**Transpose:**
$$(\mathbf{A}^T)_{ij} = a_{ji}$$

**Identity Matrix:**
$$\mathbf{I} = \begin{bmatrix} 
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}$$

```python
def demonstrate_matrix_operations():
    """Demonstrate essential matrix operations"""
    
    print("Matrix Operations Demo")
    
    # Define matrices
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    
    C = np.array([[1, 1],
                  [2, 2]])
    
    print(f"Matrix A (2x3):\n{A}")
    print(f"Matrix B (3x2):\n{B}")
    print(f"Matrix C (2x2):\n{C}")
    
    # Matrix multiplication
    AB = np.dot(A, B)  # or A @ B
    print(f"\nA × B (2x2):\n{AB}")
    
    # Matrix addition (same dimensions)
    AB_plus_C = AB + C
    print(f"\n(A × B) + C:\n{AB_plus_C}")
    
    # Transpose
    A_T = A.T
    print(f"\nA^T (3x2):\n{A_T}")
    
    # Identity matrix
    I = np.eye(2)
    print(f"\nIdentity matrix (2x2):\n{I}")
    
    # Verify identity property
    AB_times_I = AB @ I
    print(f"\n(A × B) × I = A × B:\n{AB_times_I}")
    print(f"Equal to AB? {np.allclose(AB, AB_times_I)}")
    
    return {
        'matrices': {'A': A, 'B': B, 'C': C},
        'products': {'AB': AB, 'AB_plus_C': AB_plus_C},
        'transpose': A_T
    }

matrix_demo = demonstrate_matrix_operations()
```

### B.2.3 Eigenvalues and Eigenvectors

For a square matrix **A**, an eigenvector **v** and eigenvalue λ satisfy:
$$\mathbf{Av} = \lambda\mathbf{v}$$

#### Characteristic Equation
$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

```python
def demonstrate_eigenvalues():
    """Demonstrate eigenvalue decomposition"""
    
    print("Eigenvalue Decomposition Demo")
    
    # Create a symmetric matrix
    A = np.array([[4, -2],
                  [-2, 1]])
    
    print(f"Matrix A:\n{A}")
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # Verify Av = λv for each eigenvalue/eigenvector pair
    for i, (λ, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ v
        λv = λ * v
        print(f"\nEigenvalue {i+1}: λ = {λ:.3f}")
        print(f"Eigenvector: v = {v}")
        print(f"Av = {Av}")
        print(f"λv = {λv}")
        print(f"Av ≈ λv? {np.allclose(Av, λv)}")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    # Original vectors
    plt.subplot(1, 3, 1)
    x = np.linspace(-3, 3, 10)
    y = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, y)
    
    # Apply transformation to grid
    for i in range(len(x)):
        for j in range(len(y)):
            if i % 2 == 0 and j % 2 == 0:  # Subsample for clarity
                original = np.array([X[i,j], Y[i,j]])
                if np.linalg.norm(original) > 0.1:  # Avoid zero vector
                    transformed = A @ original
                    plt.arrow(0, 0, original[0], original[1], 
                            head_width=0.1, head_length=0.1, 
                            fc='blue', ec='blue', alpha=0.3)
    
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.title('Original Vectors')
    plt.axis('equal')
    
    # Transformed vectors
    plt.subplot(1, 3, 2)
    for i in range(len(x)):
        for j in range(len(y)):
            if i % 2 == 0 and j % 2 == 0:
                original = np.array([X[i,j], Y[i,j]])
                if np.linalg.norm(original) > 0.1:
                    transformed = A @ original
                    plt.arrow(0, 0, transformed[0], transformed[1], 
                            head_width=0.1, head_length=0.1, 
                            fc='red', ec='red', alpha=0.3)
    
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.title('Transformed Vectors')
    plt.axis('equal')
    
    # Eigenvectors
    plt.subplot(1, 3, 3)
    colors = ['green', 'purple']
    for i, (λ, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        # Plot eigenvector and its transformation
        plt.arrow(0, 0, v[0], v[1], 
                head_width=0.1, head_length=0.1, 
                fc=colors[i], ec=colors[i], 
                label=f'v{i+1} (λ={λ:.2f})')
        plt.arrow(0, 0, λ*v[0], λ*v[1], 
                head_width=0.1, head_length=0.1, 
                fc=colors[i], ec=colors[i], alpha=0.5,
                linestyle='--')
    
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.title('Eigenvectors and Eigenvalues')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'matrix': A,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }

eigen_demo = demonstrate_eigenvalues()
```

### B.2.4 Principal Component Analysis (PCA) Mathematics

PCA finds the eigenvectors of the covariance matrix to identify principal components.

Given data matrix **X** (n×p), the covariance matrix is:
$$\mathbf{C} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$$

The principal components are the eigenvectors of **C** corresponding to the largest eigenvalues.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

def demonstrate_pca_mathematics():
    """Demonstrate the mathematics behind PCA"""
    
    print("PCA Mathematics Demo")
    
    # Load iris dataset
    data = load_iris()
    X = data.data
    y = data.target
    
    print(f"Data shape: {X.shape}")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    n = X_centered.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)
    
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # Manual PCA calculation
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Explained variance ratio: {eigenvalues / np.sum(eigenvalues)}")
    
    # Project data onto first two principal components
    PC1 = eigenvectors[:, 0]
    PC2 = eigenvectors[:, 1]
    
    projected_data = X_centered @ np.column_stack([PC1, PC2])
    
    # Compare with sklearn PCA
    pca = PCA(n_components=2)
    sklearn_projected = pca.fit_transform(X)
    
    print(f"\nManual PCA first PC: {PC1}")
    print(f"Sklearn PCA first PC: {pca.components_[0]}")
    print(f"Components match? {np.allclose(np.abs(PC1), np.abs(pca.components_[0]))}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data (first two features)
    plt.subplot(1, 3, 1)
    colors = ['red', 'green', 'blue']
    for i in range(3):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=data.target_names[i])
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Original Data (First 2 Features)')
    plt.legend()
    plt.grid(True)
    
    # PCA projected data (manual)
    plt.subplot(1, 3, 2)
    for i in range(3):
        mask = y == i
        plt.scatter(projected_data[mask, 0], projected_data[mask, 1], c=colors[i])
    plt.xlabel(f'PC1 ({eigenvalues[0]/np.sum(eigenvalues):.1%} var)')
    plt.ylabel(f'PC2 ({eigenvalues[1]/np.sum(eigenvalues):.1%} var)')
    plt.title('Manual PCA Projection')
    plt.grid(True)
    
    # PCA projected data (sklearn)
    plt.subplot(1, 3, 3)
    for i in range(3):
        mask = y == i
        plt.scatter(sklearn_projected[mask, 0], sklearn_projected[mask, 1], c=colors[i])
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title('Sklearn PCA Projection')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'covariance_matrix': cov_matrix,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'manual_projection': projected_data,
        'sklearn_projection': sklearn_projected
    }

pca_demo = demonstrate_pca_mathematics()
```

---

## B.3 Statistics and Probability Review

### B.3.1 Descriptive Statistics

#### Central Tendency
- **Mean**: $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Median**: Middle value when data is sorted
- **Mode**: Most frequently occurring value

#### Dispersion
- **Variance**: $\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \mu)^2$
- **Standard Deviation**: $\sigma = \sqrt{\sigma^2}$
- **Range**: $\max(x) - \min(x)$

#### Distribution Shape
- **Skewness**: Measure of asymmetry
- **Kurtosis**: Measure of tail heaviness

```python
import scipy.stats as stats

def demonstrate_descriptive_statistics():
    """Demonstrate descriptive statistics"""
    
    print("Descriptive Statistics Demo")
    
    # Generate sample data
    np.random.seed(42)
    data_normal = np.random.normal(50, 15, 1000)
    data_skewed = np.random.exponential(2, 1000)
    
    datasets = {
        'Normal Distribution': data_normal,
        'Skewed Distribution': data_skewed
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, data) in enumerate(datasets.items()):
        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data, ddof=1)
        variance = np.var(data, ddof=1)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        print(f"\n{name}:")
        print(f"  Mean: {mean:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Std Dev: {std:.2f}")
        print(f"  Variance: {variance:.2f}")
        print(f"  Skewness: {skewness:.2f}")
        print(f"  Kurtosis: {kurtosis:.2f}")
        
        # Histogram
        plt.subplot(2, 2, 2*i + 1)
        plt.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
        plt.title(f'{name}\nSkewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2*i + 2)
        plt.boxplot(data, vert=True)
        plt.title(f'{name} - Box Plot')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'normal_stats': {
            'mean': np.mean(data_normal),
            'std': np.std(data_normal, ddof=1),
            'skewness': stats.skew(data_normal)
        },
        'skewed_stats': {
            'mean': np.mean(data_skewed),
            'std': np.std(data_skewed, ddof=1),
            'skewness': stats.skew(data_skewed)
        }
    }

stats_demo = demonstrate_descriptive_statistics()
```

### B.3.2 Probability Distributions

#### Normal Distribution
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

#### Binomial Distribution
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

#### Poisson Distribution
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

```python
def demonstrate_probability_distributions():
    """Demonstrate common probability distributions"""
    
    print("Probability Distributions Demo")
    
    plt.figure(figsize=(15, 10))
    
    # Normal Distribution
    plt.subplot(2, 3, 1)
    x = np.linspace(-4, 4, 100)
    for mu, sigma in [(0, 1), (0, 0.5), (1, 1)]:
        y = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, y, label=f'μ={mu}, σ={sigma}')
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    
    # Binomial Distribution
    plt.subplot(2, 3, 2)
    x = np.arange(0, 21)
    for n, p in [(20, 0.3), (20, 0.5), (20, 0.7)]:
        y = stats.binom.pmf(x, n, p)
        plt.plot(x, y, 'o-', label=f'n={n}, p={p}')
    plt.title('Binomial Distribution')
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # Poisson Distribution
    plt.subplot(2, 3, 3)
    x = np.arange(0, 15)
    for lam in [1, 3, 5]:
        y = stats.poisson.pmf(x, lam)
        plt.plot(x, y, 'o-', label=f'λ={lam}')
    plt.title('Poisson Distribution')
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # Exponential Distribution
    plt.subplot(2, 3, 4)
    x = np.linspace(0, 5, 100)
    for lam in [0.5, 1, 2]:
        y = stats.expon.pdf(x, scale=1/lam)
        plt.plot(x, y, label=f'λ={lam}')
    plt.title('Exponential Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    
    # Uniform Distribution
    plt.subplot(2, 3, 5)
    x = np.linspace(-1, 3, 100)
    for a, b in [(0, 1), (0, 2), (-0.5, 1.5)]:
        y = stats.uniform.pdf(x, a, b-a)
        plt.plot(x, y, label=f'a={a}, b={b}')
    plt.title('Uniform Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    
    # Beta Distribution
    plt.subplot(2, 3, 6)
    x = np.linspace(0, 1, 100)
    for alpha, beta in [(0.5, 0.5), (2, 2), (5, 1)]:
        y = stats.beta.pdf(x, alpha, beta)
        plt.plot(x, y, label=f'α={alpha}, β={beta}')
    plt.title('Beta Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return "Probability distributions demonstrated"

prob_demo = demonstrate_probability_distributions()
```

### B.3.3 Bayes' Theorem and Conditional Probability

**Bayes' Theorem:**
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

Where:
- $P(A|B)$ is the posterior probability
- $P(B|A)$ is the likelihood
- $P(A)$ is the prior probability
- $P(B)$ is the marginal probability

```python
def demonstrate_bayes_theorem():
    """Demonstrate Bayes' theorem with practical example"""
    
    print("Bayes' Theorem Demo: Medical Diagnosis")
    
    # Medical diagnosis example
    # Disease prevalence (prior)
    P_disease = 0.01  # 1% of population has the disease
    P_no_disease = 1 - P_disease
    
    # Test accuracy
    P_positive_given_disease = 0.95  # Sensitivity (true positive rate)
    P_negative_given_no_disease = 0.90  # Specificity (true negative rate)
    P_positive_given_no_disease = 1 - P_negative_given_no_disease  # False positive rate
    
    # Calculate marginal probability of positive test
    P_positive = (P_positive_given_disease * P_disease + 
                 P_positive_given_no_disease * P_no_disease)
    
    # Apply Bayes' theorem
    P_disease_given_positive = ((P_positive_given_disease * P_disease) / P_positive)
    
    print(f"Prior probability of disease: {P_disease:.1%}")
    print(f"Test sensitivity (P(+|Disease)): {P_positive_given_disease:.1%}")
    print(f"Test specificity (P(-|No Disease)): {P_negative_given_no_disease:.1%}")
    print(f"Probability of positive test: {P_positive:.1%}")
    print(f"Posterior probability (P(Disease|+)): {P_disease_given_positive:.1%}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Create confusion matrix visualization
    population = 100000
    diseased = int(population * P_disease)
    healthy = population - diseased
    
    true_positives = int(diseased * P_positive_given_disease)
    false_negatives = diseased - true_positives
    true_negatives = int(healthy * P_negative_given_no_disease)
    false_positives = healthy - true_negatives
    
    # Confusion matrix
    plt.subplot(2, 2, 1)
    conf_matrix = np.array([[true_negatives, false_positives],
                           [false_negatives, true_positives]])
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Predicted Negative', 'Predicted Positive'],
               yticklabels=['Actually Healthy', 'Actually Diseased'])
    plt.title('Confusion Matrix\n(Population: 100,000)')
    
    # Prior vs Posterior
    plt.subplot(2, 2, 2)
    categories = ['Prior\n(Population)', 'Posterior\n(Test Positive)']
    probabilities = [P_disease, P_disease_given_positive]
    
    bars = plt.bar(categories, probabilities, color=['lightblue', 'darkblue'])
    plt.ylabel('Probability of Disease')
    plt.title('Prior vs Posterior Probability')
    plt.ylim(0, max(probabilities) * 1.2)
    
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{prob:.1%}', ha='center', va='bottom')
    
    # Effect of prevalence
    plt.subplot(2, 2, 3)
    prevalences = np.logspace(-4, -1, 50)  # 0.01% to 10%
    posteriors = []
    
    for prev in prevalences:
        P_pos = P_positive_given_disease * prev + P_positive_given_no_disease * (1 - prev)
        posterior = (P_positive_given_disease * prev) / P_pos
        posteriors.append(posterior)
    
    plt.semilogx(prevalences * 100, np.array(posteriors) * 100)
    plt.axvline(P_disease * 100, color='red', linestyle='--', 
               label=f'Current prevalence ({P_disease:.1%})')
    plt.axhline(P_disease_given_positive * 100, color='red', linestyle='--',
               label=f'Current posterior ({P_disease_given_positive:.1%})')
    plt.xlabel('Disease Prevalence (%)')
    plt.ylabel('Posterior Probability (%)')
    plt.title('Effect of Disease Prevalence')
    plt.grid(True)
    plt.legend()
    
    # Effect of test accuracy
    plt.subplot(2, 2, 4)
    sensitivities = np.linspace(0.5, 1.0, 50)
    posteriors_sens = []
    
    for sens in sensitivities:
        P_pos = sens * P_disease + P_positive_given_no_disease * P_no_disease
        posterior = (sens * P_disease) / P_pos
        posteriors_sens.append(posterior)
    
    plt.plot(sensitivities * 100, np.array(posteriors_sens) * 100)
    plt.axvline(P_positive_given_disease * 100, color='red', linestyle='--',
               label=f'Current sensitivity ({P_positive_given_disease:.1%})')
    plt.axhline(P_disease_given_positive * 100, color='red', linestyle='--',
               label=f'Current posterior ({P_disease_given_positive:.1%})')
    plt.xlabel('Test Sensitivity (%)')
    plt.ylabel('Posterior Probability (%)')
    plt.title('Effect of Test Sensitivity')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'prior': P_disease,
        'posterior': P_disease_given_positive,
        'confusion_matrix': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    }

bayes_demo = demonstrate_bayes_theorem()
```

---

## B.4 Calculus Concepts for ML

### B.4.1 Derivatives and Gradients

#### Single Variable Derivatives
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

#### Partial Derivatives
For multivariable functions:
$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

#### Gradient
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

```python
def demonstrate_gradients():
    """Demonstrate gradient computation and visualization"""
    
    print("Gradient Computation Demo")
    
    # Define a simple function: f(x, y) = x^2 + y^2 + 2xy
    def f(x, y):
        return x**2 + y**2 + 2*x*y
    
    # Analytical gradients
    def grad_f_analytical(x, y):
        df_dx = 2*x + 2*y
        df_dy = 2*y + 2*x
        return np.array([df_dx, df_dy])
    
    # Numerical gradients
    def grad_f_numerical(x, y, h=1e-5):
        df_dx = (f(x+h, y) - f(x-h, y)) / (2*h)
        df_dy = (f(x, y+h) - f(x, y-h)) / (2*h)
        return np.array([df_dx, df_dy])
    
    # Test point
    x0, y0 = 1.5, 1.0
    
    grad_analytical = grad_f_analytical(x0, y0)
    grad_numerical = grad_f_numerical(x0, y0)
    
    print(f"At point ({x0}, {y0}):")
    print(f"Analytical gradient: {grad_analytical}")
    print(f"Numerical gradient: {grad_numerical}")
    print(f"Difference: {np.linalg.norm(grad_analytical - grad_numerical):.2e}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Function surface
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    plt.subplot(1, 3, 1)
    contour = plt.contour(X, Y, Z, levels=20)
    plt.colorbar(contour)
    plt.plot(x0, y0, 'ro', markersize=8, label=f'Point ({x0}, {y0})')
    
    # Plot gradient vector
    plt.arrow(x0, y0, -grad_analytical[0]*0.2, -grad_analytical[1]*0.2,
             head_width=0.1, head_length=0.1, fc='red', ec='red', 
             label='Negative gradient')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Contours and Gradient')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # Gradient descent visualization
    plt.subplot(1, 3, 2)
    
    # Perform gradient descent
    x_current, y_current = 2.5, 2.0
    learning_rate = 0.1
    trajectory_x = [x_current]
    trajectory_y = [y_current]
    values = [f(x_current, y_current)]
    
    for i in range(20):
        grad = grad_f_analytical(x_current, y_current)
        x_current -= learning_rate * grad[0]
        y_current -= learning_rate * grad[1]
        trajectory_x.append(x_current)
        trajectory_y.append(y_current)
        values.append(f(x_current, y_current))
    
    plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.plot(trajectory_x, trajectory_y, 'ro-', linewidth=2, markersize=4)
    plt.plot(trajectory_x[0], trajectory_y[0], 'go', markersize=10, label='Start')
    plt.plot(trajectory_x[-1], trajectory_y[-1], 'bo', markersize=10, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Trajectory')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # Convergence plot
    plt.subplot(1, 3, 3)
    plt.plot(values, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Gradient Descent Convergence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'analytical_gradient': grad_analytical,
        'numerical_gradient': grad_numerical,
        'trajectory': {'x': trajectory_x, 'y': trajectory_y, 'values': values}
    }

grad_demo = demonstrate_gradients()
```

### B.4.2 Chain Rule and Backpropagation

The chain rule is fundamental to backpropagation in neural networks:

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u} \cdot \frac{\partial u}{\partial x}$$

For a composition $f(g(x))$:
$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

```python
def demonstrate_chain_rule():
    """Demonstrate chain rule computation"""
    
    print("Chain Rule and Backpropagation Demo")
    
    # Simple neural network example: f(x) = sigmoid(w2 * relu(w1 * x + b1) + b2)
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    # Network parameters
    w1, b1 = 0.5, 0.1
    w2, b2 = 0.8, -0.2
    
    # Forward pass
    def forward_pass(x):
        # Layer 1: linear transformation
        z1 = w1 * x + b1
        # Layer 1: activation
        a1 = relu(z1)
        # Layer 2: linear transformation
        z2 = w2 * a1 + b2
        # Layer 2: activation (output)
        a2 = sigmoid(z2)
        
        return {
            'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2
        }
    
    # Backward pass (manual chain rule)
    def backward_pass(forward_values, target):
        x = forward_values['x']
        z1 = forward_values['z1']
        a1 = forward_values['a1']
        z2 = forward_values['z2']
        a2 = forward_values['a2']
        
        # Loss function: L = 0.5 * (a2 - target)^2
        loss = 0.5 * (a2 - target)**2
        
        # Backward pass using chain rule
        # dL/da2
        dL_da2 = a2 - target
        
        # dL/dz2 = dL/da2 * da2/dz2
        dL_dz2 = dL_da2 * sigmoid_derivative(z2)
        
        # dL/dw2 = dL/dz2 * dz2/dw2 = dL/dz2 * a1
        dL_dw2 = dL_dz2 * a1
        
        # dL/db2 = dL/dz2 * dz2/db2 = dL/dz2 * 1
        dL_db2 = dL_dz2
        
        # dL/da1 = dL/dz2 * dz2/da1 = dL/dz2 * w2
        dL_da1 = dL_dz2 * w2
        
        # dL/dz1 = dL/da1 * da1/dz1
        dL_dz1 = dL_da1 * relu_derivative(z1)
        
        # dL/dw1 = dL/dz1 * dz1/dw1 = dL/dz1 * x
        dL_dw1 = dL_dz1 * x
        
        # dL/db1 = dL/dz1 * dz1/db1 = dL/dz1 * 1
        dL_db1 = dL_dz1
        
        return {
            'loss': loss,
            'dL_dw1': dL_dw1,
            'dL_db1': dL_db1,
            'dL_dw2': dL_dw2,
            'dL_db2': dL_db2
        }
    
    # Numerical gradients for verification
    def numerical_gradient(x, target, param_name, h=1e-5):
        global w1, b1, w2, b2
        
        # Store original value
        if param_name == 'w1':
            original = w1
            w1 += h
            loss_plus = 0.5 * (forward_pass(x)['a2'] - target)**2
            w1 = original - h
            loss_minus = 0.5 * (forward_pass(x)['a2'] - target)**2
            w1 = original
        elif param_name == 'b1':
            original = b1
            b1 += h
            loss_plus = 0.5 * (forward_pass(x)['a2'] - target)**2
            b1 = original - h
            loss_minus = 0.5 * (forward_pass(x)['a2'] - target)**2
            b1 = original
        elif param_name == 'w2':
            original = w2
            w2 += h
            loss_plus = 0.5 * (forward_pass(x)['a2'] - target)**2
            w2 = original - h
            loss_minus = 0.5 * (forward_pass(x)['a2'] - target)**2
            w2 = original
        elif param_name == 'b2':
            original = b2
            b2 += h
            loss_plus = 0.5 * (forward_pass(x)['a2'] - target)**2
            b2 = original - h
            loss_minus = 0.5 * (forward_pass(x)['a2'] - target)**2
            b2 = original
        
        return (loss_plus - loss_minus) / (2 * h)
    
    # Test with a sample input
    x_test = 2.0
    target = 0.8
    
    # Forward and backward pass
    forward_values = forward_pass(x_test)
    backward_values = backward_pass(forward_values, target)
    
    print(f"Input: {x_test}, Target: {target}")
    print(f"Output: {forward_values['a2']:.4f}")
    print(f"Loss: {backward_values['loss']:.4f}")
    
    print("\nGradient Verification:")
    gradients = ['dL_dw1', 'dL_db1', 'dL_dw2', 'dL_db2']
    param_names = ['w1', 'b1', 'w2', 'b2']
    
    for grad_name, param_name in zip(gradients, param_names):
        analytical = backward_values[grad_name]
        numerical = numerical_gradient(x_test, target, param_name)
        print(f"{param_name}: Analytical = {analytical:.6f}, Numerical = {numerical:.6f}, "
              f"Diff = {abs(analytical - numerical):.2e}")
    
    # Visualization of function and gradients
    plt.figure(figsize=(15, 5))
    
    # Function output vs input
    x_range = np.linspace(-2, 4, 100)
    outputs = [forward_pass(x)['a2'] for x in x_range]
    
    plt.subplot(1, 3, 1)
    plt.plot(x_range, outputs, 'b-', linewidth=2, label='Network Output')
    plt.plot(x_test, forward_values['a2'], 'ro', markersize=8, label=f'Test Point')
    plt.axhline(target, color='g', linestyle='--', label='Target')
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.title('Neural Network Function')
    plt.legend()
    plt.grid(True)
    
    # Gradient descent on parameters
    plt.subplot(1, 3, 2)
    
    # Simulate parameter updates
    w1_history = [w1]
    w2_history = [w2]
    loss_history = [backward_values['loss']]
    
    learning_rate = 0.1
    current_w1, current_w2 = w1, b1
    
    for i in range(50):
        # Use current parameters
        w1, w2 = current_w1, current_w2
        forward_vals = forward_pass(x_test)
        backward_vals = backward_pass(forward_vals, target)
        
        # Update parameters
        current_w1 -= learning_rate * backward_vals['dL_dw1']
        current_w2 -= learning_rate * backward_vals['dL_dw2']
        
        w1_history.append(current_w1)
        w2_history.append(current_w2)
        loss_history.append(backward_vals['loss'])
    
    # Restore original parameters
    w1, w2 = 0.5, 0.8
    
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Decrease via Gradient Descent')
    plt.grid(True)
    
    # Parameter space
    plt.subplot(1, 3, 3)
    plt.plot(w1_history, w2_history, 'ro-', linewidth=2, markersize=4)
    plt.plot(w1_history[0], w2_history[0], 'go', markersize=10, label='Start')
    plt.plot(w1_history[-1], w2_history[-1], 'bo', markersize=10, label='End')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title('Parameter Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'forward_pass': forward_values,
        'backward_pass': backward_values,
        'parameter_history': {
            'w1': w1_history,
            'w2': w2_history,
            'loss': loss_history
        }
    }

chain_rule_demo = demonstrate_chain_rule()
```

---

## B.5 Key Formulas and Derivations

### B.5.1 Linear Regression Derivation

Given training data $(x_i, y_i)$ for $i = 1, ..., n$, we want to find parameters $w$ and $b$ that minimize:

$$L(w, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (wx_i + b))^2$$

**Analytical Solution:**
$$w = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$b = \bar{y} - w\bar{x}$$

**Gradient Descent:**
$$\frac{\partial L}{\partial w} = -\frac{1}{n}\sum_{i=1}^{n} x_i(y_i - (wx_i + b))$$

$$\frac{\partial L}{\partial b} = -\frac{1}{n}\sum_{i=1}^{n} (y_i - (wx_i + b))$$

### B.5.2 Logistic Regression Derivation

**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Probability Model:**
$$P(y=1|x) = \sigma(w^Tx + b)$$

**Log-Likelihood:**
$$\ell(w, b) = \sum_{i=1}^{n} [y_i \log(\sigma(w^Tx_i + b)) + (1-y_i) \log(1-\sigma(w^Tx_i + b))]$$

**Gradients:**
$$\frac{\partial \ell}{\partial w} = \sum_{i=1}^{n} (y_i - \sigma(w^Tx_i + b))x_i$$

$$\frac{\partial \ell}{\partial b} = \sum_{i=1}^{n} (y_i - \sigma(w^Tx_i + b))$$

### B.5.3 Support Vector Machine Derivation

**Primal Problem:**
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to:
$$y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Dual Problem:**
$$\max_{\alpha} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j y_i y_j x_i^T x_j$$

Subject to:
$$0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_i y_i = 0$$

### B.5.4 Neural Network Backpropagation

For a layer $l$ with input $a^{(l-1)}$ and output $a^{(l)}$:

**Forward Pass:**
$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

**Backward Pass:**
$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^T$$

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

$$\delta^{(l-1)} = (W^{(l)})^T\delta^{(l)} \odot \sigma'(z^{(l-1)})$$

---

## B.6 Quick Reference Tables

### B.6.1 Common Derivatives

| Function | Derivative |
|----------|------------|
| $f(x) = c$ | $f'(x) = 0$ |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ |
| $f(x) = e^x$ | $f'(x) = e^x$ |
| $f(x) = \ln(x)$ | $f'(x) = \frac{1}{x}$ |
| $f(x) = \sin(x)$ | $f'(x) = \cos(x)$ |
| $f(x) = \cos(x)$ | $f'(x) = -\sin(x)$ |
| $f(x) = \frac{1}{1+e^{-x}}$ | $f'(x) = f(x)(1-f(x))$ |

### B.6.2 Common Distributions Parameters

| Distribution | Parameters | Mean | Variance |
|-------------|------------|------|----------|
| Normal | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| Binomial | $n, p$ | $np$ | $np(1-p)$ |
| Poisson | $\lambda$ | $\lambda$ | $\lambda$ |
| Exponential | $\lambda$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |
| Uniform | $a, b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |

### B.6.3 Matrix Properties

| Property | Formula |
|----------|---------|
| $(AB)^T = B^TA^T$ | Transpose of product |
| $(A^{-1})^T = (A^T)^{-1}$ | Inverse transpose |
| $\det(AB) = \det(A)\det(B)$ | Determinant of product |
| $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$ | Trace linearity |
| $A\mathbf{v} = \lambda\mathbf{v}$ | Eigenvalue equation |

This completes Appendix B with comprehensive mathematical foundations essential for understanding machine learning algorithms, including practical demonstrations and quick reference materials.
