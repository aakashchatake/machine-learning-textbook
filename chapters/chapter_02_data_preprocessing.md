# Chapter 2: Data Preprocessing

> "Data preprocessing is like preparing ingredients before cooking—no matter how skilled the chef, poor ingredients lead to poor results."
> 
> — Anonymous Data Scientist

## What You'll Learn in This Chapter

By the end of this chapter, you'll master:
- Essential data cleaning techniques
- Methods for handling missing values strategically
- Dataset splitting strategies for robust model evaluation
- Data quality assessment and validation
- Best practices for preprocessing pipelines

## The Mathematical Foundation of Data Quality

Before any machine learning algorithm can extract meaningful patterns, the data must satisfy certain mathematical and statistical assumptions. In practice, raw data rarely meets these requirements, making preprocessing not just helpful, but mathematically necessary.

Consider a learning algorithm trying to minimize a loss function L(θ) over a dataset D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}. If our data contains significant noise ε, missing values, or inconsistent scaling, the optimization process becomes:

**L'(θ) = L(θ) + N(ε) + M(missing) + S(scale)**

Where N(ε) represents noise interference, M(missing) accounts for missing value bias, and S(scale) reflects scaling inconsistencies. Each term can dramatically alter the loss landscape, leading algorithms toward suboptimal solutions.

This mathematical reality explains why preprocessing isn't just practical housekeeping—it's about ensuring our algorithms can find the true underlying patterns rather than fitting to artifacts in the data. Think of it as preparing the mathematical foundation upon which learning can occur.

The famous "garbage in, garbage out" principle has deep mathematical roots: if our input data violates the assumptions of our chosen algorithm (independence, normality, homoscedasticity, etc.), even sophisticated models will produce unreliable results.

## The Statistical Reality of Real-World Data

Real-world datasets systematically violate the clean, well-structured assumptions underlying most machine learning theory. Understanding these violations from a statistical perspective helps us choose appropriate preprocessing strategies.

**The Data Quality Spectrum**

From a theoretical standpoint, data quality issues can be categorized by their impact on statistical learning:

1. **Systematic Errors** - These bias our estimators and shift the true data distribution
2. **Random Errors** - These increase variance and reduce signal-to-noise ratio  
3. **Structural Issues** - These violate distributional assumptions (normality, independence, etc.)

Consider how each type affects the fundamental learning equation:

**P(hypothesis|data) ∝ P(data|hypothesis) × P(hypothesis)**

When data quality issues are present, we're actually working with:

**P(hypothesis|corrupted_data) ∝ P(corrupted_data|hypothesis) × P(hypothesis)**

The corrupted likelihood P(corrupted_data|hypothesis) can lead to entirely different posterior beliefs about which hypothesis best explains our observations.

Real-world datasets come with numerous challenges:

```python
# Example of messy real-world data
messy_data = {
    'customer_name': ['John Doe', 'jane smith', 'BOB JOHNSON', None, ''],
    'age': [25, 'thirty', 150, -5, None],
    'income': ['$50,000', '75000', '$invalid$', '', '60K'],
    'email': ['john@email.com', 'invalid-email', 'jane@.com', None, 'bob@email.com'],
    'signup_date': ['2023-01-15', '01/15/2023', 'invalid', '2023-13-45', None]
}
```

**Common Data Quality Issues:**
- **Missing values**: Empty cells, null values, placeholder text
- **Inconsistent formatting**: Different date formats, mixed case text
- **Outliers**: Unrealistic values (age = 150, negative income)
- **Duplicates**: Identical or near-identical records
- **Invalid data**: Malformed emails, impossible dates
- **Mixed data types**: Numbers stored as text, inconsistent units

## Data Cleaning: Statistical Foundations

Data cleaning is fundamentally about identifying and correcting deviations from the true underlying data generating process. From a statistical perspective, we can formalize data cleaning as the process of transforming observed data X_obs to recover the true data X_true.

**Mathematical Framework of Data Quality**

Let's define the relationship between observed and true data:

**X_obs = X_true + ε_noise + ε_systematic + ε_missing**

Where:
- **ε_noise** represents random measurement errors
- **ε_systematic** represents consistent biases or distortions
- **ε_missing** represents information loss due to missing values

The goal of data cleaning is to estimate and minimize these error components, thereby recovering an approximation of X_true that preserves the essential statistical properties needed for learning.

**Data Quality Metrics**

We can quantify data quality using several statistical measures:

1. **Completeness**: C = (1 - missing_rate), where missing_rate = |missing_values| / |total_values|
2. **Consistency**: Measured by entropy reduction after standardization
3. **Accuracy**: Distance between cleaned and true values (when ground truth is available)
4. **Validity**: Proportion of values that satisfy domain constraints

Data cleaning involves systematically identifying and correcting deviations from these quality standards.

### Identifying Data Quality Issues

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    print("📊 DATA QUALITY ASSESSMENT")
    print("=" * 50)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    print(f"\n🔍 Missing Values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_percent[col]:.1f}%)")
    
    # Data types
    print(f"\n📋 Data Types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Potential duplicates
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {duplicates}")
    
    # Unique values (for categorical columns)
    print(f"\n🏷️ Unique Values (categorical columns):")
    for col in df.select_dtypes(include=['object']).columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")
        if unique_count <= 10:  # Show values if few
            print(f"    Values: {list(df[col].dropna().unique())}")

# Example usage
sample_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Alice', None],
    'age': [25, 30, None, 25, 35],
    'salary': [50000, 60000, 75000, 50000, None],
    'department': ['Engineering', 'Sales', 'Engineering', 'Engineering', 'Marketing']
})

assess_data_quality(sample_data)
```

### Handling Inconsistent Data

```python
def clean_text_data(series):
    """Clean and standardize text data"""
    return (series
            .str.strip()                    # Remove leading/trailing spaces
            .str.lower()                    # Convert to lowercase
            .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
            .str.replace(r'\s+', ' ', regex=True)     # Normalize whitespace
           )

def standardize_phone_numbers(series):
    """Standardize phone number format"""
    return (series
            .str.replace(r'[^\d]', '', regex=True)  # Keep only digits
            .str.replace(r'(\d{3})(\d{3})(\d{4})', r'(\1) \2-\3', regex=True)
           )

def parse_currency(series):
    """Convert currency strings to numeric values"""
    return (series
            .str.replace(r'[\$,K]', '', regex=True)  # Remove $, commas, K
            .str.replace('K', '000')  # Convert K to thousands
            .astype(float)
           )

# Example
messy_text = pd.Series(['  JOHN DOE  ', 'jane-smith!!!', 'Bob    Johnson'])
clean_text = clean_text_data(messy_text)
print("Original:", messy_text.tolist())
print("Cleaned:", clean_text.tolist())
```

### Removing Duplicates

```python
def handle_duplicates(df, subset=None, keep='first'):
    """Identify and handle duplicate records"""
    
    # Find duplicates
    duplicates = df.duplicated(subset=subset, keep=False)
    
    print(f"📋 Duplicate Analysis:")
    print(f"Total duplicates: {duplicates.sum()}")
    
    if duplicates.sum() > 0:
        print(f"Duplicate records:")
        print(df[duplicates].sort_values(by=df.columns.tolist()))
        
        # Remove duplicates
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        print(f"Rows before: {len(df)}")
        print(f"Rows after: {len(df_clean)}")
        return df_clean
    
    return df
```

### Outlier Detection and Handling

```python
def detect_outliers(series, method='iqr', threshold=1.5):
    """Detect outliers using different methods"""
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
        
    elif method == 'percentile':
        lower_bound = series.quantile(0.01)
        upper_bound = series.quantile(0.99)
        outliers = (series < lower_bound) | (series > upper_bound)
    
    return outliers

def visualize_outliers(df, column):
    """Visualize outliers in a column"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot
    axes[0].boxplot(df[column].dropna())
    axes[0].set_title(f'{column} - Box Plot')
    axes[0].set_ylabel('Value')
    
    # Histogram
    axes[1].hist(df[column].dropna(), bins=30, alpha=0.7)
    axes[1].set_title(f'{column} - Histogram')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    
    # Scatter plot with outliers highlighted
    outliers = detect_outliers(df[column].dropna())
    axes[2].scatter(range(len(df[column].dropna())), 
                   df[column].dropna(), 
                   c=['red' if x else 'blue' for x in outliers], 
                   alpha=0.6)
    axes[2].set_title(f'{column} - Outliers (red)')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()

# Example usage
np.random.seed(42)
data_with_outliers = np.concatenate([
    np.random.normal(50, 10, 100),  # Normal data
    [150, 200, -50]  # Outliers
])

outliers = detect_outliers(pd.Series(data_with_outliers))
print(f"Detected {outliers.sum()} outliers")
```

## Missing Data: A Statistical Theory Perspective

Missing data poses one of the most significant challenges to valid statistical inference. The mechanism that causes data to be missing determines both the appropriate handling strategy and the validity of our conclusions.

**Rubin's Missing Data Theory**

Donald Rubin's seminal work established the mathematical framework for understanding missing data mechanisms. Let R be a missing data indicator matrix where R_ij = 1 if X_ij is observed and R_ij = 0 if X_ij is missing.

The missing data mechanism is characterized by the conditional probability:

**P(R | X_obs, X_miss, ψ)**

Where ψ represents parameters governing the missing data process.

### Mathematical Classification of Missing Data Mechanisms

**1. Missing Completely at Random (MCAR)**
- **Formal Definition**: P(R | X_obs, X_miss, ψ) = P(R | ψ)
- **Mathematical Implication**: Missing values are statistically independent of all data values
- **Key Property**: The observed data is a random subsample of the complete data
- **Statistical Consequence**: Complete case analysis yields unbiased estimates (though with reduced power)
- **Example**: Laboratory equipment randomly failing during data collection

**2. Missing at Random (MAR)**  
- **Formal Definition**: P(R | X_obs, X_miss, ψ) = P(R | X_obs, ψ)
- **Mathematical Implication**: Missingness depends only on observed data, not on missing values
- **Key Property**: Given the observed data, the missing data mechanism is ignorable
- **Statistical Consequence**: Maximum likelihood and Bayesian inference remain valid under MAR
- **Example**: Survey non-response that varies systematically by age or education level

**3. Missing Not at Random (MNAR)**
- **Formal Definition**: P(R | X_obs, X_miss, ψ) depends on X_miss
- **Mathematical Implication**: Missingness depends on the unobserved values themselves  
- **Key Property**: The missing data mechanism is non-ignorable
- **Statistical Consequence**: Requires explicit modeling of the missing data mechanism
- **Example**: High earners systematically refusing to disclose income information

**Statistical Testing for Missing Data Mechanisms**

We can test MCAR using Little's test, which examines whether missing data patterns are consistent with MCAR:

**H₀**: Data are MCAR
**Test Statistic**: Follows χ² distribution under null hypothesis

Understanding the missing data mechanism is crucial because different mechanisms require fundamentally different analytical approaches.

### Identifying Missing Values

```python
def analyze_missing_values(df):
    """Comprehensive missing value analysis"""
    
    # Calculate missing statistics
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    print("🔍 Missing Value Analysis:")
    print(missing_stats.to_string(index=False))
    
    # Visualize missing patterns
    if not missing_stats.empty:
        plt.figure(figsize=(12, 6))
        
        # Missing value heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        plt.title('Missing Value Pattern')
        
        # Missing value bar chart
        plt.subplot(1, 2, 2)
        missing_stats.plot(x='Column', y='Missing_Percentage', kind='bar', ax=plt.gca())
        plt.title('Missing Values by Column')
        plt.xlabel('Columns')
        plt.ylabel('Missing Percentage (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return missing_stats

# Example dataset with missing values
sample_data = pd.DataFrame({
    'age': [25, None, 35, 40, None, 30],
    'income': [50000, 60000, None, 80000, 55000, None],
    'education': ['Bachelor', 'Master', None, 'PhD', 'Bachelor', 'Master'],
    'experience': [2, 5, None, 15, 3, 7]
})

missing_analysis = analyze_missing_values(sample_data)
```

### Strategies for Handling Missing Values

#### 1. Removal Strategies

```python
def remove_missing_data(df, strategy='rows', threshold=0.5):
    """Remove missing data using different strategies"""
    
    if strategy == 'rows':
        # Remove rows with any missing values
        df_clean = df.dropna()
        print(f"Removed {len(df) - len(df_clean)} rows with missing values")
        
    elif strategy == 'columns':
        # Remove columns with missing values above threshold
        missing_percent = df.isnull().sum() / len(df)
        cols_to_drop = missing_percent[missing_percent > threshold].index
        df_clean = df.drop(columns=cols_to_drop)
        print(f"Removed columns: {list(cols_to_drop)}")
        
    elif strategy == 'selective':
        # Remove rows only if multiple values are missing
        df_clean = df.dropna(thresh=len(df.columns) - 1)  # Keep if at most 1 missing
        print(f"Removed {len(df) - len(df_clean)} rows with multiple missing values")
    
    return df_clean

# Example
print("Original shape:", sample_data.shape)
cleaned_data = remove_missing_data(sample_data, strategy='selective')
print("After cleaning:", cleaned_data.shape)
```

#### 2. Imputation Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute_missing_values(df, strategy='mean', n_neighbors=5):
    """Impute missing values using various strategies"""
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df_imputed = df.copy()
    
    # Handle numeric columns
    if len(numeric_cols) > 0:
        if strategy in ['mean', 'median', 'constant']:
            imputer = SimpleImputer(strategy=strategy, fill_value=0 if strategy=='constant' else None)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
        elif strategy == 'iterative':
            imputer = IterativeImputer(random_state=42)
            df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Handle categorical columns (mode imputation)
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df_imputed

# Compare different imputation strategies
strategies = ['mean', 'median', 'knn']
imputation_results = {}

for strategy in strategies:
    imputed_data = impute_missing_values(sample_data, strategy=strategy)
    imputation_results[strategy] = imputed_data
    
    print(f"\n{strategy.upper()} Imputation Results:")
    print(imputed_data[['age', 'income']].describe())
```

#### 3. Advanced Imputation Techniques

```python
def custom_imputation(df):
    """Custom imputation based on business logic"""
    df_custom = df.copy()
    
    # Example: Impute income based on education level
    education_income_map = {
        'Bachelor': df.groupby('education')['income'].mean()['Bachelor'],
        'Master': df.groupby('education')['income'].mean()['Master'],
        'PhD': df.groupby('education')['income'].mean()['PhD']
    }
    
    # Fill missing income based on education
    for idx, row in df_custom.iterrows():
        if pd.isna(row['income']) and pd.notna(row['education']):
            df_custom.at[idx, 'income'] = education_income_map.get(row['education'], df['income'].mean())
    
    return df_custom

def forward_fill_imputation(df, time_column):
    """Forward fill for time series data"""
    df_sorted = df.sort_values(time_column)
    df_filled = df_sorted.fillna(method='ffill')  # Forward fill
    return df_filled

# Time series example
time_series_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10, freq='D'),
    'temperature': [20, None, 22, None, None, 25, 24, None, 23, 22],
    'humidity': [60, 65, None, 70, 68, None, 72, 71, None, 69]
})

filled_ts = forward_fill_imputation(time_series_data, 'date')
print("Time series with forward fill:")
print(filled_ts)
```

## Dataset Splitting: Statistical Learning Theory

Dataset splitting addresses one of the fundamental challenges in statistical learning: estimating how well our model will perform on future, unseen data. This is formalized through the bias-variance decomposition of generalization error.

**Mathematical Foundation of Generalization**

Consider a learning algorithm that produces hypothesis h based on training data D_train. The true risk (generalization error) is:

**R(h) = E_{(x,y)~P}[L(h(x), y)]**

Since we can't compute this directly, we estimate it using test data D_test:

**R̂(h) = (1/|D_test|) Σ_{(x,y)∈D_test} L(h(x), y)**

The key insight is that R̂(h) is an unbiased estimator of R(h) only if D_test is drawn from the same distribution as future data and is independent of the training process.

**The Fundamental Trade-off in Data Splitting**

When splitting dataset D into training D_train and testing D_test portions, we face a fundamental trade-off:

- **Larger D_train**: Reduces bias in our learned model (more data for learning)
- **Larger D_test**: Reduces variance in our performance estimate (more reliable evaluation)

**Mathematically**: If |D| = n, |D_train| = k, then:
- **Training Error Bias** ∝ 1/k (decreases as training set grows)
- **Test Error Variance** ∝ 1/(n-k) (decreases as test set grows)

**Statistical Properties of Different Split Ratios**

The optimal split ratio depends on the bias-variance characteristics of our learning algorithm:

- **High-bias algorithms** (e.g., linear models) benefit from larger test sets (60-40 or 70-30 splits)
- **High-variance algorithms** (e.g., deep networks) benefit from larger training sets (80-20 or 90-10 splits)

**Cross-Validation: A Statistical Sampling Perspective**

Cross-validation provides a more sophisticated approach by treating dataset splitting as a statistical sampling problem. K-fold CV estimates generalization error as:

**CV_k = (1/k) Σ_{i=1}^k R̂_i(h_i)**

Where each R̂_i is computed on the i-th fold, and h_i is trained on the remaining k-1 folds.

The statistical properties of this estimator are well-understood:
- **Bias**: Generally lower than single train-test split
- **Variance**: Depends on k (higher k → lower bias, higher variance)
- **Computational Cost**: k times higher than single split

Dataset splitting is fundamentally about creating reliable statistical estimates of future performance.

### Train-Test Split Fundamentals

```python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def basic_train_test_split(X, y, test_size=0.2, random_state=42):
    """Basic train-test split with explanation"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"📊 Dataset Split Information:")
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# Example with Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = basic_train_test_split(X, y)

# Check class distribution
print(f"\n🏷️ Class Distribution:")
print(f"Training: {np.bincount(y_train)}")
print(f"Test: {np.bincount(y_test)}")
```

### Stratified Sampling: Statistical Foundations

Stratified sampling addresses a critical statistical issue: ensuring that our training and test distributions remain representative of the population distribution, particularly when dealing with imbalanced classes or important subgroups.

**Mathematical Motivation**

Consider a dataset with class proportions π₁, π₂, ..., πₖ. Under simple random sampling, the probability that our test set has dramatically different class proportions follows a multinomial distribution. For small datasets or rare classes, this can lead to:

1. **Test sets missing entire classes** (probability > 0 for rare classes)
2. **Biased performance estimates** due to distribution shift
3. **Invalid statistical inference** when test ≠ population distribution

**Stratified Sampling Algorithm**

Stratified sampling ensures that each stratum (class) is represented proportionally:

**For each class i**: 
- **n_i^train = ⌊n_i × (1 - test_ratio)⌋**
- **n_i^test = n_i - n_i^train**

This guarantees that **P̂_test(class = i) ≈ P_population(class = i)** for all classes.

**Statistical Properties**

Compared to simple random sampling, stratified sampling provides:

1. **Lower variance** in performance estimates
2. **Unbiased representation** of all subgroups  
3. **More reliable** confidence intervals for performance metrics
4. **Better preservation** of correlation structure between features and target

**When Stratification is Critical**

Stratification becomes mathematically necessary when:
- **Class imbalance ratio > 10:1** (high probability of missing rare classes)
- **Small dataset size** (n < 1000, where sampling variation is high)
- **Multi-class problems** with > 5 classes (combinatorial explosion of possible imbalances)

For imbalanced datasets, stratified sampling ensures each split contains approximately the same percentage of samples from each class, maintaining the statistical validity of our evaluation.

```python
def stratified_split_analysis(X, y, test_size=0.2):
    """Compare regular vs stratified splitting"""
    
    # Regular split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Stratified split
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Compare distributions
    print("📊 Split Comparison:")
    print("\nRegular Split:")
    print(f"  Train distribution: {np.bincount(y_train_reg) / len(y_train_reg)}")
    print(f"  Test distribution:  {np.bincount(y_test_reg) / len(y_test_reg)}")
    
    print("\nStratified Split:")
    print(f"  Train distribution: {np.bincount(y_train_strat) / len(y_train_strat)}")
    print(f"  Test distribution:  {np.bincount(y_test_strat) / len(y_test_strat)}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Regular split
    axes[0,0].bar(range(len(np.bincount(y_train_reg))), np.bincount(y_train_reg))
    axes[0,0].set_title('Regular Split - Training')
    axes[0,1].bar(range(len(np.bincount(y_test_reg))), np.bincount(y_test_reg))
    axes[0,1].set_title('Regular Split - Test')
    
    # Stratified split
    axes[1,0].bar(range(len(np.bincount(y_train_strat))), np.bincount(y_train_strat))
    axes[1,0].set_title('Stratified Split - Training')
    axes[1,1].bar(range(len(np.bincount(y_test_strat))), np.bincount(y_test_strat))
    axes[1,1].set_title('Stratified Split - Test')
    
    plt.tight_layout()
    plt.show()
    
    return (X_train_strat, X_test_strat, y_train_strat, y_test_strat)

# Create imbalanced dataset example
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20, n_redundant=10,
    n_classes=3, weights=[0.7, 0.2, 0.1], random_state=42
)

print(f"Original class distribution: {np.bincount(y_imb)}")
X_train_final, X_test_final, y_train_final, y_test_final = stratified_split_analysis(X_imb, y_imb)
```

### Cross-Validation: Advanced Statistical Theory

Cross-validation represents one of the most important innovations in statistical learning, providing a principled approach to the bias-variance trade-off in performance estimation.

**Theoretical Foundation**

Cross-validation addresses the fundamental problem that using the same data for both training and evaluation leads to optimistically biased performance estimates. The CV estimator:

**CV_k = (1/k) Σ_{i=1}^k L(f^{-i}, D_i)**

Where f^{-i} is trained on all data except fold i, and D_i is the i-th fold, provides a nearly unbiased estimate of generalization error.

**Statistical Properties of Different CV Strategies**

**K-Fold Cross-Validation**
- **Bias**: E[CV_k] ≈ E[R(f)] when k is large
- **Variance**: Var[CV_k] increases with k due to overlapping training sets  
- **Optimal k**: Often k=5 or k=10 balances bias and variance

**Leave-One-Out (LOO) Cross-Validation**
- **Bias**: Minimal (uses n-1 samples for training)
- **Variance**: High due to maximum overlap between training sets
- **Computational**: O(n) times more expensive than single split
- **Special Property**: For certain algorithms (like KNN), LOO has analytical solutions

**Mathematical Analysis of CV Bias and Variance**

The bias-variance decomposition of k-fold CV shows:

**MSE[CV_k] = Bias²[CV_k] + Variance[CV_k]**

Where:
- **Bias²[CV_k] ≈ (1/k) × (training_set_size_penalty)**  
- **Variance[CV_k] ≈ (k-1)/k × (correlation_between_folds)**

This explains why k=5 often provides the best bias-variance trade-off.

**Nested Cross-Validation for Hyperparameter Selection**

When hyperparameter tuning is involved, we need nested CV to avoid selection bias:

**Outer CV**: Estimates generalization performance
**Inner CV**: Selects optimal hyperparameters

Without this nesting, performance estimates are optimistically biased because hyperparameters are chosen to minimize CV error on the test folds.

Cross-validation provides a more robust evaluation by using multiple train-test splits while maintaining rigorous statistical properties.

```python
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, 
    LeaveOneOut, ShuffleSplit
)
from sklearn.ensemble import RandomForestClassifier

def compare_cv_strategies(X, y):
    """Compare different cross-validation strategies"""
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    cv_strategies = {
        'K-Fold (5)': KFold(n_splits=5, shuffle=True, random_state=42),
        'Stratified K-Fold (5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'Leave-One-Out': LeaveOneOut(),
        'Shuffle Split (10)': ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    }
    
    results = {}
    
    print("🔄 Cross-Validation Comparison:")
    print("=" * 50)
    
    for name, cv in cv_strategies.items():
        if name == 'Leave-One-Out' and len(X) > 100:
            print(f"{name}: Skipped (too many samples)")
            continue
            
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        
        print(f"{name}:")
        print(f"  Scores: {scores.round(3)}")
        print(f"  Mean: {scores.mean():.3f} ± {scores.std():.3f}")
        print()
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    
    # Box plot of CV scores
    plt.subplot(1, 2, 1)
    data_to_plot = [scores for scores in results.values()]
    labels = list(results.keys())
    plt.boxplot(data_to_plot, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy Score')
    plt.title('Cross-Validation Score Distribution')
    
    # Mean and std comparison
    plt.subplot(1, 2, 2)
    means = [scores.mean() for scores in results.values()]
    stds = [scores.std() for scores in results.values()]
    
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    plt.xticks(x_pos, labels, rotation=45)
    plt.ylabel('Mean Accuracy ± Std')
    plt.title('Cross-Validation Performance')
    
    plt.tight_layout()
    plt.show()
    
    return results

# Test with iris dataset
cv_results = compare_cv_strategies(X, y)
```

### Time Series Splitting

For time series data, we need to respect temporal order when splitting.

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_split_demo():
    """Demonstrate time series splitting"""
    
    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100  # Random walk
    
    ts_data = pd.DataFrame({
        'date': dates,
        'value': values,
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    plt.figure(figsize=(15, 8))
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(ts_data)):
        plt.subplot(2, 3, i+1)
        
        # Plot training data
        plt.plot(ts_data.iloc[train_idx]['date'], ts_data.iloc[train_idx]['value'], 
                'b-', label='Training', alpha=0.7)
        
        # Plot test data
        plt.plot(ts_data.iloc[test_idx]['date'], ts_data.iloc[test_idx]['value'], 
                'r-', label='Test', alpha=0.7)
        
        plt.title(f'Split {i+1}')
        plt.xticks(rotation=45)
        if i == 0:
            plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5, 
             "Time Series Cross-Validation:\n\n"
             "• Respects temporal order\n"
             "• Training always before test\n"
             "• No data leakage from future\n"
             "• Each split uses more training data",
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return ts_data

ts_example = time_series_split_demo()
```

## Data Validation and Quality Checks

After preprocessing, it's crucial to validate that your data meets the requirements for machine learning.

```python
def validate_processed_data(X_train, X_test, y_train, y_test):
    """Comprehensive data validation"""
    
    print("✅ DATA VALIDATION CHECKLIST")
    print("=" * 40)
    
    # 1. Shape consistency
    print(f"1. Shape Consistency:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    feature_match = X_train.shape[1] == X_test.shape[1]
    sample_match = X_train.shape[0] == len(y_train) and X_test.shape[0] == len(y_test)
    print(f"   ✅ Features match: {feature_match}")
    print(f"   ✅ Samples match: {sample_match}")
    
    # 2. Missing values check
    train_missing = np.isnan(X_train).sum() if isinstance(X_train, np.ndarray) else X_train.isnull().sum().sum()
    test_missing = np.isnan(X_test).sum() if isinstance(X_test, np.ndarray) else X_test.isnull().sum().sum()
    
    print(f"\n2. Missing Values:")
    print(f"   Training set: {train_missing}")
    print(f"   Test set: {test_missing}")
    print(f"   ✅ No missing values: {train_missing == 0 and test_missing == 0}")
    
    # 3. Data type consistency
    if hasattr(X_train, 'dtypes'):
        train_types = set(X_train.dtypes)
        test_types = set(X_test.dtypes)
        type_match = train_types == test_types
        print(f"\n3. Data Types:")
        print(f"   Types consistent: ✅ {type_match}")
    
    # 4. Value ranges
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
    else:
        X_train_df, X_test_df = X_train, X_test
    
    print(f"\n4. Value Ranges:")
    for col in X_train_df.columns:
        train_range = (X_train_df[col].min(), X_train_df[col].max())
        test_range = (X_test_df[col].min(), X_test_df[col].max())
        
        # Check if test range is within training range (approximately)
        reasonable_range = (test_range[0] >= train_range[0] * 0.8 and 
                          test_range[1] <= train_range[1] * 1.2)
        
        print(f"   Column {col}: Train{train_range}, Test{test_range} ✅ {reasonable_range}")
    
    # 5. Class distribution (for classification)
    if len(np.unique(y_train)) < 20:  # Assume classification if few unique values
        print(f"\n5. Class Distribution:")
        train_dist = np.bincount(y_train) / len(y_train)
        test_dist = np.bincount(y_test) / len(y_test)
        
        print(f"   Training: {train_dist.round(3)}")
        print(f"   Test: {test_dist.round(3)}")
        
        # Check if distributions are similar (within 10% difference)
        dist_similar = np.allclose(train_dist, test_dist, atol=0.1)
        print(f"   ✅ Similar distributions: {dist_similar}")
    
    return {
        'shape_consistent': feature_match and sample_match,
        'no_missing': train_missing == 0 and test_missing == 0,
        'ready_for_ml': feature_match and sample_match and train_missing == 0 and test_missing == 0
    }

# Example validation
validation_results = validate_processed_data(X_train_final, X_test_final, 
                                           y_train_final, y_test_final)
print(f"\n🎯 Overall Assessment: {'Ready for ML!' if validation_results['ready_for_ml'] else 'Needs more work'}")
```

## Data Normalization and Standardization: Mathematical Foundations

Many machine learning algorithms are sensitive to the scale of input features, making normalization and standardization critical preprocessing steps. Understanding the mathematical rationale helps us choose the appropriate scaling method.

**The Scale Sensitivity Problem**

Consider a dataset with features of vastly different scales:
- Age: [20, 65] (range ≈ 45)  
- Income: [30000, 200000] (range ≈ 170000)
- Years of education: [12, 20] (range ≈ 8)

Distance-based algorithms (KNN, SVM, clustering) will be dominated by the income feature simply due to its large numerical range, not its predictive importance. This violates the assumption that all features should contribute equally to similarity calculations.

**Mathematical Justification for Scaling**

For algorithms that use Euclidean distance, the distance between two points is:

**d(x₁, x₂) = √(Σᵢ (x₁ᵢ - x₂ᵢ)²)**

Without scaling, features with larger ranges contribute disproportionately to this distance calculation. This creates a mathematical bias toward high-variance features.

**Standardization (Z-score Normalization)**

Standardization transforms features to have zero mean and unit variance:

**z = (x - μ) / σ**

Where:
- **μ** = sample mean = (1/n) Σᵢ xᵢ  
- **σ** = sample standard deviation = √[(1/n) Σᵢ (xᵢ - μ)²]

**Mathematical Properties**:
- **E[z] = 0** (zero mean)
- **Var[z] = 1** (unit variance) 
- **Preserves shape** of original distribution
- **Robust to outliers**: No (outliers affect μ and σ)

**Min-Max Normalization**

Min-max scaling transforms features to a fixed range, typically [0,1]:

**x_norm = (x - min(x)) / (max(x) - min(x))**

**Mathematical Properties**:
- **Range**: [0, 1] (or any specified [a, b])
- **Preserves relationships**: Linear transformation maintains relative distances
- **Outlier sensitivity**: High (min and max are affected by outliers)

**Robust Scaling**

Uses median and interquartile range instead of mean and standard deviation:

**x_robust = (x - median(x)) / IQR(x)**

Where **IQR = Q₃ - Q₁** (75th percentile - 25th percentile)

**Mathematical Properties**:
- **Robust to outliers**: Uses median and IQR  
- **No guarantee of fixed range**: Unlike min-max scaling
- **Preserves distribution shape**: Better than standard scaling for skewed data

**When to Use Each Method**

1. **Standardization**: When features are approximately normally distributed
2. **Min-Max**: When you need bounded ranges (e.g., for neural networks)
3. **Robust**: When data contains outliers or is heavily skewed

## Building Preprocessing Pipelines

Creating reusable preprocessing pipelines ensures consistency and makes your workflow more maintainable.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self, numeric_strategy='mean', categorical_strategy='most_frequent'):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline"""
        
        # Identify column types
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.numeric_strategy)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.categorical_strategy)),
            ('encoder', LabelEncoder())  # Note: In practice, use OneHotEncoder
        ])
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        self.preprocessor.fit(X)
        
        if y is not None:
            self.label_encoder.fit(y)
        
        return self
    
    def transform(self, X, y=None):
        """Transform the data"""
        X_transformed = self.preprocessor.transform(X)
        
        if y is not None:
            y_transformed = self.label_encoder.transform(y)
            return X_transformed, y_transformed
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X, y)
    
    def get_feature_names(self):
        """Get feature names after transformation"""
        return self.preprocessor.get_feature_names_out()

# Example usage
sample_messy_data = pd.DataFrame({
    'age': [25, None, 35, 40, 150],  # Has missing and outlier
    'income': [50000, 60000, None, 80000, 55000],  # Has missing
    'education': ['Bachelor', 'Master', None, 'PhD', 'Bachelor'],  # Has missing
    'target': ['A', 'B', 'A', 'C', 'B']
})

# Separate features and target
X = sample_messy_data.drop('target', axis=1)
y = sample_messy_data['target']

# Apply preprocessing
preprocessor = DataPreprocessor()
X_processed, y_processed = preprocessor.fit_transform(X, y)

print("📊 Preprocessing Results:")
print(f"Original shape: {X.shape}")
print(f"Processed shape: {X_processed.shape}")
print(f"Feature names: {preprocessor.get_feature_names()}")
print(f"Processed data sample:\n{X_processed[:3]}")
```

## Statistical Validity in Preprocessing: Best Practices

### Theoretical Foundations of Best Practices

**1. The Principle of Statistical Independence**
   
All preprocessing decisions must maintain the independence between training and test data. Mathematically, this means:

**P(preprocess | test_data) = P(preprocess | training_data_only)**

Any violation of this principle leads to **data leakage**, where information from the test set influences the preprocessing, creating optimistically biased performance estimates.

**2. Distribution Preservation Principle**

Preprocessing should preserve the essential statistical properties of the data generating process. For any preprocessing function f():

**P(y | f(X)) should approximate P(y | X)**

This ensures that learned relationships remain valid after transformation.

**3. Stationarity Assumption**

Preprocessing parameters (means, variances, encodings) should be stable across train/test splits:

**θ_preprocessing^train ≈ θ_preprocessing^test**

Significant differences suggest non-stationary data or inadequate sampling.

### Statistical Best Practices

**1. Exploratory Data Analysis First**
   - Understand distributional assumptions before choosing preprocessing methods
   - Test for stationarity, normality, and independence
   - Identify the data generating mechanism to inform preprocessing choices

**2. Missing Data Mechanism Analysis**
   - Statistically test for MCAR using Little's test
   - Choose imputation methods based on missing data theory
   - Validate that imputation preserves important relationships

**3. Preprocessing Parameter Estimation**
   - Fit all preprocessing parameters only on training data
   - Use cross-validation to validate preprocessing choices
   - Monitor parameter stability across different train/test splits

**4. Pipeline Validation and Monitoring**
   - Implement statistical tests for preprocessing invariants
   - Monitor distribution drift in production
   - Version preprocessing transformations with data

### Statistical Pitfalls and Their Mathematical Consequences

**1. Data Leakage: A Statistical Violation**

Data leakage occurs when preprocessing uses information that wouldn't be available in practice. Mathematically, this creates:

**P̂(performance | train + test info) > P(performance | train info only)**

The performance estimate becomes optimistically biased because the model indirectly accesses test set information through preprocessing parameters.

**Example**: Fitting a scaler on the entire dataset
- **Wrong**: scaler.fit(X_entire)
- **Mathematical Problem**: Test set statistics influence training set normalization
- **Result**: Performance estimates are inflated by ~5-15%

**2. Preprocessing Inconsistency: Distribution Shift**

When train and test preprocessing differs, we create artificial distribution shift:

**P(X_train_processed) ≠ P(X_test_processed)**

This violates the fundamental assumption that train and test data come from the same distribution.

**3. Over-preprocessing: Information Loss**

Excessive preprocessing can remove signal along with noise. The trade-off is:

**Total Error = Bias² + Variance + Irreducible Error**

Over-preprocessing can reduce variance (noise) but increase bias (signal loss), potentially worsening overall performance.

**4. Statistical Type Errors**

Treating data types incorrectly creates mathematical inconsistencies:

- **Categorical as Numeric**: Creates false ordinal relationships where none exist
- **Ordinal as Nominal**: Loses important ranking information  
- **Continuous as Discrete**: Introduces artificial boundaries in smooth relationships

Each violation changes the mathematical structure that algorithms assume, leading to suboptimal learning.

```python
# Example of data leakage (WRONG way)
def wrong_preprocessing(X, y):
    """Example of what NOT to do"""
    
    # WRONG: Fitting imputer on entire dataset
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)  # Should only fit on training data
    
    # WRONG: Scaling entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)  # Should only fit on training data
    
    # Then splitting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

# Correct way
def correct_preprocessing(X, y):
    """Example of correct preprocessing"""
    
    # CORRECT: Split first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CORRECT: Fit on training data only
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)  # Only transform, not fit
    
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)  # Only transform, not fit
    
    return X_train_scaled, X_test_scaled, y_train, y_test

print("✅ Always remember: Fit on training data, transform on both!")
```

## Key Theoretical and Practical Takeaways

1. **Mathematical Necessity**: Data preprocessing isn't just practical housekeeping—it's mathematically required to satisfy algorithmic assumptions and optimize the learning objective function.

2. **Statistical Missing Data Theory**: Understanding Rubin's missing data mechanisms (MCAR, MAR, MNAR) is crucial for choosing valid imputation strategies and maintaining statistical inference properties.

3. **Generalization Theory**: Proper dataset splitting and cross-validation are grounded in statistical learning theory, specifically the bias-variance decomposition of generalization error.

4. **Scale Invariance**: Feature scaling addresses mathematical biases in distance-based algorithms, ensuring that feature importance is determined by predictive value, not numerical scale.

5. **Statistical Independence**: All preprocessing must maintain independence between training and test data to preserve the validity of performance estimates.

6. **Information Preservation**: The goal is to reduce noise while preserving signal, balancing the bias-variance trade-off inherent in any data transformation.

## What's Next?

In **Chapter 3: Feature Engineering**, you'll learn how to:
- Create new features from existing data
- Select the most informative features
- Apply dimensionality reduction techniques
- Engineer domain-specific features
- Evaluate feature importance

Data preprocessing sets the foundation, but feature engineering is where you can truly unlock the potential hidden in your data!

## Exercises

### Exercise 2.1: Data Quality Assessment
Create a comprehensive data quality report for a messy dataset:
1. Load a real-world dataset (from Kaggle or UCI repository)
2. Identify all data quality issues
3. Create visualizations showing the problems
4. Propose solutions for each issue

### Exercise 2.2: Missing Data Strategies
Compare different missing data handling strategies:
1. Create a dataset with different types of missing data (MCAR, MAR, MNAR)
2. Apply various imputation methods
3. Evaluate the impact on model performance
4. Determine which strategy works best for each scenario

### Exercise 2.3: Preprocessing Pipeline
Build a complete preprocessing pipeline:
1. Handle mixed data types (numeric, categorical, dates)
2. Include outlier detection and handling
3. Implement proper train-test splitting
4. Add data validation checks
5. Make it reusable for new datasets

---

*Data preprocessing might not be glamorous, but it's the foundation upon which all successful machine learning projects are built. Master these skills, and you'll be well-equipped to handle real-world data challenges!*
