# Chapter 5: Regression Algorithms

> "All models are wrong, but some are useful."
> 
> — George E. P. Box

## Learning Objectives

By the end of this chapter, you will be able to:
- **Understand** the mathematical foundations of regression algorithms
- **Implement** linear, polynomial, and regularized regression models
- **Evaluate** regression model performance using appropriate metrics
- **Apply** feature engineering techniques specific to regression problems
- **Handle** real-world regression challenges like overfitting and multicollinearity
- **Build** end-to-end regression pipelines for practical applications

---

## Statistical Foundations of Regression Learning

Regression represents the cornerstone of statistical modeling, seeking to learn the conditional expectation E[Y|X] from observed data. The theoretical framework draws from probability theory, linear algebra, and optimization to provide both predictive power and inferential insights.

**The Regression Learning Problem**

Given training data D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)} where xᵢ ∈ ℝᵈ are feature vectors and yᵢ ∈ ℝ are continuous targets, we seek to learn a function:

**f: ℝᵈ → ℝ**

That minimizes the expected prediction error on future data.

**Bias-Variance Decomposition in Regression**

The expected prediction error can be decomposed into three fundamental components:

**E[(Y - f̂(X))²] = Bias²[f̂(X)] + Var[f̂(X)] + σ²**

Where:
- **Bias²**: Error from incorrect model assumptions
- **Variance**: Error from sensitivity to training data variations  
- **σ²**: Irreducible noise in the data generating process

This decomposition guides algorithm selection and regularization strategies.

**Statistical Assumptions**

Classical regression theory relies on several key assumptions:
1. **Linearity**: Relationship between predictors and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant error variance across all prediction levels
4. **Normality**: Errors follow a normal distribution (for inference)

Understanding when these assumptions hold—and how to address violations—is crucial for effective regression modeling.

### 5.1.1 Types of Regression Problems

#### Prediction vs. Estimation
- **Prediction**: Forecasting future values (stock prices, sales)
- **Estimation**: Understanding relationships (price elasticity, effect sizes)

#### Examples of Regression Applications

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Set style for consistent plotting
plt.style.use('default')
sns.set_palette("husl")

print("Common Regression Applications:")
print("=" * 50)
applications = {
    "Real Estate": "Predicting house prices based on location, size, amenities",
    "Finance": "Stock price forecasting, risk assessment, portfolio optimization", 
    "Marketing": "Sales prediction, customer lifetime value estimation",
    "Healthcare": "Drug dosage optimization, treatment outcome prediction",
    "Engineering": "Quality control, performance optimization, failure prediction",
    "Economics": "GDP forecasting, inflation modeling, market analysis"
}

for domain, description in applications.items():
    print(f"📊 {domain:<12}: {description}")

# Generate sample regression data for visualization
X_sample, y_sample = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

plt.figure(figsize=(12, 4))

# Simple regression example
plt.subplot(1, 3, 1)
plt.scatter(X_sample, y_sample, alpha=0.6)
reg = LinearRegression().fit(X_sample, y_sample)
plt.plot(X_sample, reg.predict(X_sample), color='red', linewidth=2)
plt.title('Linear Regression')
plt.xlabel('Feature')
plt.ylabel('Target')

# Polynomial regression example  
plt.subplot(1, 3, 2)
X_poly = np.linspace(-3, 3, 100).reshape(-1, 1)
y_poly = 2 * X_poly.ravel()**2 + np.random.normal(0, 3, 100)
plt.scatter(X_poly, y_poly, alpha=0.6)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
poly_reg = Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())])
poly_reg.fit(X_poly, y_poly)
plt.plot(X_poly, poly_reg.predict(X_poly), color='red', linewidth=2)
plt.title('Polynomial Regression')
plt.xlabel('Feature')
plt.ylabel('Target')

# Multiple regression visualization
plt.subplot(1, 3, 3) 
X_multi, y_multi = make_regression(n_samples=100, n_features=2, noise=10, random_state=42)
reg_multi = LinearRegression().fit(X_multi, y_multi)
predicted = reg_multi.predict(X_multi)
plt.scatter(y_multi, predicted, alpha=0.6)
plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', linewidth=2)
plt.title('Multiple Regression\n(Actual vs Predicted)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()
```

### 5.1.2 Regression vs. Classification

```python
# Comparison of regression and classification
comparison_data = {
    'Aspect': ['Target Variable', 'Output Type', 'Algorithms', 'Evaluation Metrics', 'Applications'],
    'Regression': [
        'Continuous numerical values',
        'Real numbers (∞ possibilities)', 
        'Linear, Polynomial, Ridge, Lasso',
        'MSE, RMSE, MAE, R²',
        'Price prediction, forecasting'
    ],
    'Classification': [
        'Discrete categories/classes',
        'Limited set of classes',
        'Logistic, SVM, Decision Trees', 
        'Accuracy, Precision, Recall, F1',
        'Spam detection, image recognition'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("Regression vs Classification Comparison:")
print("=" * 60)
for _, row in comparison_df.iterrows():
    print(f"{row['Aspect']:<20}: {row['Regression']:<35} | {row['Classification']}")
```

## Linear Regression: Least Squares Theory and Statistical Inference

Linear regression forms the theoretical backbone of statistical learning, providing both optimal parameter estimation through least squares and a complete probabilistic framework for inference about relationships between variables.

**The Linear Model Framework**

The population linear regression model assumes:

**Y = Xβ + ε**

Where:
- **Y** ∈ ℝⁿ is the response vector
- **X** ∈ ℝⁿˣᵖ is the design matrix (includes intercept column)
- **β** ∈ ℝᵖ is the parameter vector
- **ε** ~ N(0, σ²I) is the error vector (iid Gaussian noise)

### Simple Linear Regression: Univariate Case

For the single-predictor case:

**yᵢ = β₀ + β₁xᵢ + εᵢ, εᵢ ~ N(0, σ²)**

**Least Squares Principle**

The method of least squares minimizes the residual sum of squares:

**RSS(β) = Σᵢ₌₁ⁿ (yᵢ - β₀ - β₁xᵢ)²**

**Analytical Solution via Calculus**

Setting partial derivatives to zero:
- **∂RSS/∂β₀ = 0 ⟹ β̂₀ = ȳ - β̂₁x̄**
- **∂RSS/∂β₁ = 0 ⟹ β̂₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²**

**Statistical Properties of Estimators**

Under the Gauss-Markov conditions, the OLS estimators are:
1. **Unbiased**: E[β̂] = β
2. **Consistent**: β̂ → β as n → ∞  
3. **BLUE**: Best Linear Unbiased Estimators (minimum variance among all linear unbiased estimators)
4. **Asymptotically Normal**: β̂ ~ N(β, σ²(XᵀX)⁻¹) for large n

#### Implementation from Scratch

```python
class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None
        
    def fit(self, X, y):
        """Fit the linear regression model"""
        # Convert to numpy arrays
        X = np.array(X).flatten()
        y = np.array(y).flatten()
        
        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope and intercept
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X).flatten()
        return self.intercept + self.slope * X
    
    def score(self, X, y):
        """Calculate R-squared"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Generate sample data
np.random.seed(42)
X_simple = np.random.randn(100)
y_simple = 2 + 3 * X_simple + np.random.randn(100) * 0.5

# Fit custom implementation
slr_custom = SimpleLinearRegression()
slr_custom.fit(X_simple, y_simple)

# Compare with sklearn
from sklearn.linear_model import LinearRegression
slr_sklearn = LinearRegression()
slr_sklearn.fit(X_simple.reshape(-1, 1), y_simple)

print("Simple Linear Regression Comparison:")
print("=" * 40)
print(f"Custom Implementation:")
print(f"  Slope: {slr_custom.slope:.4f}")
print(f"  Intercept: {slr_custom.intercept:.4f}")
print(f"  R²: {slr_custom.score(X_simple, y_simple):.4f}")

print(f"\nScikit-learn:")
print(f"  Slope: {slr_sklearn.coef_[0]:.4f}")
print(f"  Intercept: {slr_sklearn.intercept_:.4f}")
print(f"  R²: {slr_sklearn.score(X_simple.reshape(-1, 1), y_simple):.4f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_simple, y_simple, alpha=0.6, label='Data Points')
plt.plot(X_simple, slr_custom.predict(X_simple), color='red', linewidth=2, label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_simple - slr_custom.predict(X_simple)
plt.scatter(slr_custom.predict(X_simple), residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Multiple Linear Regression: Matrix Algebra and Optimization

Multiple linear regression extends to multivariate predictors using matrix algebra for elegant mathematical treatment and computational efficiency.

**Matrix Formulation**

The multiple regression model in matrix notation:

**y = Xβ + ε**

Where:
- **y** ∈ ℝⁿ is the response vector
- **X** ∈ ℝⁿˣ⁽ᵖ⁺¹⁾ is the design matrix [1 x₁ x₂ ... xₚ]
- **β** ∈ ℝᵖ⁺¹ is the parameter vector [β₀ β₁ ... βₚ]ᵀ
- **ε** ∈ ℝⁿ is the error vector

**Least Squares Optimization**

Minimizing the quadratic loss function:

**RSS(β) = ||y - Xβ||² = (y - Xβ)ᵀ(y - Xβ)**

**Normal Equations Derivation**

Taking the gradient with respect to β:

**∇_β RSS = -2Xᵀy + 2XᵀXβ = 0**

**Closed-Form Solution**:

**β̂ = (XᵀX)⁻¹Xᵀy**

**Geometric Interpretation**

The OLS solution projects y onto the column space of X:
- **ŷ = Xβ̂ = X(XᵀX)⁻¹Xᵀy = Hy** (H is the "hat" matrix)
- **Residuals**: e = y - ŷ are orthogonal to the column space of X
- **Projection**: ŷ is the closest point in Col(X) to y

**Computational Considerations**

- **Condition Number**: κ(XᵀX) determines numerical stability
- **Rank Deficiency**: When p > n or features are collinear, (XᵀX) is singular
- **Alternative Solutions**: QR decomposition, SVD for numerical stability

#### Real-World Example: Boston Housing Dataset

```python
# Load and explore Boston Housing dataset
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Note: Boston housing dataset is deprecated, using California housing instead
housing = fetch_california_housing()
X_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
y_housing = housing.target

print("California Housing Dataset:")
print("=" * 30)
print(f"Shape: {X_housing.shape}")
print(f"Features: {list(X_housing.columns)}")
print(f"Target: House prices in hundreds of thousands of dollars")

# Display basic statistics
print("\nDataset Statistics:")
print(X_housing.describe())

# Correlation analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
correlation_matrix = X_housing.corrwith(pd.Series(y_housing))
correlation_matrix.plot(kind='bar')
plt.title('Feature Correlation with Price')
plt.ylabel('Correlation')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.scatter(X_housing['MedInc'], y_housing, alpha=0.5)
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.title('Income vs Price')

plt.subplot(1, 3, 3)
plt.scatter(X_housing['AveRooms'], y_housing, alpha=0.5)
plt.xlabel('Average Rooms')
plt.ylabel('House Price')  
plt.title('Rooms vs Price')

plt.tight_layout()
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# Train multiple linear regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Make predictions
y_pred_train = mlr.predict(X_train)
y_pred_test = mlr.predict(X_test)

# Evaluate performance
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train) 
test_r2 = r2_score(y_test, y_pred_test)

print("\nMultiple Linear Regression Results:")
print("=" * 40)
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X_housing.columns,
    'Coefficient': mlr.coef_
})
feature_importance = feature_importance.sort_values('Coefficient', key=abs, ascending=False)

print(f"\nFeature Importance (Coefficients):")
print(feature_importance)

# Visualization of results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Actual vs Predicted\nR² = {test_r2:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
feature_importance.plot(x='Feature', y='Coefficient', kind='bar', ax=plt.gca())
plt.title('Feature Coefficients')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.2.3 Assumptions of Linear Regression

Linear regression makes several important assumptions:

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

#### Checking Assumptions

```python
def check_regression_assumptions(X, y, model, model_name="Linear Regression"):
    """
    Check linear regression assumptions with visualizations
    """
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Linearity check (Actual vs Predicted)
    axes[0,0].scatter(y, y_pred, alpha=0.6)
    axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    axes[0,0].set_xlabel('Actual Values')
    axes[0,0].set_ylabel('Predicted Values')
    axes[0,0].set_title('Linearity Check: Actual vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Homoscedasticity check (Residuals vs Fitted)
    axes[0,1].scatter(y_pred, residuals, alpha=0.6)
    axes[0,1].axhline(y=0, color='red', linestyle='--')
    axes[0,1].set_xlabel('Fitted Values')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Homoscedasticity Check: Residuals vs Fitted')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Normality check (Q-Q plot)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Normality Check: Q-Q Plot')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Residual distribution
    axes[1,1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Residual Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Assumption Checks', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print(f"\n{model_name} - Assumption Test Results:")
    print("=" * 50)
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for large datasets
    print(f"Shapiro-Wilk Test (Normality):")
    print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
    print(f"  Normal residuals: {'Yes' if shapiro_p > 0.05 else 'No'}")
    
    # Breusch-Pagan test for homoscedasticity (simplified)
    mean_residual_sq = np.mean(residuals**2)
    print(f"\nResidual Analysis:")
    print(f"  Mean squared residual: {mean_residual_sq:.4f}")
    print(f"  Standard deviation: {np.std(residuals):.4f}")

# Check assumptions for our housing model
check_regression_assumptions(X_test, y_test, mlr, "Multiple Linear Regression")
```

## 5.3 Polynomial Regression

When the relationship between variables is non-linear, polynomial regression can capture curved patterns by adding polynomial terms.

### 5.3.1 Mathematical Foundation

Polynomial regression extends linear regression by including polynomial terms:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... + \beta_d x^d + \epsilon$$

This is still a **linear model** in terms of the coefficients $\beta_i$, but non-linear in terms of the features.

#### Implementation and Comparison

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve

# Generate non-linear data
np.random.seed(42)
X_poly_demo = np.linspace(-2, 2, 100).reshape(-1, 1)
y_poly_demo = 2 + 3*X_poly_demo.ravel() - 1.5*X_poly_demo.ravel()**2 + 0.5*X_poly_demo.ravel()**3 + np.random.normal(0, 0.5, 100)

# Compare different polynomial degrees
degrees = [1, 2, 3, 4, 6, 8]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

X_plot = np.linspace(-2, 2, 300).reshape(-1, 1)

for i, degree in enumerate(degrees):
    # Create polynomial pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # Fit model
    poly_pipeline.fit(X_poly_demo, y_poly_demo)
    y_plot = poly_pipeline.predict(X_plot)
    
    # Plot
    axes[i].scatter(X_poly_demo, y_poly_demo, alpha=0.6, label='Data')
    axes[i].plot(X_plot, y_plot, color='red', linewidth=2, label=f'Degree {degree}')
    axes[i].set_title(f'Polynomial Degree {degree}')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('y')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Calculate R²
    r2 = poly_pipeline.score(X_poly_demo, y_poly_demo)
    axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Validation curve to find optimal degree
degrees_range = range(1, 16)
train_scores, val_scores = validation_curve(
    Pipeline([('poly', PolynomialFeatures()), ('linear', LinearRegression())]),
    X_poly_demo, y_poly_demo,
    param_name='poly__degree', param_range=degrees_range,
    cv=5, scoring='neg_mean_squared_error'
)

# Convert to positive MSE
train_mse = -train_scores
val_mse = -val_scores

plt.figure(figsize=(10, 6))
plt.plot(degrees_range, np.mean(train_mse, axis=1), 'o-', color='blue', label='Training MSE')
plt.plot(degrees_range, np.mean(val_mse, axis=1), 'o-', color='red', label='Validation MSE')
plt.fill_between(degrees_range, 
                np.mean(train_mse, axis=1) - np.std(train_mse, axis=1),
                np.mean(train_mse, axis=1) + np.std(train_mse, axis=1),
                color='blue', alpha=0.1)
plt.fill_between(degrees_range,
                np.mean(val_mse, axis=1) - np.std(val_mse, axis=1), 
                np.mean(val_mse, axis=1) + np.std(val_mse, axis=1),
                color='red', alpha=0.1)

plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Tradeoff in Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

optimal_degree = degrees_range[np.argmin(np.mean(val_mse, axis=1))]
print(f"Optimal polynomial degree: {optimal_degree}")
```

## 5.4 Regularized Regression

Regularization prevents overfitting by adding a penalty term to the cost function, constraining the model complexity.

### Ridge Regression: Regularization Theory and Bias-Variance Trade-off

Ridge regression addresses the fundamental challenge of overfitting by introducing a bias-variance trade-off through L2 regularization, providing both theoretical guarantees and practical benefits for high-dimensional problems.

**Regularized Optimization Problem**

Ridge regression modifies the OLS objective with an L2 penalty term:

**minimize_β ||y - Xβ||² + λ||β||²**

Where:
- **λ > 0** is the regularization parameter
- **||β||² = Σⱼ βⱼ²** is the L2 norm of coefficients

**Closed-Form Solution**

The regularized normal equations yield:

**β̂_ridge = (X^T X + λI)^{-1} X^T y**

**Key Mathematical Properties**:

1. **Always Invertible**: X^T X + λI is always positive definite for λ > 0
2. **Shrinkage**: Ridge shrinks coefficients toward zero (but not exactly zero)
3. **Continuous**: Small changes in λ produce smooth changes in β̂

**Bayesian Interpretation**

Ridge regression corresponds to MAP estimation with Gaussian priors:

**β ~ N(0, σ²/λ I)**

The regularization parameter λ = σ²/τ² where τ² is the prior variance.

**Bias-Variance Decomposition**

Ridge regression introduces bias to reduce variance:
- **Bias increases**: E[β̂_ridge] ≠ β (shrinkage introduces bias)
- **Variance decreases**: Var[β̂_ridge] < Var[β̂_OLS] (regularization reduces variance)
- **MSE Optimum**: Optimal λ minimizes Bias² + Variance

**Effective Degrees of Freedom**

Ridge regression's model complexity can be quantified as:

**df(λ) = trace(X(X^T X + λI)^{-1} X^T) = Σᵢ σᵢ²/(σᵢ² + λ)**

Where σᵢ² are eigenvalues of X^T X. As λ → 0, df → p (OLS); as λ → ∞, df → 0.

```python
from sklearn.linear_model import Ridge, RidgeCV

# Ridge regression implementation and comparison
ridge_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# Split housing data for regularization demo
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_housing, y_housing, test_size=0.3, random_state=42
)

# Standardize features (important for regularization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled = scaler.transform(X_test_reg)

ridge_results = {}

print("Ridge Regression Results:")
print("=" * 50)

for alpha in ridge_alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train_reg)
    
    train_score = ridge.score(X_train_scaled, y_train_reg)
    test_score = ridge.score(X_test_scaled, y_test_reg)
    
    ridge_results[alpha] = {
        'train_r2': train_score,
        'test_r2': test_score,
        'coefficients': ridge.coef_
    }
    
    print(f"Alpha {alpha:6.3f}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

# Cross-validation to find optimal alpha
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train_reg)

print(f"\nOptimal Alpha (CV): {ridge_cv.alpha_}")
print(f"CV Score: {ridge_cv.score(X_test_scaled, y_test_reg):.4f}")

# Visualize coefficient paths
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i, feature in enumerate(X_housing.columns):
    coef_path = [ridge_results[alpha]['coefficients'][i] for alpha in ridge_alphas]
    plt.plot(ridge_alphas, coef_path, 'o-', label=feature)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Coefficient Paths')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
train_scores = [ridge_results[alpha]['train_r2'] for alpha in ridge_alphas]
test_scores = [ridge_results[alpha]['test_r2'] for alpha in ridge_alphas]
plt.plot(ridge_alphas, train_scores, 'o-', label='Train R²')
plt.plot(ridge_alphas, test_scores, 'o-', label='Test R²')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('Ridge Regression Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Compare coefficients: Linear vs Ridge
linear_coefs = LinearRegression().fit(X_train_scaled, y_train_reg).coef_
ridge_coefs = ridge_cv.coef_

x_pos = np.arange(len(X_housing.columns))
width = 0.35

plt.bar(x_pos - width/2, linear_coefs, width, label='Linear Regression', alpha=0.7)
plt.bar(x_pos + width/2, ridge_coefs, width, label='Ridge Regression', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Comparison')
plt.xticks(x_pos, X_housing.columns, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.4.2 Lasso Regression (L1 Regularization)

Lasso regression uses L1 regularization, which can drive coefficients to exactly zero:

$$J(\boldsymbol{\beta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\beta}}(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^{n}|\beta_j|$$

This enables **automatic feature selection**.

```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso regression analysis
lasso_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

lasso_results = {}
print("\nLasso Regression Results:")
print("=" * 50)

for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=2000)
    lasso.fit(X_train_scaled, y_train_reg)
    
    train_score = lasso.score(X_train_scaled, y_train_reg)
    test_score = lasso.score(X_test_scaled, y_test_reg)
    
    # Count non-zero coefficients
    non_zero_coefs = np.sum(lasso.coef_ != 0)
    
    lasso_results[alpha] = {
        'train_r2': train_score,
        'test_r2': test_score,
        'coefficients': lasso.coef_,
        'non_zero_coefs': non_zero_coefs
    }
    
    print(f"Alpha {alpha:6.3f}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}, Features: {non_zero_coefs}")

# Cross-validation for optimal alpha
lasso_cv = LassoCV(alphas=lasso_alphas, cv=5, max_iter=2000)
lasso_cv.fit(X_train_scaled, y_train_reg)

print(f"\nOptimal Alpha (CV): {lasso_cv.alpha_:.4f}")
print(f"CV Score: {lasso_cv.score(X_test_scaled, y_test_reg):.4f}")

# Feature selection analysis
selected_features = X_housing.columns[lasso_cv.coef_ != 0]
print(f"\nSelected Features by Lasso: {len(selected_features)} out of {len(X_housing.columns)}")
for feature, coef in zip(X_housing.columns, lasso_cv.coef_):
    if coef != 0:
        print(f"  {feature}: {coef:.4f}")

# Visualize Lasso paths
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for i, feature in enumerate(X_housing.columns):
    coef_path = [lasso_results[alpha]['coefficients'][i] for alpha in lasso_alphas]
    plt.plot(lasso_alphas, coef_path, 'o-', label=feature)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Coefficient Paths')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
train_scores_lasso = [lasso_results[alpha]['train_r2'] for alpha in lasso_alphas]
test_scores_lasso = [lasso_results[alpha]['test_r2'] for alpha in lasso_alphas]
plt.plot(lasso_alphas, train_scores_lasso, 'o-', label='Train R²')
plt.plot(lasso_alphas, test_scores_lasso, 'o-', label='Test R²')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('Lasso Regression Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
num_features = [lasso_results[alpha]['non_zero_coefs'] for alpha in lasso_alphas]
plt.plot(lasso_alphas, num_features, 'o-', color='green')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Number of Selected Features')
plt.title('Feature Selection by Lasso')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.4.3 Elastic Net Regression

Elastic Net combines both L1 and L2 regularization:

$$J(\boldsymbol{\beta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\beta}}(x^{(i)}) - y^{(i)})^2 + \lambda_1\sum_{j=1}^{n}|\beta_j| + \lambda_2\sum_{j=1}^{n}\beta_j^2$$

This provides a balance between Ridge's stability and Lasso's feature selection.

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Elastic Net with different l1_ratio values
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]  # 0 = Ridge, 1 = Lasso
elastic_results = {}

print("Elastic Net Results:")
print("=" * 40)

for l1_ratio in l1_ratios:
    elastic_cv = ElasticNetCV(l1_ratio=l1_ratio, cv=5, max_iter=2000)
    elastic_cv.fit(X_train_scaled, y_train_reg)
    
    test_score = elastic_cv.score(X_test_scaled, y_test_reg)
    non_zero_coefs = np.sum(elastic_cv.coef_ != 0)
    
    elastic_results[l1_ratio] = {
        'test_r2': test_score,
        'alpha': elastic_cv.alpha_,
        'non_zero_coefs': non_zero_coefs
    }
    
    print(f"L1 ratio {l1_ratio:.1f}: R² = {test_score:.4f}, Alpha = {elastic_cv.alpha_:.4f}, Features = {non_zero_coefs}")

# Find best l1_ratio
best_l1_ratio = max(elastic_results.keys(), key=lambda k: elastic_results[k]['test_r2'])
print(f"\nBest L1 ratio: {best_l1_ratio} (R² = {elastic_results[best_l1_ratio]['test_r2']:.4f})")
```

## 5.5 Model Evaluation for Regression

Proper evaluation is crucial for understanding regression model performance and comparing different approaches.

### Statistical Theory of Regression Evaluation Metrics

Regression evaluation metrics quantify different aspects of prediction quality, each with specific mathematical properties and interpretations rooted in statistical theory.

**Loss Function Perspective**

Different metrics correspond to different loss functions being optimized:

**Mean Absolute Error (L1 Loss)**
**MAE = 1/n Σᵢ|yᵢ - ŷᵢ|**

- **Robust to outliers**: Linear growth with error magnitude
- **Median minimizer**: Optimal predictor is conditional median
- **Non-differentiable**: Requires subgradient methods for optimization

**Mean Squared Error (L2 Loss)**
**MSE = 1/n Σᵢ(yᵢ - ŷᵢ)²**

- **Sensitive to outliers**: Quadratic growth amplifies large errors
- **Mean minimizer**: Optimal predictor is conditional mean E[Y|X]
- **Differentiable**: Enables gradient-based optimization (OLS)

**Root Mean Squared Error**
**RMSE = √MSE**

- **Same units as target**: Interpretable in original scale
- **Penalty structure**: Same as MSE but different scale

**R-squared: Coefficient of Determination**

**R² = 1 - SSres/SStot = 1 - Σ(yᵢ - ŷᵢ)²/Σ(yᵢ - ȳ)²**

**Statistical Interpretation**:
- **Proportion of variance explained** by the model
- **Range**: (-∞, 1], where 1 = perfect fit, 0 = no better than mean
- **Baseline comparison**: Compares model against naive mean predictor

**Adjusted R-squared: Complexity Penalty**

**R²adj = 1 - (1 - R²)(n-1)/(n-p-1)**

**Purpose**: Penalizes model complexity to prevent overfitting
- **Decreases** when adding irrelevant features (even if R² increases)
- **Model selection**: Favors parsimonious models
- **Degrees of freedom**: Accounts for parameters used

**Information-Theoretic Metrics**

**Akaike Information Criterion (AIC)**
**AIC = 2p - 2ln(L) ≈ n ln(MSE) + 2p**

**Bayesian Information Criterion (BIC)**  
**BIC = p ln(n) - 2ln(L) ≈ n ln(MSE) + p ln(n)**

Both penalize complexity but BIC more heavily for large n.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def comprehensive_regression_evaluation(models_dict, X_train, X_test, y_train, y_test):
    """
    Comprehensive evaluation of multiple regression models
    """
    results = {}
    
    print("Comprehensive Model Evaluation:")
    print("=" * 80)
    print(f"{'Model':<20} {'MAE':<8} {'MSE':<8} {'RMSE':<8} {'R²':<8} {'Adj R²':<8}")
    print("-" * 80)
    
    for name, model in models_dict.items():
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)
        
        # Adjusted R²
        n = len(y_test)
        p = X_test.shape[1] if hasattr(X_test, 'shape') else len(X_test[0])
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        results[name] = {
            'MAE': mae,
            'MSE': mse, 
            'RMSE': rmse,
            'R²': r2,
            'Adjusted R²': adj_r2,
            'predictions': y_pred_test
        }
        
        print(f"{name:<20} {mae:<8.4f} {mse:<8.4f} {rmse:<8.4f} {r2:<8.4f} {adj_r2:<8.4f}")
    
    return results

# Prepare models for comparison
models_comparison = {
    'Linear Regression': LinearRegression().fit(X_train_scaled, y_train_reg),
    'Ridge (CV)': ridge_cv,
    'Lasso (CV)': lasso_cv,
    'Elastic Net': ElasticNetCV(cv=5, max_iter=2000).fit(X_train_scaled, y_train_reg),
    'Polynomial (deg=2)': Pipeline([
        ('poly', PolynomialFeatures(2)),
        ('linear', LinearRegression())
    ]).fit(X_train_scaled, y_train_reg)
}

# Evaluate all models
evaluation_results = comprehensive_regression_evaluation(
    models_comparison, X_train_scaled, X_test_scaled, y_train_reg, y_test_reg
)

# Visualize model performance
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Metrics comparison
metrics = ['MAE', 'MSE', 'RMSE', 'R²']
for i, metric in enumerate(metrics):
    ax = axes[i//2, i%3]
    values = [evaluation_results[model][metric] for model in models_comparison.keys()]
    bars = ax.bar(range(len(values)), values)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(models_comparison.keys(), rotation=45)
    
    # Add value labels on bars
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Actual vs Predicted for best model
best_model_name = max(evaluation_results.keys(), key=lambda k: evaluation_results[k]['R²'])
ax = axes[1, 2]
best_predictions = evaluation_results[best_model_name]['predictions']
ax.scatter(y_test_reg, best_predictions, alpha=0.6)
ax.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', linewidth=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title(f'Best Model: {best_model_name}\nR² = {evaluation_results[best_model_name]["R²"]:.4f}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nBest performing model: {best_model_name}")
```

### 5.5.2 Cross-Validation for Regression

```python
from sklearn.model_selection import cross_val_score, KFold

def regression_cross_validation(models_dict, X, y, cv_folds=5):
    """
    Perform cross-validation for regression models
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_results = {}
    
    print(f"{cv_folds}-Fold Cross-Validation Results:")
    print("=" * 60)
    print(f"{'Model':<20} {'Mean R²':<10} {'Std R²':<10} {'Mean RMSE':<12} {'Std RMSE':<10}")
    print("-" * 60)
    
    for name, model in models_dict.items():
        # R² scores
        r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
        
        # RMSE scores (note: sklearn returns negative MSE, so we need to convert)
        neg_mse_scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        cv_results[name] = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std()
        }
        
        print(f"{name:<20} {r2_scores.mean():<10.4f} {r2_scores.std():<10.4f} "
              f"{rmse_scores.mean():<12.4f} {rmse_scores.std():<10.4f}")
    
    return cv_results

# Perform cross-validation
cv_results = regression_cross_validation(models_comparison, X_train_scaled, y_train_reg)

# Visualize CV results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# R² comparison with error bars
model_names = list(cv_results.keys())
r2_means = [cv_results[name]['r2_mean'] for name in model_names]
r2_stds = [cv_results[name]['r2_std'] for name in model_names]

ax1.bar(range(len(model_names)), r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
ax1.set_title('Cross-Validation R² Scores')
ax1.set_ylabel('R² Score')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45)
ax1.grid(True, alpha=0.3)

# RMSE comparison with error bars
rmse_means = [cv_results[name]['rmse_mean'] for name in model_names]
rmse_stds = [cv_results[name]['rmse_std'] for name in model_names]

ax2.bar(range(len(model_names)), rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='orange')
ax2.set_title('Cross-Validation RMSE Scores')
ax2.set_ylabel('RMSE')
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.5.3 Learning Curves and Model Diagnosis

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title="Learning Curve"):
    """
    Plot learning curve to diagnose bias/variance
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     color='blue', alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     color='red', alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('R² Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Diagnosis
    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.1:
        print("⚠️  High variance (overfitting) detected")
        print("   Consider: more data, regularization, simpler model")
    elif val_mean[-1] < 0.6:
        print("⚠️  High bias (underfitting) detected") 
        print("   Consider: more complex model, more features")
    else:
        print("✅ Model appears well-fitted")

# Plot learning curves for different models
for name, model in [('Linear Regression', LinearRegression()), 
                   ('Ridge Regression', Ridge(alpha=1.0)),
                   ('Lasso Regression', Lasso(alpha=0.1))]:
    print(f"\nLearning Curve Analysis: {name}")
    plot_learning_curve(model, X_train_scaled, y_train_reg, f"Learning Curve - {name}")
```
## 5.6 Practical Applications and Case Studies

### 5.6.1 Case Study: Sales Forecasting

Let's apply regression techniques to a real-world sales forecasting problem.

```python
# Create a realistic sales dataset
np.random.seed(42)
n_samples = 1000

# Generate features
advertising_spend = np.random.exponential(10, n_samples)
season = np.random.randint(1, 5, n_samples)  # 1-4 representing quarters
price = np.random.normal(100, 20, n_samples)
competitor_price = price + np.random.normal(0, 10, n_samples)
economic_index = np.random.normal(100, 15, n_samples)

# Generate target with realistic relationships
sales = (
    50 +  # Base sales
    2.5 * advertising_spend +  # Advertising effect
    np.where(season == 4, 20, 0) +  # Holiday season boost
    -0.8 * price +  # Price sensitivity
    0.3 * competitor_price +  # Competitor effect
    0.1 * economic_index +  # Economic conditions
    np.random.normal(0, 10, n_samples)  # Noise
)

# Create DataFrame
sales_data = pd.DataFrame({
    'advertising_spend': advertising_spend,
    'season': season,
    'price': price,
    'competitor_price': competitor_price,
    'economic_index': economic_index,
    'sales': sales
})

print("Sales Forecasting Dataset:")
print("=" * 30)
print(sales_data.describe())

# Feature engineering
sales_data['price_difference'] = sales_data['competitor_price'] - sales_data['price']
sales_data['advertising_per_price'] = sales_data['advertising_spend'] / sales_data['price']

# One-hot encode season
sales_encoded = pd.get_dummies(sales_data, columns=['season'], prefix='season')

# Prepare features and target
X_sales = sales_encoded.drop('sales', axis=1)
y_sales = sales_encoded['sales']

# Split data
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42
)

# Scale features
scaler_sales = StandardScaler()
X_train_sales_scaled = scaler_sales.fit_transform(X_train_sales)
X_test_sales_scaled = scaler_sales.transform(X_test_sales)

# Apply different regression models
sales_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Polynomial (degree=2)': Pipeline([
        ('poly', PolynomialFeatures(2, interaction_only=True)),
        ('linear', LinearRegression())
    ])
}

sales_results = {}

print("\nSales Forecasting Model Comparison:")
print("=" * 50)

for name, model in sales_models.items():
    # Fit model
    if name == 'Polynomial (degree=2)':
        model.fit(X_train_sales, y_train_sales)  # Don't scale for polynomial
        y_pred = model.predict(X_test_sales)
        score = r2_score(y_test_sales, y_pred)
    else:
        model.fit(X_train_sales_scaled, y_train_sales)
        y_pred = model.predict(X_test_sales_scaled)
        score = model.score(X_test_sales_scaled, y_test_sales)
    
    mae = mean_absolute_error(y_test_sales, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_sales, y_pred))
    
    sales_results[name] = {
        'R²': score,
        'MAE': mae,
        'RMSE': rmse,
        'predictions': y_pred
    }
    
    print(f"{name:<25}: R² = {score:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# Business insights from best model
best_sales_model = max(sales_results.keys(), key=lambda k: sales_results[k]['R²'])
print(f"\nBest model: {best_sales_model}")

# Feature importance analysis (for linear model)
if best_sales_model == 'Linear Regression':
    lr_sales = LinearRegression().fit(X_train_sales_scaled, y_train_sales)
    feature_importance = pd.DataFrame({
        'Feature': X_sales.columns,
        'Coefficient': lr_sales.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\nFeature Importance (Business Insights):")
    print("-" * 40)
    for _, row in feature_importance.iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"• {row['Feature']:<20}: {direction} sales by {abs(row['Coefficient']):.2f} units per unit change")

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
# Model performance comparison
models_list = list(sales_results.keys())
r2_scores = [sales_results[model]['R²'] for model in models_list]
plt.bar(range(len(models_list)), r2_scores)
plt.title('Model Performance Comparison')
plt.ylabel('R² Score')
plt.xticks(range(len(models_list)), models_list, rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
# Best model predictions
best_pred = sales_results[best_sales_model]['predictions']
plt.scatter(y_test_sales, best_pred, alpha=0.6)
plt.plot([y_test_sales.min(), y_test_sales.max()], [y_test_sales.min(), y_test_sales.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title(f'{best_sales_model}\nActual vs Predicted')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Residual analysis
residuals = y_test_sales - best_pred
plt.scatter(best_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.6.2 Model Deployment Considerations

```python
# Model persistence and deployment pipeline
import joblib
from datetime import datetime

class SalesForecaster:
    """
    Production-ready sales forecasting model
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_version = None
        
    def train(self, X_train, y_train, model_type='ridge'):
        """Train the forecasting model"""
        self.feature_names = X_train.columns.tolist()
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Select and train model
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        else:
            self.model = LinearRegression()
            
        self.model.fit(X_train_scaled, y_train)
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Model trained successfully (version: {self.model_version})")
        
    def predict(self, X_new):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Ensure feature consistency
        X_new = X_new[self.feature_names]
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Make prediction
        predictions = self.model.predict(X_new_scaled)
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance for business insights"""
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.model.coef_)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': self.model_version
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_version = model_data['version']
        print(f"Model loaded (version: {self.model_version})")

# Example usage
forecaster = SalesForecaster()
forecaster.train(X_train_sales, y_train_sales, model_type='ridge')

# Make predictions on new data
sample_data = X_test_sales.head(5)
predictions = forecaster.predict(sample_data)

print("\nSample Predictions:")
print("=" * 30)
for i, (idx, row) in enumerate(sample_data.iterrows()):
    print(f"Sample {i+1}: Predicted sales = ${predictions[i]:.2f}")
    print(f"  Advertising: ${row['advertising_spend']:.2f}")
    print(f"  Price: ${row['price']:.2f}")
    print()

# Save model for deployment
# forecaster.save_model('sales_forecaster_model.pkl')
```

## 5.7 Best Practices and Guidelines

### 5.7.1 Regression Development Checklist

```python
def regression_best_practices_checklist():
    """
    Comprehensive checklist for regression projects
    """
    checklist = {
        "Data Preparation": [
            "☐ Check for missing values and handle appropriately",
            "☐ Identify and handle outliers",
            "☐ Examine feature distributions and transform if needed", 
            "☐ Create meaningful feature interactions",
            "☐ Scale/normalize features for regularized models",
            "☐ Split data into train/validation/test sets"
        ],
        "Model Selection": [
            "☐ Start with simple linear regression baseline",
            "☐ Try regularized models (Ridge/Lasso) if overfitting",
            "☐ Consider polynomial features for non-linear relationships",
            "☐ Use cross-validation for hyperparameter tuning",
            "☐ Compare multiple algorithms systematically"
        ],
        "Evaluation": [
            "☐ Use appropriate metrics (R², MAE, RMSE)",
            "☐ Check model assumptions (linearity, homoscedasticity)",
            "☐ Analyze residuals for patterns", 
            "☐ Perform cross-validation for robust estimates",
            "☐ Test on truly unseen data"
        ],
        "Deployment": [
            "☐ Document model assumptions and limitations",
            "☐ Implement prediction confidence intervals",
            "☐ Set up model monitoring and retraining pipeline",
            "☐ Consider business constraints and interpretability needs",
            "☐ Plan for model updates as data changes"
        ]
    }
    
    print("Regression Best Practices Checklist:")
    print("=" * 50)
    
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

regression_best_practices_checklist()
```

### 5.7.2 Common Pitfalls and Solutions

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Data Leakage** | Using future information to predict past | Ensure temporal ordering, proper train/test splits |
| **Overfitting** | Model too complex for available data | Use regularization, cross-validation, more data |
| **Multicollinearity** | Highly correlated features | Use VIF analysis, Ridge regression, PCA |
| **Non-linearity** | Linear model for non-linear relationships | Polynomial features, non-linear models |
| **Heteroscedasticity** | Non-constant error variance | Transform target variable, robust regression |
| **Outliers** | Extreme values affecting model | Robust regression, outlier detection/removal |

## Theoretical and Practical Synthesis of Regression Learning

**1. Statistical Learning Foundation**: Regression seeks to learn E[Y|X] through least squares optimization, providing both predictive power and statistical inference capabilities under appropriate distributional assumptions.

**2. Matrix Algebra and Optimization**: The normal equations β̂ = (XᵀX)⁻¹Xᵀy provide the closed-form OLS solution, with geometric interpretation as projection onto the column space of X.

**3. Bias-Variance Trade-off Principles**: 
   - **OLS**: Unbiased but high variance in high dimensions
   - **Ridge**: Introduces bias to reduce variance via L2 regularization
   - **Optimal λ**: Minimizes total MSE through bias-variance balance

**4. Statistical Properties of Estimators**: Under Gauss-Markov conditions, OLS estimators are BLUE (Best Linear Unbiased Estimators) with well-defined asymptotic distributions for inference.

**5. Regularization Theory**: Ridge regression β̂ = (XᵀX + λI)⁻¹Xᵀy ensures invertibility and provides shrinkage with Bayesian interpretation as MAP estimation with Gaussian priors.

**6. Loss Function Perspectives**: Different metrics optimize different objectives:
   - **MSE**: Optimizes conditional mean (L2 loss)
   - **MAE**: Optimizes conditional median (L1 loss)  
   - **R²**: Measures proportion of variance explained

**7. Model Selection Criteria**: AIC and BIC provide principled approaches to model complexity selection through information-theoretic penalization of parameters.

**8. Assumption Validation**: Statistical validity requires checking linearity, independence, homoscedasticity, and normality assumptions through residual analysis and diagnostic tests.

## 5.9 Exercises

### Exercise 5.1: Simple Linear Regression Analysis
Using the advertising dataset:
1. Implement simple linear regression from scratch
2. Compare with scikit-learn implementation
3. Analyze residuals and check assumptions
4. Calculate confidence intervals for predictions

### Exercise 5.2: Multiple Regression and Feature Engineering
With the Boston/California housing dataset:
1. Perform exploratory data analysis
2. Create polynomial and interaction features
3. Build multiple regression models
4. Interpret coefficients in business terms

### Exercise 5.3: Regularization Comparison
Using a high-dimensional dataset:
1. Compare Ridge, Lasso, and Elastic Net
2. Use cross-validation for hyperparameter tuning
3. Analyze feature selection by Lasso
4. Evaluate bias-variance tradeoff

### Exercise 5.4: Model Evaluation and Diagnosis
For any regression problem:
1. Implement all regression metrics from scratch
2. Create comprehensive residual analysis
3. Perform cross-validation with multiple metrics
4. Diagnose overfitting/underfitting using learning curves

### Exercise 5.5: End-to-End Regression Project
Choose a real-world regression problem:
1. Data collection and preprocessing
2. Exploratory data analysis and feature engineering
3. Model selection and hyperparameter tuning
4. Evaluation and business interpretation
5. Deployment pipeline design

---

*This completes Chapter 5: Regression Algorithms. You now have a comprehensive understanding of regression techniques, from simple linear regression to advanced regularization methods, along with proper evaluation and deployment practices.*
