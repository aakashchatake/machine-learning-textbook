# Appendix C: Datasets and Resources

This appendix provides a comprehensive collection of datasets, code templates, and resources to support your machine learning journey. From built-in scikit-learn datasets to major public repositories, you'll find everything needed to practice and implement machine learning algorithms.

---

## C.1 Built-in Scikit-learn Datasets

Scikit-learn provides several built-in datasets that are perfect for learning and experimentation. These datasets are small, well-curated, and come preloaded with the library.

### C.1.1 Classification Datasets

#### Iris Dataset
```python
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Convert to DataFrame for easier handling
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y
iris_df['target_name'] = iris_df['target'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

print(f"Dataset shape: {iris_df.shape}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
```

**Key Features:**
- **Size:** 150 samples, 4 features
- **Classes:** 3 (setosa, versicolor, virginica)
- **Use Case:** Multi-class classification, clustering
- **Features:** Sepal length, sepal width, petal length, petal width

#### Wine Dataset
```python
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(wine.target_names)}")
print(f"Class distribution: {pd.Series(y).value_counts().sort_index()}")
```

**Key Features:**
- **Size:** 178 samples, 13 features
- **Classes:** 3 wine cultivars
- **Use Case:** Multi-class classification, feature selection
- **Features:** Chemical analysis results

#### Breast Cancer Dataset
```python
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {cancer.target_names}")
print(f"Class distribution: {pd.Series(y).value_counts()}")
```

**Key Features:**
- **Size:** 569 samples, 30 features
- **Classes:** 2 (malignant, benign)
- **Use Case:** Binary classification, medical diagnosis
- **Features:** Cell nuclei characteristics

### C.1.2 Regression Datasets

#### Boston Housing Dataset
```python
import warnings
warnings.filterwarnings('ignore')  # Deprecated warning

from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target

print(f"Dataset shape: {X.shape}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")
print(f"Feature names: {boston.feature_names}")
```

**Key Features:**
- **Size:** 506 samples, 13 features
- **Target:** Housing prices in $1000s
- **Use Case:** Regression, feature importance analysis
- **Note:** Deprecated due to ethical concerns, use California housing instead

#### California Housing Dataset
```python
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()
X, y = california.data, california.target

print(f"Dataset shape: {X.shape}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")
print(f"Feature names: {california.feature_names}")
```

**Key Features:**
- **Size:** 20,640 samples, 8 features
- **Target:** Housing prices in $100,000s
- **Use Case:** Regression, large dataset handling

#### Diabetes Dataset
```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f"Dataset shape: {X.shape}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")
```

**Key Features:**
- **Size:** 442 samples, 10 features
- **Target:** Diabetes progression measure
- **Use Case:** Regression, medical prediction

### C.1.3 Clustering and Dimensionality Reduction Datasets

#### Digits Dataset
```python
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target

# Visualize first few digits
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'Digit: {digits.target[i]}')
    ax.axis('off')
plt.tight_layout()
```

**Key Features:**
- **Size:** 1,797 samples, 64 features (8x8 pixels)
- **Classes:** 10 digits (0-9)
- **Use Case:** Image classification, clustering, dimensionality reduction

#### Olivetti Faces Dataset
```python
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X, y = faces.data, faces.target

print(f"Dataset shape: {X.shape}")
print(f"Number of people: {len(set(y))}")
```

**Key Features:**
- **Size:** 400 samples, 4,096 features (64x64 pixels)
- **Classes:** 40 different people
- **Use Case:** Face recognition, PCA, clustering

### C.1.4 Synthetic Dataset Generation

```python
from sklearn.datasets import (
    make_classification, make_regression, make_blobs, 
    make_circles, make_moons
)

# Classification dataset
X_class, y_class = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=3, random_state=42
)

# Regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, noise=0.1, random_state=42
)

# Clustering dataset
X_blobs, y_blobs = make_blobs(
    n_samples=300, centers=4, cluster_std=1.0, random_state=42
)

# Non-linear datasets
X_circles, y_circles = make_circles(
    n_samples=1000, noise=0.05, factor=0.6, random_state=42
)

X_moons, y_moons = make_moons(
    n_samples=1000, noise=0.1, random_state=42
)
```

---

## C.2 Public Dataset Repositories

### C.2.1 Kaggle Datasets

Kaggle is one of the most popular platforms for machine learning datasets and competitions.

#### Popular Datasets:
1. **Titanic Dataset** - Binary classification (survival prediction)
2. **House Prices** - Regression (advanced house price prediction)
3. **Iris Species** - Multi-class classification
4. **Netflix Movies and TV Shows** - Data analysis and visualization
5. **COVID-19 Dataset** - Time series analysis and forecasting

#### Accessing Kaggle Datasets:
```python
# Install Kaggle API
# pip install kaggle

# Setup authentication (place kaggle.json in ~/.kaggle/)
import kaggle

# Download a dataset
kaggle.api.dataset_download_files(
    'c/titanic', 
    path='./datasets/raw/', 
    unzip=True
)

# Load downloaded data
import pandas as pd
titanic = pd.read_csv('./datasets/raw/train.csv')
```

### C.2.2 UCI Machine Learning Repository

The UCI ML Repository is a collection of databases, domain theories, and datasets widely used by the machine learning community.

#### Popular UCI Datasets:
```python
import pandas as pd

# Adult Income Dataset (Census Income)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
adult_data = pd.read_csv(url, names=column_names, na_values=' ?')

# Car Evaluation Dataset
car_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
car_columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_data = pd.read_csv(car_url, names=car_columns)

# Heart Disease Dataset
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
heart_data = pd.read_csv(heart_url, names=heart_columns, na_values='?')
```

### C.2.3 Seaborn Built-in Datasets

Seaborn provides easy access to several interesting datasets:

```python
import seaborn as sns

# Load various datasets
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
mpg = sns.load_dataset('mpg')
diamonds = sns.load_dataset('diamonds')

# List all available datasets
print(sns.get_dataset_names())
```

### C.2.4 Government and Open Data Portals

#### United States
- **data.gov** - US government's open data portal
- **Census Bureau** - Demographic and economic data
- **Bureau of Labor Statistics** - Employment and economic indicators

#### International
- **World Bank Open Data** - Global development data
- **WHO Global Health Observatory** - Health statistics
- **UN Data** - United Nations statistical databases

### C.2.5 Specialized Dataset Sources

#### Image Datasets
```python
# CIFAR-10/100 (via TensorFlow/Keras)
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# MNIST (via TensorFlow/Keras)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Fashion-MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

#### Text Datasets
```python
# 20 Newsgroups Dataset
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# Movie Reviews (via NLTK)
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
```

---

## C.3 Data Preprocessing Templates

### C.3.1 Complete Data Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    """
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit(self, X, y=None):
        """Fit preprocessing transformations"""
        # Separate numeric and categorical columns
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.categorical_columns = X.select_dtypes(include=['object']).columns
        
        # Fit imputers
        if len(self.numeric_columns) > 0:
            self.numeric_imputer.fit(X[self.numeric_columns])
            
        if len(self.categorical_columns) > 0:
            self.categorical_imputer.fit(X[self.categorical_columns])
            
            # Fit label encoders for categorical variables
            for col in self.categorical_columns:
                le = LabelEncoder()
                # Handle missing values for label encoder
                non_null_values = X[col].dropna()
                le.fit(non_null_values)
                self.label_encoders[col] = le
        
        return self
    
    def transform(self, X):
        """Apply preprocessing transformations"""
        X_processed = X.copy()
        
        # Handle numeric columns
        if len(self.numeric_columns) > 0:
            X_processed[self.numeric_columns] = self.numeric_imputer.transform(
                X_processed[self.numeric_columns]
            )
            X_processed[self.numeric_columns] = self.scaler.fit_transform(
                X_processed[self.numeric_columns]
            )
        
        # Handle categorical columns
        if len(self.categorical_columns) > 0:
            X_processed[self.categorical_columns] = self.categorical_imputer.transform(
                X_processed[self.categorical_columns]
            )
            
            for col in self.categorical_columns:
                le = self.label_encoders[col]
                # Handle unseen categories
                X_processed[col] = X_processed[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return X_processed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

# Usage example
def preprocess_dataset(df, target_column, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for a dataset
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Preprocess features
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Encode target if categorical
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y_train_encoded = target_encoder.fit_transform(y_train)
        y_test_encoded = target_encoder.transform(y_test)
    else:
        y_train_encoded = y_train
        y_test_encoded = y_test
        target_encoder = None
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'preprocessor': preprocessor,
        'target_encoder': target_encoder
    }
```

### C.3.2 Missing Value Handling Templates

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MissingValueHandler:
    """
    Comprehensive missing value handling strategies
    """
    
    @staticmethod
    def analyze_missing_values(df):
        """Analyze missing value patterns"""
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes
        })
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
        missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
        
        return missing_stats
    
    @staticmethod
    def simple_imputation(df, strategy_numeric='median', strategy_categorical='most_frequent'):
        """Simple imputation strategies"""
        df_imputed = df.copy()
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer_numeric = SimpleImputer(strategy=strategy_numeric)
            df_imputed[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy=strategy_categorical)
            df_imputed[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
        
        return df_imputed
    
    @staticmethod
    def knn_imputation(df, n_neighbors=5):
        """K-Nearest Neighbors imputation"""
        # Encode categorical variables first
        df_encoded = df.copy()
        label_encoders = {}
        
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_encoded),
            columns=df_encoded.columns,
            index=df_encoded.index
        )
        
        # Decode categorical variables
        for col, le in label_encoders.items():
            df_imputed[col] = le.inverse_transform(df_imputed[col].astype(int))
        
        return df_imputed
    
    @staticmethod
    def iterative_imputation(df, random_state=42):
        """Iterative (MICE) imputation"""
        # Similar encoding process as KNN
        df_encoded = df.copy()
        label_encoders = {}
        
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Apply iterative imputation
        imputer = IterativeImputer(random_state=random_state)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df_encoded),
            columns=df_encoded.columns,
            index=df_encoded.index
        )
        
        # Decode categorical variables
        for col, le in label_encoders.items():
            df_imputed[col] = le.inverse_transform(df_imputed[col].astype(int))
        
        return df_imputed
```

### C.3.3 Feature Engineering Templates

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif
)

class FeatureEngineer:
    """
    Feature engineering and selection utilities
    """
    
    @staticmethod
    def create_polynomial_features(X, degree=2, include_bias=False):
        """Create polynomial features"""
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out()
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    @staticmethod
    def create_interaction_features(df, columns):
        """Create interaction features between specified columns"""
        df_interactions = df.copy()
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                interaction_name = f"{col1}_{col2}_interaction"
                df_interactions[interaction_name] = df[col1] * df[col2]
        
        return df_interactions
    
    @staticmethod
    def create_binning_features(df, column, bins=5, strategy='equal_width'):
        """Create binned categorical features from continuous variables"""
        if strategy == 'equal_width':
            df[f"{column}_binned"] = pd.cut(df[column], bins=bins)
        elif strategy == 'equal_frequency':
            df[f"{column}_binned"] = pd.qcut(df[column], q=bins)
        
        return df
    
    @staticmethod
    def select_features_univariate(X, y, k=10, score_func=f_classif):
        """Univariate feature selection"""
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        scores = selector.scores_
        
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': scores,
            'Selected': selector.get_support()
        }).sort_values('Score', ascending=False)
        
        return X_selected, selected_features, feature_scores
    
    @staticmethod
    def scale_features(X, method='standard'):
        """Scale features using different methods"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if method not in scalers:
            raise ValueError(f"Method must be one of: {list(scalers.keys())}")
        
        scaler = scalers[method]
        X_scaled = scaler.fit_transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler
```

---

## C.4 Code Snippets Library

### C.4.1 Data Loading and Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_dataset(file_path, target_column=None):
    """
    Load and perform initial exploration of a dataset
    """
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    print("="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found!")
    
    # Basic statistics
    print("\nNumerical Statistics:")
    print(df.describe())
    
    # Categorical overview
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Variables:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"{col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"  Values: {df[col].unique()}")
    
    # Target variable analysis
    if target_column and target_column in df.columns:
        print(f"\nTarget Variable Analysis ({target_column}):")
        if df[target_column].dtype in ['object', 'category']:
            print(df[target_column].value_counts())
        else:
            print(df[target_column].describe())
    
    return df

def create_correlation_heatmap(df, figsize=(12, 8)):
    """Create correlation heatmap for numerical variables"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("Not enough numerical columns for correlation analysis")
        return
    
    plt.figure(figsize=figsize)
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def plot_target_distribution(df, target_column, plot_type='auto'):
    """Plot target variable distribution"""
    if target_column not in df.columns:
        print(f"Column '{target_column}' not found in dataset")
        return
    
    plt.figure(figsize=(10, 6))
    
    if df[target_column].dtype in ['object', 'category'] or plot_type == 'categorical':
        # Categorical target
        value_counts = df[target_column].value_counts()
        
        plt.subplot(1, 2, 1)
        value_counts.plot(kind='bar')
        plt.title(f'{target_column} Distribution')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        plt.title(f'{target_column} Proportion')
        
    else:
        # Numerical target
        plt.subplot(1, 2, 1)
        plt.hist(df[target_column].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'{target_column} Distribution')
        plt.xlabel(target_column)
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(df[target_column].dropna())
        plt.title(f'{target_column} Box Plot')
        plt.ylabel(target_column)
    
    plt.tight_layout()
    plt.show()
```

### C.4.2 Model Training and Evaluation Templates

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_classification_model(model, X_train, X_test, y_train, y_test, 
                             model_name="Model"):
    """
    Train and evaluate a classification model
    """
    print(f"Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n{model_name} Results:")
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Testing Accuracy: {test_score:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nCross-validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model

def train_regression_model(model, X_train, X_test, y_train, y_test, 
                          model_name="Model"):
    """
    Train and evaluate a regression model
    """
    print(f"Training {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n{model_name} Results:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")
    
    # Residual plot
    residuals = y_test - y_pred_test
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred_test, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return model

def hyperparameter_tuning(model, param_grid, X_train, y_train, 
                         scoring='accuracy', cv=5):
    """
    Perform hyperparameter tuning using GridSearchCV
    """
    print("Performing hyperparameter tuning...")
    
    grid_search = GridSearchCV(
        model, param_grid, scoring=scoring, cv=cv, 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_
```

### C.4.3 Visualization Templates

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def create_feature_distribution_plots(df, columns=None, figsize=(15, 10)):
    """Create distribution plots for numerical features"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, column in enumerate(columns):
        if i < len(axes):
            axes[i].hist(df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{column} Distribution')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Frequency')
    
    # Remove empty subplots
    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def create_categorical_plots(df, target_column, categorical_columns=None):
    """Create plots for categorical variables vs target"""
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
    
    n_cols = 2
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, column in enumerate(categorical_columns):
        if i < len(axes):
            if df[target_column].dtype in ['object', 'category']:
                # Categorical target
                crosstab = pd.crosstab(df[column], df[target_column])
                crosstab.plot(kind='bar', ax=axes[i], stacked=False)
                axes[i].set_title(f'{column} vs {target_column}')
                axes[i].set_xlabel(column)
                axes[i].legend(title=target_column)
            else:
                # Numerical target
                sns.boxplot(data=df, x=column, y=target_column, ax=axes[i])
                axes[i].set_title(f'{column} vs {target_column}')
            
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for i in range(len(categorical_columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def create_interactive_scatter_plot(df, x_column, y_column, color_column=None):
    """Create interactive scatter plot using Plotly"""
    if color_column:
        fig = px.scatter(df, x=x_column, y=y_column, color=color_column,
                        title=f'{y_column} vs {x_column}',
                        hover_data=df.columns)
    else:
        fig = px.scatter(df, x=x_column, y=y_column,
                        title=f'{y_column} vs {x_column}',
                        hover_data=df.columns)
    
    fig.show()

def plot_model_comparison(models_results, metric_name='Accuracy'):
    """Plot comparison of multiple model performances"""
    model_names = list(models_results.keys())
    scores = list(models_results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, scores, alpha=0.8, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.title(f'Model Performance Comparison ({metric_name})')
    plt.ylabel(metric_name)
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.ylim(0, max(scores) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

## C.5 Quick Reference Guides

### C.5.1 Scikit-learn Cheat Sheet

```python
# CLASSIFICATION ALGORITHMS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# REGRESSION ALGORITHMS
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# CLUSTERING ALGORITHMS
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# DIMENSIONALITY REDUCTION
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# MODEL SELECTION AND EVALUATION
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# PREPROCESSING
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, RFE
```

### C.5.2 Common Data Issues and Solutions

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| Missing Values | NaN, NULL values | SimpleImputer, KNNImputer, IterativeImputer |
| Outliers | Extreme values | IQR method, Z-score, Isolation Forest |
| Categorical Data | Text/object columns | LabelEncoder, OneHotEncoder, Target Encoding |
| Feature Scale | Different value ranges | StandardScaler, MinMaxScaler, RobustScaler |
| High Dimensionality | Too many features | PCA, Feature Selection, Regularization |
| Imbalanced Classes | Unequal class distribution | SMOTE, Undersampling, Class weights |
| Multicollinearity | Highly correlated features | VIF analysis, PCA, Ridge regression |

### C.5.3 Model Selection Guidelines

| Problem Type | Recommended Algorithms | Key Considerations |
|-------------|----------------------|-------------------|
| **Binary Classification** | Logistic Regression, SVM, Random Forest | Interpretability vs Performance |
| **Multi-class Classification** | Random Forest, Gradient Boosting, Neural Networks | Handle class imbalance |
| **Regression** | Linear Regression, Random Forest, XGBoost | Linear vs Non-linear relationships |
| **Clustering** | K-Means, DBSCAN, Hierarchical | Number of clusters, cluster shapes |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP | Preserve variance vs visualization |
| **Time Series** | ARIMA, LSTM, Prophet | Seasonality, trend, stationarity |

---

## C.6 Best Practices and Tips

### C.6.1 Data Handling Best Practices

1. **Always backup your raw data** - Keep original datasets unchanged
2. **Document data sources** - Track where data comes from and when it was collected
3. **Validate data integrity** - Check for duplicates, inconsistencies, and errors
4. **Version control datasets** - Use tools like DVC for data versioning
5. **Create reproducible pipelines** - Use random seeds and document preprocessing steps

### C.6.2 Performance Optimization

```python
# Memory optimization techniques
import pandas as pd

def optimize_memory_usage(df):
    """Optimize pandas DataFrame memory usage"""
    start_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    
    end_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Memory usage decreased from {start_memory:.2f} MB to {end_memory:.2f} MB')
    print(f'({100 * (start_memory - end_memory) / start_memory:.1f}% reduction)')
    
    return df
```

### C.6.3 Code Organization Tips

```python
# Project structure template
"""
project/
├── data/
│   ├── raw/           # Original datasets
│   ├── processed/     # Cleaned datasets
│   └── external/      # External data sources
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
│   ├── data/         # Data processing modules
│   ├── features/     # Feature engineering
│   ├── models/       # Model definitions
│   └── utils/        # Utility functions
├── tests/            # Unit tests
├── requirements.txt  # Dependencies
└── README.md         # Project documentation
"""

# Configuration management
import os
from dataclasses import dataclass

@dataclass
class Config:
    """Project configuration"""
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Preprocessing parameters
    NUMERIC_STRATEGY = 'median'
    CATEGORICAL_STRATEGY = 'most_frequent'
```

---

This comprehensive appendix provides you with all the essential datasets, templates, and code snippets needed for your machine learning projects. Use these resources as starting points and adapt them to your specific needs and datasets.

**Remember:** The key to successful machine learning is not just having the right tools, but understanding when and how to use them effectively. Always start with thorough data exploration and choose methods that align with your problem domain and data characteristics.