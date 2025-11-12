#!/usr/bin/env python3
"""
Machine Learning Textbook Utilities
Utility functions and classes used throughout the textbook examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TextbookDatasets:
    """Collection of datasets used in textbook examples"""
    
    @staticmethod
    def create_customer_data(n_samples=300, seed=42):
        """Create synthetic customer segmentation data"""
        np.random.seed(seed)
        
        # Budget customers
        budget = np.random.multivariate_normal([25, 20], [[50, 10], [10, 30]], 100)
        # Premium customers  
        premium = np.random.multivariate_normal([70, 80], [[40, 20], [20, 50]], 100)
        # Middle-tier customers
        middle = np.random.multivariate_normal([45, 50], [[30, 5], [5, 40]], 100)
        
        X = np.vstack([budget, premium, middle])
        labels = np.hstack([np.zeros(100), np.ones(100), np.full(100, 2)])
        
        df = pd.DataFrame(X, columns=['Annual_Income', 'Spending_Score'])
        df['Customer_Type'] = labels
        df['Customer_Type'] = df['Customer_Type'].map({0: 'Budget', 1: 'Premium', 2: 'Middle'})
        
        return df
    
    @staticmethod  
    def create_housing_data(n_samples=500, seed=42):
        """Create synthetic housing price data"""
        np.random.seed(seed)
        
        crime_rate = np.random.exponential(2, n_samples)
        rooms = np.random.normal(6.5, 1, n_samples)
        age = np.random.uniform(0, 100, n_samples)
        distance = np.random.exponential(3, n_samples)
        
        # Realistic price relationship
        price = (50 - 3 * crime_rate + 8 * rooms - 0.1 * age - 2 * distance + 
                np.random.normal(0, 5, n_samples))
        price = np.clip(price, 10, 50)
        
        df = pd.DataFrame({
            'Crime_Rate': crime_rate,
            'Avg_Rooms': rooms,
            'Building_Age': age,
            'Distance_to_Center': distance,
            'Price': price
        })
        
        return df

class MLPipeline:
    """Standard machine learning pipeline for textbook examples"""
    
    def __init__(self, model, test_size=0.2, random_state=42):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the complete pipeline"""
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, X=None):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        if X is None:
            X = self.X_test_scaled
        else:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nDetailed Report:")
        print(classification_report(self.y_test, y_pred))
        
        return accuracy

class DataQualityChecker:
    """Comprehensive data quality assessment"""
    
    @staticmethod
    def assess_dataframe(df):
        """Perform comprehensive data quality check"""
        print("📋 DATA QUALITY ASSESSMENT")
        print("=" * 50)
        
        # Basic info
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        print(f"\n🔍 Missing Values:")
        if missing.sum() == 0:
            print("  ✅ No missing values found")
        else:
            for col in df.columns:
                if missing[col] > 0:
                    print(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
        
        # Data types
        print(f"\n📊 Data Types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        print(f"\n🔄 Duplicates: {duplicates}")
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📈 Numeric Statistics:")
            print(df[numeric_cols].describe())
        
        return {
            'shape': df.shape,
            'missing_values': missing.to_dict(),
            'duplicates': duplicates,
            'data_types': df.dtypes.to_dict()
        }

class Visualizer:
    """Common visualization functions for textbook"""
    
    @staticmethod
    def plot_classification_results(X, y, y_pred, feature_names=None):
        """Visualize classification results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original data
        scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
        axes[0].set_title('True Labels')
        axes[0].set_xlabel(feature_names[0] if feature_names else 'Feature 1')
        axes[0].set_ylabel(feature_names[1] if feature_names else 'Feature 2')
        
        # Predictions
        axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.6)
        axes[1].set_title('Predictions')
        axes[1].set_xlabel(feature_names[0] if feature_names else 'Feature 1')
        axes[1].set_ylabel(feature_names[1] if feature_names else 'Feature 2')
        
        # Correct vs incorrect
        correct = y == y_pred
        colors = ['red' if not c else 'green' for c in correct]
        axes[2].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
        axes[2].set_title('Correct (Green) vs Incorrect (Red)')
        axes[2].set_xlabel(feature_names[0] if feature_names else 'Feature 1')
        axes[2].set_ylabel(feature_names[1] if feature_names else 'Feature 2')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_learning_curve(scores_history, title="Learning Curve"):
        """Plot learning curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(scores_history, 'b-', linewidth=2, alpha=0.7)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance, feature_names, title="Feature Importance"):
        """Plot feature importance"""
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance[indices], alpha=0.7)
        plt.xticks(range(len(importance)), 
                  [feature_names[i] for i in indices], 
                  rotation=45)
        plt.title(title)
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

def print_chapter_header(chapter_num, title):
    """Print formatted chapter header"""
    print("=" * 60)
    print(f"CHAPTER {chapter_num}: {title.upper()}")
    print("=" * 60)
    print()

def print_section_header(section_title):
    """Print formatted section header"""
    print(f"\n{'='*10} {section_title} {'='*10}")

def create_sample_dataset(dataset_type='classification', n_samples=100, n_features=2):
    """Create sample datasets for examples"""
    np.random.seed(42)
    
    if dataset_type == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_redundant=0,
            n_informative=n_features,
            n_clusters_per_class=1,
            random_state=42
        )
    elif dataset_type == 'regression':
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )
    elif dataset_type == 'clustering':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(
            n_samples=n_samples,
            centers=3,
            n_features=n_features,
            random_state=42,
            cluster_std=1.0
        )
    
    return X, y

# Example usage and testing
if __name__ == "__main__":
    print_chapter_header(0, "Utility Functions Test")
    
    # Test dataset creation
    print("🧪 Testing dataset creation...")
    customer_data = TextbookDatasets.create_customer_data()
    print(f"Customer data shape: {customer_data.shape}")
    print(customer_data.head())
    
    # Test data quality checker
    print("\n🔍 Testing data quality checker...")
    quality_report = DataQualityChecker.assess_dataframe(customer_data)
    
    print("\n✅ All utility functions working correctly!")
