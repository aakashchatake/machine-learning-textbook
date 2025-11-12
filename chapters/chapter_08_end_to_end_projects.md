# Chapter 8: The Grand Symphony - End-to-End Machine Learning Projects

## Learning Outcomes: Mastering the Art of ML Orchestration
By the end of this chapter, you will have evolved from a student of algorithms to a **conductor of intelligent systems**:
- Orchestrate complete ML symphonies using CRISP-DM methodology as your musical score
- Design resilient pipelines that gracefully handle the chaos of real-world data
- Transform business whispers into algorithmic solutions that sing with clarity
- Navigate the complex dance between statistical rigor and business pragmatism
- Deploy ML solutions that don't just work in notebooks, but thrive in production storms
- Craft documentation and processes that tell the story of your analytical journey

## Chapter Overview: Where Theory Meets the Beautiful Chaos of Reality

*"In theory, there is no difference between theory and practice. In practice, there is."* — Yogi Berra

Welcome to the most exhilarating chapter of our journey—where the elegant mathematical theories we've mastered meet the wild, unpredictable, and utterly fascinating world of real problems. This is where data scientists are truly born, not in the comfort of clean datasets and perfect algorithms, but in the trenches of missing values, shifting business requirements, and the eternal question: "Will it work on Monday morning?"

### The Art of Real-World ML Alchemy

Imagine you're a master craftsperson in an ancient guild, but instead of forging steel or weaving tapestries, you're creating intelligent systems that solve real human problems. Each project is a masterpiece waiting to be discovered, hidden within the raw materials of data, business needs, and computational constraints.

This chapter is your **apprenticeship in ML craftsmanship**—where we don't just build models, we architect solutions that endure, evolve, and enchant. We'll embark on four extraordinary quests:

🏠 **The Housing Oracle**: Predicting real estate prices with the wisdom of statistical learning  
📊 **The Stock Market Whisperer**: Dancing with financial time series and market psychology  
💼 **The Employee Loyalty Detective**: Solving the mystery of workforce retention  
🛍️ **The Customer Journey Archaeologist**: Uncovering the hidden tribes within your customer base

### The Philosophy of End-to-End Excellence

This isn't just about connecting code blocks—it's about developing the **systems thinking** that separates great data scientists from mere algorithm implementers. You'll learn to see the invisible threads that connect business strategy to mathematical elegance, to anticipate failure modes before they occur, and to build solutions that grow more intelligent over time.

---

## 8.1 The CRISP-DM Methodology: Your North Star in the ML Universe

### 8.1.1 The Sacred Geometry of Data Science

*"A goal without a plan is just a wish. A plan without methodology is just hope."* — Modern Data Science Proverb

In the swirling cosmos of data science, where infinite possibilities exist and every path seems equally valid, CRISP-DM emerges as your **philosophical compass**—not just a methodology, but a way of thinking that has guided thousands of successful ML projects across industries and continents.

Think of CRISP-DM as the **DNA of intelligent problem-solving**. Just as DNA provides the blueprint for life, CRISP-DM provides the genetic code for transforming business questions into algorithmic answers. It's not rigid scaffolding but **adaptive wisdom**—a living framework that breathes with your project's unique rhythms.

### The Six Sacred Phases: A Journey of Transformation

**1. Business Understanding** - *The Art of Asking the Right Questions*  
Where we transform vague hunches into precise, measurable objectives

**2. Data Understanding** - *The Detective Phase*  
Where we become data archaeologists, uncovering stories hidden in numbers

**3. Data Preparation** - *The Alchemical Transformation*  
Where raw data becomes refined intelligence through careful craftsmanship

**4. Modeling** - *The Creative Laboratory*  
Where mathematical theories dance with computational reality

**5. Evaluation** - *The Moment of Truth*  
Where we separate genuine insights from statistical mirages

**6. Deployment** - *The Birth of Intelligence*  
Where algorithms become living systems that serve human needs

### The Philosophy Behind the Process

CRISP-DM isn't just about following steps—it's about **developing intuition** for the natural rhythms of discovery. Like a river that knows its path to the sea, great ML projects have an organic flow that CRISP-DM helps you recognize and honor.

### 8.1.2 Phase 1: Business Understanding

The foundation of any successful ML project is a clear understanding of the business problem.

**Key Activities:**
- Define business objectives
- Assess the situation and resources
- Determine data mining goals
- Produce project plan

**Example Framework:**
```python
class ProjectDefinition:
    def __init__(self):
        self.business_objective = ""
        self.success_criteria = []
        self.constraints = {}
        self.risks = []
        self.timeline = {}
        self.stakeholders = []
    
    def define_problem(self, objective, metrics, constraints=None):
        """Define the core business problem and success metrics"""
        self.business_objective = objective
        self.success_criteria = metrics
        self.constraints = constraints or {}
        
        return {
            'problem_type': self._classify_problem_type(),
            'data_requirements': self._estimate_data_needs(),
            'success_metrics': self.success_criteria
        }
    
    def _classify_problem_type(self):
        """Classify the ML problem type based on business objective"""
        keywords = self.business_objective.lower()
        
        if any(word in keywords for word in ['predict', 'forecast', 'estimate']):
            if any(word in keywords for word in ['category', 'class', 'type']):
                return 'classification'
            else:
                return 'regression'
        elif any(word in keywords for word in ['group', 'segment', 'cluster']):
            return 'clustering'
        elif any(word in keywords for word in ['recommend', 'suggest']):
            return 'recommendation'
        else:
            return 'exploratory'
```

### 8.1.3 Phase 2: Data Understanding

Understanding your data is crucial for project success.

**Data Assessment Checklist:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataProfiler:
    def __init__(self, df):
        self.df = df
        self.profile = {}
    
    def generate_profile(self):
        """Generate comprehensive data profile"""
        self.profile = {
            'basic_info': self._get_basic_info(),
            'missing_data': self._analyze_missing_data(),
            'data_types': self._analyze_data_types(),
            'distributions': self._analyze_distributions(),
            'correlations': self._analyze_correlations(),
            'outliers': self._detect_outliers(),
            'data_quality': self._assess_data_quality()
        }
        return self.profile
    
    def _get_basic_info(self):
        """Get basic dataset information"""
        return {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'column_count': len(self.df.columns),
            'row_count': len(self.df)
        }
    
    def _analyze_missing_data(self):
        """Analyze missing data patterns"""
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        return {
            'missing_counts': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'missing_patterns': self._identify_missing_patterns()
        }
    
    def _identify_missing_patterns(self):
        """Identify patterns in missing data"""
        # Create missing data indicator matrix
        missing_matrix = self.df.isnull()
        
        # Find common missing patterns
        patterns = missing_matrix.value_counts().head(10)
        return patterns.to_dict()
    
    def _analyze_data_types(self):
        """Analyze data types and suggest improvements"""
        type_analysis = {}
        
        for column in self.df.columns:
            col_type = str(self.df[column].dtype)
            unique_count = self.df[column].nunique()
            
            type_analysis[column] = {
                'current_type': col_type,
                'unique_values': unique_count,
                'suggested_type': self._suggest_optimal_type(column),
                'memory_optimization': self._suggest_memory_optimization(column)
            }
        
        return type_analysis
    
    def _suggest_optimal_type(self, column):
        """Suggest optimal data type for a column"""
        col = self.df[column]
        
        if col.dtype == 'object':
            # Check if it's actually numeric
            try:
                pd.to_numeric(col, errors='raise')
                return 'numeric'
            except:
                if col.nunique() < len(col) * 0.05:  # Less than 5% unique
                    return 'category'
                else:
                    return 'string'
        
        elif col.dtype in ['int64', 'float64']:
            # Check if we can downcast
            if col.dtype == 'int64':
                if col.min() >= 0:
                    if col.max() < 256:
                        return 'uint8'
                    elif col.max() < 65536:
                        return 'uint16'
                    else:
                        return 'uint32'
                else:
                    if col.min() >= -128 and col.max() < 128:
                        return 'int8'
                    elif col.min() >= -32768 and col.max() < 32768:
                        return 'int16'
                    else:
                        return 'int32'
            
        return str(col.dtype)  # Keep current type
    
    def visualize_data_quality(self):
        """Create visualizations for data quality assessment"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Missing data heatmap
        sns.heatmap(self.df.isnull(), cbar=True, ax=axes[0,0])
        axes[0,0].set_title('Missing Data Pattern')
        
        # Data type distribution
        type_counts = self.df.dtypes.value_counts()
        axes[0,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Data Type Distribution')
        
        # Correlation matrix
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
            axes[1,0].set_title('Correlation Matrix')
        
        # Outlier detection summary
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                       (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts[col] = outliers
        
        if outlier_counts:
            axes[1,1].bar(outlier_counts.keys(), outlier_counts.values())
            axes[1,1].set_title('Outlier Count by Column')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

# Example usage
def demonstrate_data_profiling():
    """Demonstrate data profiling capabilities"""
    # Generate sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'is_default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    # Introduce missing values
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(df.index, size=100, replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    # Profile the data
    profiler = DataProfiler(df)
    profile = profiler.generate_profile()
    
    print("Data Profile Summary:")
    print(f"Shape: {profile['basic_info']['shape']}")
    print(f"Memory Usage: {profile['basic_info']['memory_usage'] / 1024:.2f} KB")
    print(f"Missing Data: {len(profile['missing_data']['missing_counts'])} columns affected")
    
    return df, profile
```

### 8.1.4 Phase 3: Data Preparation

Data preparation typically consumes 60-80% of project time but is crucial for model success.

**Comprehensive Data Pipeline:**
```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 numeric_strategy='median',
                 categorical_strategy='most_frequent',
                 encoding_strategy='onehot',
                 scaling_strategy='standard',
                 outlier_treatment='iqr',
                 feature_selection=None):
        
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.encoding_strategy = encoding_strategy
        self.scaling_strategy = scaling_strategy
        self.outlier_treatment = outlier_treatment
        self.feature_selection = feature_selection
        
        self.numeric_pipeline = None
        self.categorical_pipeline = None
        self.preprocessor = None
    
    def fit(self, X, y=None):
        """Fit preprocessing pipeline"""
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Build numeric pipeline
        numeric_steps = []
        
        # Outlier treatment
        if self.outlier_treatment == 'iqr':
            numeric_steps.append(('outlier_treatment', OutlierTreatment()))
        
        # Imputation
        if self.numeric_strategy == 'knn':
            numeric_steps.append(('imputer', KNNImputer()))
        else:
            numeric_steps.append(('imputer', SimpleImputer(strategy=self.numeric_strategy)))
        
        # Scaling
        if self.scaling_strategy == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif self.scaling_strategy == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            numeric_steps.append(('scaler', MinMaxScaler()))
        
        self.numeric_pipeline = Pipeline(numeric_steps)
        
        # Build categorical pipeline
        categorical_steps = []
        
        # Imputation
        categorical_steps.append(('imputer', 
                                SimpleImputer(strategy=self.categorical_strategy)))
        
        # Encoding
        if self.encoding_strategy == 'onehot':
            categorical_steps.append(('encoder', 
                                    OneHotEncoder(drop='first', sparse=False)))
        elif self.encoding_strategy == 'label':
            categorical_steps.append(('encoder', LabelEncoder()))
        
        self.categorical_pipeline = Pipeline(categorical_steps)
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('numeric', self.numeric_pipeline, numeric_features),
            ('categorical', self.categorical_pipeline, categorical_features)
        ], remainder='drop')
        
        # Fit the preprocessor
        self.preprocessor.fit(X, y)
        
        return self
    
    def transform(self, X):
        """Transform data using fitted pipeline"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        return self.preprocessor.transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform data"""
        return self.fit(X, y).transform(X)

class OutlierTreatment(BaseEstimator, TransformerMixin):
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.bounds = {}
    
    def fit(self, X, y=None):
        """Calculate outlier bounds for each column"""
        for column in X.columns:
            if self.method == 'iqr':
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                
            elif self.method == 'zscore':
                mean = X[column].mean()
                std = X[column].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
            
            self.bounds[column] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """Apply outlier treatment"""
        X_transformed = X.copy()
        
        for column, (lower_bound, upper_bound) in self.bounds.items():
            X_transformed[column] = np.clip(X_transformed[column], 
                                          lower_bound, upper_bound)
        
        return X_transformed

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 create_interactions=False,
                 create_polynomials=False,
                 polynomial_degree=2,
                 create_ratios=False,
                 create_aggregations=False):
        
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        self.create_ratios = create_ratios
        self.create_aggregations = create_aggregations
        
        self.feature_names = []
        self.numeric_columns = []
    
    def fit(self, X, y=None):
        """Fit feature engineer"""
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = X.columns.tolist()
        return self
    
    def transform(self, X):
        """Create engineered features"""
        X_transformed = X.copy()
        
        # Interaction features
        if self.create_interactions and len(self.numeric_columns) >= 2:
            for i, col1 in enumerate(self.numeric_columns):
                for col2 in self.numeric_columns[i+1:]:
                    interaction_name = f"{col1}_{col2}_interaction"
                    X_transformed[interaction_name] = X[col1] * X[col2]
        
        # Polynomial features
        if self.create_polynomials:
            for col in self.numeric_columns:
                for degree in range(2, self.polynomial_degree + 1):
                    poly_name = f"{col}_poly_{degree}"
                    X_transformed[poly_name] = X[col] ** degree
        
        # Ratio features
        if self.create_ratios and len(self.numeric_columns) >= 2:
            for i, col1 in enumerate(self.numeric_columns):
                for col2 in self.numeric_columns[i+1:]:
                    ratio_name = f"{col1}_{col2}_ratio"
                    # Avoid division by zero
                    X_transformed[ratio_name] = X[col1] / (X[col2] + 1e-8)
        
        return X_transformed
```

### 8.1.5 Phase 4: Modeling

The modeling phase involves selecting appropriate algorithms, training models, and optimizing performance.

**Model Selection Framework:**
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform

class ModelSelector:
    def __init__(self, problem_type='classification', cv_folds=5, scoring=None):
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.scoring = scoring or self._get_default_scoring()
        
        self.models = self._get_base_models()
        self.results = {}
        self.best_model = None
        self.best_params = None
    
    def _get_default_scoring(self):
        """Get default scoring metric based on problem type"""
        if self.problem_type == 'classification':
            return 'roc_auc'
        elif self.problem_type == 'regression':
            return 'r2'
        else:
            return 'accuracy'
    
    def _get_base_models(self):
        """Get base models for comparison"""
        if self.problem_type == 'classification':
            return {
                'logistic_regression': LogisticRegression(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'svm': SVC(random_state=42, probability=True),
                'naive_bayes': GaussianNB(),
                'knn': KNeighborsClassifier(),
                'xgboost': xgb.XGBClassifier(random_state=42),
                'lightgbm': lgb.LGBMClassifier(random_state=42, verbosity=-1)
            }
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.svm import SVR
            
            return {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(random_state=42),
                'lasso_regression': Lasso(random_state=42),
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'svm': SVR(),
                'xgboost': xgb.XGBRegressor(random_state=42),
                'lightgbm': lgb.LGBMRegressor(random_state=42, verbosity=-1)
            }
    
    def compare_models(self, X, y):
        """Compare multiple models using cross-validation"""
        print(f"Comparing models using {self.cv_folds}-fold cross-validation...")
        print(f"Scoring metric: {self.scoring}")
        print("-" * 60)
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y, 
                                       cv=self.cv_folds, 
                                       scoring=self.scoring,
                                       n_jobs=-1)
                
                self.results[name] = {
                    'scores': scores,
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'model': model
                }
                
                print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"{name:20s}: Error - {str(e)}")
        
        # Sort by mean score (descending)
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1]['mean_score'], 
                              reverse=True)
        
        print("-" * 60)
        print(f"Best model: {sorted_results[0][0]} with score: {sorted_results[0][1]['mean_score']:.4f}")
        
        return sorted_results
    
    def hyperparameter_tuning(self, X, y, model_name=None, search_type='random'):
        """Perform hyperparameter tuning for the best model or specified model"""
        if model_name is None:
            # Use the best model from comparison
            if not self.results:
                raise ValueError("No models have been compared yet. Run compare_models first.")
            model_name = max(self.results.keys(), key=lambda k: self.results[k]['mean_score'])
        
        model = self.models[model_name]
        param_grid = self._get_param_grid(model_name)
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, 
                                cv=self.cv_folds, 
                                scoring=self.scoring,
                                n_jobs=-1, 
                                verbose=1)
        else:  # randomized search
            search = RandomizedSearchCV(model, param_grid, 
                                      n_iter=50,
                                      cv=self.cv_folds, 
                                      scoring=self.scoring,
                                      n_jobs=-1, 
                                      verbose=1,
                                      random_state=42)
        
        search.fit(X, y)
        
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        return search
    
    def _get_param_grid(self, model_name):
        """Get parameter grid for hyperparameter tuning"""
        param_grids = {
            'random_forest': {
                'n_estimators': randint(50, 200),
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4)
            },
            'xgboost': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            },
            'lightgbm': {
                'n_estimators': randint(50, 200),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 10),
                'num_leaves': randint(10, 100),
                'subsample': uniform(0.6, 0.4)
            },
            'logistic_regression': {
                'C': uniform(0.001, 10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': uniform(0.1, 10),
                'gamma': ['scale', 'auto'] + [uniform(0.001, 1)],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        }
        
        return param_grids.get(model_name, {})

class ModelEvaluator:
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.evaluation_results = {}
    
    def comprehensive_evaluation(self, model, X_test, y_test, X_train=None, y_train=None):
        """Perform comprehensive model evaluation"""
        results = {}
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        if self.problem_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            results = self._evaluate_classification(y_test, y_pred, y_pred_proba)
        else:
            results = self._evaluate_regression(y_test, y_pred)
        
        # Add training performance if training data provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            results['training_performance'] = self._calculate_training_metrics(y_train, y_train_pred)
            results['overfitting_analysis'] = self._analyze_overfitting(results, y_train, y_train_pred)
        
        self.evaluation_results = results
        return results
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """Evaluate classification model"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, classification_report, 
                                   confusion_matrix, roc_curve, precision_recall_curve)
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            results['roc_curve'] = roc_curve(y_true, y_pred_proba)
            results['precision_recall_curve'] = precision_recall_curve(y_true, y_pred_proba)
        
        return results
    
    def _evaluate_regression(self, y_true, y_pred):
        """Evaluate regression model"""
        from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                                   r2_score, explained_variance_score)
        
        results = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
        
        # Add residual analysis
        residuals = y_true - y_pred
        results['residual_analysis'] = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'residuals': residuals
        }
        
        return results
    
    def visualize_performance(self, model, X_test, y_test):
        """Create performance visualizations"""
        if self.problem_type == 'classification':
            return self._plot_classification_results(model, X_test, y_test)
        else:
            return self._plot_regression_results(model, X_test, y_test)
    
    def _plot_classification_results(self, model, X_test, y_test):
        """Create classification performance plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # ROC Curve
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0,1].set_xlim([0.0, 1.0])
            axes[0,1].set_ylim([0.0, 1.05])
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend(loc="lower right")
        
        # Precision-Recall Curve
        if y_pred_proba is not None:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            axes[1,0].plot(recall, precision, color='blue', lw=2,
                          label=f'PR curve (AP = {avg_precision:.2f})')
            axes[1,0].set_xlabel('Recall')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].set_title('Precision-Recall Curve')
            axes[1,0].legend()
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[1,1].barh(importance_df['feature'], importance_df['importance'])
            axes[1,1].set_title('Top 10 Feature Importances')
            axes[1,1].set_xlabel('Importance')
        
        plt.tight_layout()
        return fig
    
    def _plot_regression_results(self, model, X_test, y_test):
        """Create regression performance plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Actual vs Predicted
        axes[0,0].scatter(y_test, y_pred, alpha=0.5)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Values')
        axes[0,0].set_ylabel('Predicted Values')
        axes[0,0].set_title('Actual vs Predicted Values')
        
        # Residuals vs Predicted
        axes[0,1].scatter(y_pred, residuals, alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Values')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs Predicted Values')
        
        # Residuals Distribution
        axes[1,0].hist(residuals, bins=30, alpha=0.7)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Residuals')
        
        # Q-Q Plot
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        return fig
```

### 8.1.6 Phase 5: Evaluation

Model evaluation goes beyond simple metrics to assess business value and deployment readiness.

**Comprehensive Evaluation Framework:**
```python
class BusinessValueEvaluator:
    def __init__(self, business_metrics):
        self.business_metrics = business_metrics
        self.cost_benefit_analysis = {}
    
    def calculate_business_impact(self, model_results, baseline_results=None):
        """Calculate business impact of the model"""
        impact_analysis = {}
        
        # Revenue Impact
        if 'revenue_per_tp' in self.business_metrics:
            tp = model_results.get('true_positives', 0)
            revenue_impact = tp * self.business_metrics['revenue_per_tp']
            impact_analysis['revenue_increase'] = revenue_impact
        
        # Cost Savings
        if 'cost_per_fp' in self.business_metrics:
            fp = model_results.get('false_positives', 0)
            cost_savings = fp * self.business_metrics['cost_per_fp']
            impact_analysis['cost_reduction'] = cost_savings
        
        # Efficiency Gains
        if 'time_savings_per_prediction' in self.business_metrics:
            total_predictions = model_results.get('total_predictions', 0)
            time_savings = total_predictions * self.business_metrics['time_savings_per_prediction']
            impact_analysis['time_savings_hours'] = time_savings
        
        # ROI Calculation
        if baseline_results:
            improvement = self._calculate_improvement(model_results, baseline_results)
            impact_analysis['performance_improvement'] = improvement
        
        return impact_analysis
    
    def deployment_readiness_check(self, model, evaluation_results):
        """Assess if model is ready for deployment"""
        readiness_checklist = {
            'performance_threshold': self._check_performance_threshold(evaluation_results),
            'bias_fairness': self._check_bias_fairness(evaluation_results),
            'robustness': self._check_robustness(model, evaluation_results),
            'interpretability': self._check_interpretability(model),
            'scalability': self._check_scalability(model),
            'monitoring_setup': self._check_monitoring_setup()
        }
        
        # Calculate overall readiness score
        readiness_score = sum(readiness_checklist.values()) / len(readiness_checklist)
        
        return {
            'readiness_score': readiness_score,
            'checklist': readiness_checklist,
            'recommendations': self._get_deployment_recommendations(readiness_checklist)
        }
    def _check_performance_threshold(self, results):
        """Check if model meets minimum performance requirements"""
        # Define minimum thresholds (these should be business-specific)
        min_thresholds = {
            'accuracy': 0.80,
            'precision': 0.75,
            'recall': 0.70,
            'f1_score': 0.75,
            'roc_auc': 0.80
        }
        
        for metric, threshold in min_thresholds.items():
            if metric in results and results[metric] < threshold:
                return False
        
        return True
    
    def generate_model_card(self, model, evaluation_results, training_details):
        """Generate a comprehensive model card for documentation"""
        model_card = {
            'model_details': {
                'model_type': type(model).__name__,
                'model_version': training_details.get('version', '1.0'),
                'training_date': training_details.get('training_date'),
                'developer': training_details.get('developer'),
                'intended_use': training_details.get('intended_use')
            },
            'performance_metrics': evaluation_results,
            'training_data': {
                'dataset_description': training_details.get('dataset_description'),
                'data_size': training_details.get('data_size'),
                'data_preprocessing': training_details.get('preprocessing_steps')
            },
            'evaluation_data': {
                'test_size': training_details.get('test_size'),
                'validation_strategy': training_details.get('validation_strategy')
            },
            'ethical_considerations': {
                'bias_analysis': training_details.get('bias_analysis'),
                'fairness_metrics': training_details.get('fairness_metrics'),
                'limitations': training_details.get('limitations')
            },
            'deployment_considerations': {
                'infrastructure_requirements': training_details.get('infrastructure_requirements'),
                'monitoring_strategy': training_details.get('monitoring_strategy'),
                'update_frequency': training_details.get('update_frequency')
            }
        }
        
        return model_card
```

### 8.1.7 Phase 6: Deployment

Deployment transforms a model from a research artifact into a production system.

**MLOps Pipeline Implementation:**
```python
import joblib
import json
from datetime import datetime
import logging
from pathlib import Path

class ModelDeploymentPipeline:
    def __init__(self, model_name, version="1.0"):
        self.model_name = model_name
        self.version = version
        self.deployment_date = datetime.now()
        self.model_registry = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def package_model(self, model, preprocessor, feature_names, model_metadata):
        """Package model with all necessary components"""
        model_package = {
            'model': model,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'metadata': {
                **model_metadata,
                'model_name': self.model_name,
                'version': self.version,
                'deployment_date': self.deployment_date.isoformat(),
                'model_type': type(model).__name__
            }
        }
        
        return model_package
    
    def save_model_artifacts(self, model_package, artifacts_dir="model_artifacts"):
        """Save all model artifacts to disk"""
        artifacts_path = Path(artifacts_dir) / self.model_name / self.version
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_path / "model.joblib"
        joblib.dump(model_package['model'], model_path)
        
        # Save preprocessor
        preprocessor_path = artifacts_path / "preprocessor.joblib"
        joblib.dump(model_package['preprocessor'], preprocessor_path)
        
        # Save metadata
        metadata_path = artifacts_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_package['metadata'], f, indent=2, default=str)
        
        # Save feature names
        features_path = artifacts_path / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(model_package['feature_names'], f, indent=2)
        
        self.logger.info(f"Model artifacts saved to {artifacts_path}")
        
        return artifacts_path
    
    def create_prediction_api(self, model_package):
        """Create a simple Flask API for model predictions"""
        flask_code = f'''
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

app = Flask(__name__)

# Load model artifacts
MODEL_DIR = Path("model_artifacts/{self.model_name}/{self.version}")
model = joblib.load(MODEL_DIR / "model.joblib")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

with open(MODEL_DIR / "feature_names.json", 'r') as f:
    feature_names = json.load(f)

with open(MODEL_DIR / "metadata.json", 'r') as f:
    metadata = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            return jsonify({{'error': f'Missing features: {{missing_features}}'}})
        
        # Preprocess data
        X_processed = preprocessor.transform(df[feature_names])
        
        # Make predictions
        predictions = model.predict(X_processed)
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_processed)
            response = {{
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            }}
        else:
            response = {{'predictions': predictions.tolist()}}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({{'error': str(e)}}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify(metadata)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{'status': 'healthy', 'model': '{self.model_name}', 'version': '{self.version}'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''
        
        api_path = Path(f"api_{self.model_name}_{self.version}.py")
        with open(api_path, 'w') as f:
            f.write(flask_code)
        
        self.logger.info(f"API code generated: {api_path}")
        return api_path

class ModelMonitor:
    def __init__(self, model_name, monitoring_config):
        self.model_name = model_name
        self.monitoring_config = monitoring_config
        self.metrics_history = []
        
    def log_prediction(self, input_data, prediction, actual=None, timestamp=None):
        """Log a prediction for monitoring"""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp,
            'input_data': input_data,
            'prediction': prediction,
            'actual': actual
        }
        
        self.metrics_history.append(log_entry)
    
    def detect_drift(self, current_data, reference_data, method='ks_test'):
        """Detect data drift between current and reference datasets"""
        from scipy.stats import ks_2samp
        
        drift_results = {}
        
        for column in current_data.columns:
            if column in reference_data.columns:
                if method == 'ks_test':
                    statistic, p_value = ks_2samp(
                        reference_data[column].dropna(),
                        current_data[column].dropna()
                    )
                    
                    drift_results[column] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05
                    }
        
        return drift_results
    
    def calculate_performance_metrics(self, predictions, actuals):
        """Calculate performance metrics for monitoring"""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have the same length")
        
        # This would be customized based on problem type
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, average='weighted'),
            'recall': recall_score(actuals, predictions, average='weighted'),
            'sample_count': len(predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
```

---

## 8.2 Case Study 1: Customer Churn Prediction

### 8.2.1 Business Problem Definition

**Scenario:** A telecommunications company wants to predict which customers are likely to churn (cancel their service) in the next month to enable proactive retention efforts.

**Business Objectives:**
- Reduce customer churn by 15%
- Increase customer lifetime value
- Optimize retention campaign targeting
- Achieve model accuracy > 85%

```python
# Project setup and data loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionProject:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        
    def load_and_explore_data(self):
        """Load and perform initial data exploration"""
        # Generate synthetic telecom churn dataset
        np.random.seed(42)
        n_customers = 10000
        
        # Customer demographics
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(45, 15, n_customers).astype(int),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'tenure_months': np.random.exponential(24, n_customers).astype(int),
            'monthly_charges': np.random.normal(70, 20, n_customers),
            'total_charges': np.random.exponential(2000, n_customers),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                            n_customers, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'],
                                             n_customers, p=[0.3, 0.2, 0.3, 0.2]),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                               n_customers, p=[0.4, 0.4, 0.2]),
            'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
            'multiple_lines': np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5]),
            'online_security': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'tech_support': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'streaming_tv': np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'streaming_movies': np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4]),
            'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16])
        }
        
        self.data = pd.DataFrame(customer_data)
        
        # Create churn target with realistic relationships
        churn_prob = 0.1  # Base churn probability
        
        # Adjust probability based on features
        prob_adjustments = np.zeros(n_customers)
        
        # Month-to-month contracts have higher churn
        prob_adjustments += np.where(self.data['contract_type'] == 'Month-to-month', 0.15, 0)
        
        # High monthly charges increase churn probability
        prob_adjustments += np.where(self.data['monthly_charges'] > 80, 0.1, 0)
        
        # Low tenure increases churn probability
        prob_adjustments += np.where(self.data['tenure_months'] < 12, 0.12, 0)
        
        # Electronic check payment method increases churn
        prob_adjustments += np.where(self.data['payment_method'] == 'Electronic check', 0.08, 0)
        
        # No online security increases churn
        prob_adjustments += np.where(self.data['online_security'] == 'No', 0.05, 0)
        
        # Senior citizens have higher churn
        prob_adjustments += np.where(self.data['senior_citizen'] == 1, 0.06, 0)
        
        final_churn_prob = np.clip(churn_prob + prob_adjustments, 0, 1)
        self.data['churn'] = np.random.binomial(1, final_churn_prob)
        
        print("Dataset created successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Churn rate: {self.data['churn'].mean():.1%}")
        
        return self.data
    
    def perform_eda(self):
        """Perform comprehensive exploratory data analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print("\n1. Basic Dataset Information:")
        print(f"   - Dataset shape: {self.data.shape}")
        print(f"   - Missing values: {self.data.isnull().sum().sum()}")
        print(f"   - Duplicate rows: {self.data.duplicated().sum()}")
        print(f"   - Churn rate: {self.data['churn'].mean():.1%}")
        
        # Target variable distribution
        print("\n2. Target Variable Distribution:")
        churn_counts = self.data['churn'].value_counts()
        print(f"   - No Churn (0): {churn_counts[0]:,} ({churn_counts[0]/len(self.data):.1%})")
        print(f"   - Churn (1): {churn_counts[1]:,} ({churn_counts[1]/len(self.data):.1%})")
        
        # Feature analysis
        print("\n3. Feature Analysis:")
        
        # Numerical features
        numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges']
        
        print("\n   Numerical Features Summary:")
        for feature in numerical_features:
            print(f"   - {feature}:")
            print(f"     Mean: {self.data[feature].mean():.2f}")
            print(f"     Std: {self.data[feature].std():.2f}")
            print(f"     Min: {self.data[feature].min():.2f}")
            print(f"     Max: {self.data[feature].max():.2f}")
        
        # Categorical features
        categorical_features = [col for col in self.data.columns 
                              if col not in numerical_features + ['customer_id', 'churn']]
        
        print("\n   Categorical Features Summary:")
        for feature in categorical_features:
            unique_values = self.data[feature].nunique()
            print(f"   - {feature}: {unique_values} unique values")
            top_values = self.data[feature].value_counts().head(3)
            print(f"     Top values: {dict(top_values)}")
        
        # Correlation with target
        print("\n4. Feature-Target Relationships:")
        
        # Numerical features correlation
        for feature in numerical_features:
            correlation = self.data[feature].corr(self.data['churn'])
            print(f"   - {feature} correlation with churn: {correlation:.3f}")
        
        # Categorical features churn rates
        print("\n   Churn rates by categorical features:")
        for feature in categorical_features:
            churn_by_category = self.data.groupby(feature)['churn'].mean().sort_values(ascending=False)
            print(f"   - {feature}:")
            for category, rate in churn_by_category.items():
                print(f"     {category}: {rate:.1%}")
        
        return self._create_eda_visualizations()
    
    def _create_eda_visualizations(self):
        """Create comprehensive EDA visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # 1. Churn distribution
        churn_counts = self.data['churn'].value_counts()
        axes[0,0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%')
        axes[0,0].set_title('Overall Churn Distribution')
        
        # 2. Age distribution by churn
        self.data.boxplot(column='age', by='churn', ax=axes[0,1])
        axes[0,1].set_title('Age Distribution by Churn Status')
        
        # 3. Tenure vs Churn
        self.data.boxplot(column='tenure_months', by='churn', ax=axes[0,2])
        axes[0,2].set_title('Tenure Distribution by Churn Status')
        
        # 4. Monthly charges vs Churn
        self.data.boxplot(column='monthly_charges', by='churn', ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges by Churn Status')
        
        # 5. Contract type vs Churn
        contract_churn = pd.crosstab(self.data['contract_type'], self.data['churn'], normalize='index')
        contract_churn.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Churn Rate by Contract Type')
        axes[1,1].legend(['No Churn', 'Churn'])
        
        # 6. Payment method vs Churn
        payment_churn = pd.crosstab(self.data['payment_method'], self.data['churn'], normalize='index')
        payment_churn.plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Churn Rate by Payment Method')
        axes[1,2].legend(['No Churn', 'Churn'])
        
        # 7. Internet service vs Churn
        internet_churn = pd.crosstab(self.data['internet_service'], self.data['churn'], normalize='index')
        internet_churn.plot(kind='bar', ax=axes[2,0])
        axes[2,0].set_title('Churn Rate by Internet Service')
        axes[2,0].legend(['No Churn', 'Churn'])
        
        # 8. Correlation heatmap
        numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'senior_citizen', 'churn']
        corr_matrix = self.data[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2,1])
        axes[2,1].set_title('Correlation Matrix')
        
        # 9. Feature importance preview (tenure vs monthly charges colored by churn)
        churn_0 = self.data[self.data['churn'] == 0]
        churn_1 = self.data[self.data['churn'] == 1]
        axes[2,2].scatter(churn_0['tenure_months'], churn_0['monthly_charges'], 
                         alpha=0.5, label='No Churn', c='blue')
        axes[2,2].scatter(churn_1['tenure_months'], churn_1['monthly_charges'], 
                         alpha=0.5, label='Churn', c='red')
        axes[2,2].set_xlabel('Tenure (months)')
        axes[2,2].set_ylabel('Monthly Charges')
        axes[2,2].set_title('Tenure vs Monthly Charges by Churn')
        axes[2,2].legend()
        
        plt.tight_layout()
        return fig
    
    def preprocess_data(self):
        """Comprehensive data preprocessing"""
        print("=== DATA PREPROCESSING ===")
        
        # Create a copy for preprocessing
        df_processed = self.data.copy()
        
        # 1. Handle missing values (if any)
        print(f"Missing values before cleaning: {df_processed.isnull().sum().sum()}")
        
        # 2. Feature engineering
        print("\n1. Creating new features...")
        
        # Customer value score
        df_processed['customer_value_score'] = (
            df_processed['tenure_months'] * df_processed['monthly_charges'] / 
            df_processed['monthly_charges'].max()
        )
        
        # Charges per month of tenure
        df_processed['charges_per_tenure'] = df_processed['total_charges'] / (df_processed['tenure_months'] + 1)
        
        # Service count
        service_features = ['phone_service', 'multiple_lines', 'online_security', 
                          'tech_support', 'streaming_tv', 'streaming_movies']
        df_processed['total_services'] = sum([
            (df_processed[feature] == 'Yes').astype(int) for feature in service_features
        ])
        
        # High value customer flag
        df_processed['high_value_customer'] = (
            (df_processed['monthly_charges'] > df_processed['monthly_charges'].quantile(0.75)) &
            (df_processed['tenure_months'] > 12)
        ).astype(int)
        
        # Contract risk score
        contract_risk = {'Month-to-month': 2, 'One year': 1, 'Two year': 0}
        df_processed['contract_risk_score'] = df_processed['contract_type'].map(contract_risk)
        
        print(f"New features created: customer_value_score, charges_per_tenure, total_services, high_value_customer, contract_risk_score")
        
        # 3. Encode categorical variables
        print("\n2. Encoding categorical variables...")
        
        # Binary categorical variables
        binary_features = ['gender', 'phone_service', 'multiple_lines', 'online_security', 
                          'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing']
        
        label_encoders = {}
        for feature in binary_features:
            le = LabelEncoder()
            df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature])
            label_encoders[feature] = le
        
        # One-hot encode multi-class categorical variables
        multi_class_features = ['contract_type', 'payment_method', 'internet_service']
        df_encoded = pd.get_dummies(df_processed, columns=multi_class_features, prefix=multi_class_features)
        
        print(f"Encoded features: {binary_features + multi_class_features}")
        
        # 4. Select final features for modeling
        feature_columns = []
        
        # Numerical features
        numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 
                            'senior_citizen', 'customer_value_score', 'charges_per_tenure',
                            'total_services', 'high_value_customer', 'contract_risk_score']
        feature_columns.extend(numerical_features)
        
        # Encoded binary features
        encoded_binary_features = [f'{feature}_encoded' for feature in binary_features]
        feature_columns.extend(encoded_binary_features)
        
        # One-hot encoded features
        onehot_features = [col for col in df_encoded.columns 
                          if any(col.startswith(prefix) for prefix in multi_class_features)]
        feature_columns.extend(onehot_features)
        
        # Prepare final dataset
        X = df_encoded[feature_columns]
        y = df_encoded['churn']
        
        print(f"\nFinal feature set: {len(feature_columns)} features")
        print(f"Feature names: {feature_columns}")
        
        # 5. Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 6. Scale numerical features
        print("\n3. Scaling features...")
        scaler = StandardScaler()
        
        # Identify numerical columns in the final feature set
        numerical_cols_final = [col for col in numerical_features if col in X.columns]
        
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        self.X_train_scaled[numerical_cols_final] = scaler.fit_transform(self.X_train[numerical_cols_final])
        self.X_test_scaled[numerical_cols_final] = scaler.transform(self.X_test[numerical_cols_final])
        
        self.preprocessor = {
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_columns': feature_columns,
            'numerical_columns': numerical_cols_final
        }
        
        print(f"Training set: {self.X_train_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")
        print(f"Class distribution in training set:")
        print(f"  No Churn: {(self.y_train == 0).sum()} ({(self.y_train == 0).mean():.1%})")
        print(f"  Churn: {(self.y_train == 1).sum()} ({(self.y_train == 1).mean():.1%})")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def build_and_evaluate_models(self):
        """Build and evaluate multiple models"""
        print("=== MODEL BUILDING AND EVALUATION ===")
        
        # Initialize model selector
        model_selector = ModelSelector(problem_type='classification', cv_folds=5, scoring='roc_auc')
        
        # Compare models
        print("\n1. Comparing multiple models...")
        model_results = model_selector.compare_models(self.X_train_scaled, self.y_train)
        
        # Hyperparameter tuning for best model
        print("\n2. Hyperparameter tuning...")
        best_model_search = model_selector.hyperparameter_tuning(
            self.X_train_scaled, self.y_train, 
            model_name=None,  # Will use best from comparison
            search_type='random'
        )
        
        self.model = model_selector.best_model
        
        # Evaluate on test set
        print("\n3. Final evaluation on test set...")
        evaluator = ModelEvaluator(problem_type='classification')
        evaluation_results = evaluator.comprehensive_evaluation(
            self.model, self.X_test_scaled, self.y_test,
            self.X_train_scaled, self.y_train
        )
        
        # Print key results
        print("\nFinal Model Performance:")
        print(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"  Precision: {evaluation_results['precision']:.4f}")
        print(f"  Recall: {evaluation_results['recall']:.4f}")
        print(f"  F1-Score: {evaluation_results['f1_score']:.4f}")
        print(f"  ROC-AUC: {evaluation_results['roc_auc']:.4f}")
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.preprocessor['feature_columns'],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return evaluation_results
    
    def business_impact_analysis(self):
        """Analyze business impact of the model"""
        print("=== BUSINESS IMPACT ANALYSIS ===")
        
        # Business assumptions
        business_metrics = {
            'avg_customer_value': 1200,  # Average annual customer value
            'retention_campaign_cost': 50,  # Cost per retention campaign
            'campaign_success_rate': 0.3,  # 30% of targeted customers retained
        }
        
        # Get test predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate confusion matrix components
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business impact calculations
        print("1. Current Model Performance:")
        print(f"   - True Positives (Correctly identified churners): {tp}")
        print(f"   - False Positives (Incorrectly flagged as churners): {fp}")
        print(f"   - True Negatives (Correctly identified non-churners): {tn}")
        print(f"   - False Negatives (Missed churners): {fn}")
        
        # Revenue impact
        saved_customers = tp * business_metrics['campaign_success_rate']
        revenue_saved = saved_customers * business_metrics['avg_customer_value']
        campaign_costs = (tp + fp) * business_metrics['retention_campaign_cost']
        net_benefit = revenue_saved - campaign_costs
        
        print(f"\n2. Business Impact:")
        print(f"   - Customers targeted for retention: {tp + fp}")
        print(f"   - Estimated customers saved: {saved_customers:.0f}")
        print(f"   - Revenue saved: ${revenue_saved:,.0f}")
        print(f"   - Campaign costs: ${campaign_costs:,.0f}")
        print(f"   - Net benefit: ${net_benefit:,.0f}")
        
        # ROI calculation
        if campaign_costs > 0:
            roi = (revenue_saved - campaign_costs) / campaign_costs * 100
            print(f"   - Return on Investment: {roi:.1f}%")
        
        # Cost of missed opportunities
        missed_revenue = fn * business_metrics['avg_customer_value']
        print(f"   - Revenue lost from missed churners: ${missed_revenue:,.0f}")
        
        return {
            'net_benefit': net_benefit,
            'revenue_saved': revenue_saved,
            'campaign_costs': campaign_costs,
            'customers_saved': saved_customers,
            'missed_revenue': missed_revenue
        }

# Demonstrate the complete churn prediction project
def run_churn_prediction_project():
    """Run the complete churn prediction project"""
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("=" * 50)
    
    # Initialize project
    project = ChurnPredictionProject()
    
    # Load and explore data
    data = project.load_and_explore_data()
    
    # Perform EDA
    eda_fig = project.perform_eda()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = project.preprocess_data()
    
    # Build and evaluate models
    evaluation_results = project.build_and_evaluate_models()
    
    # Analyze business impact
    business_impact = project.business_impact_analysis()
    
    # Create performance visualizations
    evaluator = ModelEvaluator(problem_type='classification')
    performance_fig = evaluator.visualize_performance(project.model, project.X_test_scaled, project.y_test)
    
    print("\n" + "=" * 50)
    print("PROJECT COMPLETE - READY FOR DEPLOYMENT")
    
    return project, evaluation_results, business_impact
```

---

## 8.3 Case Study 2: House Price Prediction

### 8.3.1 Problem Definition and Data Collection

**Scenario:** A real estate company wants to build an automated valuation model (AVM) to estimate house prices for their online platform and assist real estate agents with pricing strategies.

**Business Objectives:**
- Predict house prices with MAPE < 10%
- Provide transparent price estimates to customers
- Identify key factors affecting house prices
- Support pricing strategy decisions

```python
class HousePricePredictionProject:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        
    def generate_realistic_housing_data(self, n_samples=5000):
        """Generate realistic housing dataset"""
        np.random.seed(42)
        
        # Location factors (simplified)
        neighborhoods = ['Downtown', 'Suburbs', 'Waterfront', 'Industrial', 'Rural']
        neighborhood_multipliers = [1.5, 1.2, 1.8, 0.8, 0.9]
        
        # Generate base features
        data = {}
        
        # Basic property characteristics
        data['square_feet'] = np.random.normal(2000, 600, n_samples).astype(int)
        data['square_feet'] = np.clip(data['square_feet'], 500, 5000)
        
        data['bedrooms'] = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, 
                                          p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
        
        data['bathrooms'] = np.random.normal(2.5, 1, n_samples)
        data['bathrooms'] = np.clip(data['bathrooms'], 1, 5)
        
        # Property features
        data['age_years'] = np.random.exponential(20, n_samples).astype(int)
        data['age_years'] = np.clip(data['age_years'], 0, 100)
        
        data['garage_spaces'] = np.random.choice([0, 1, 2, 3], n_samples, 
                                               p=[0.1, 0.3, 0.5, 0.1])
        
        data['lot_size_sqft'] = np.random.normal(8000, 3000, n_samples).astype(int)
        data['lot_size_sqft'] = np.clip(data['lot_size_sqft'], 1000, 20000)
        
        # Categorical features
        data['neighborhood'] = np.random.choice(neighborhoods, n_samples)
        data['property_type'] = np.random.choice(['Single Family', 'Condo', 'Townhouse'], 
                                               n_samples, p=[0.7, 0.2, 0.1])
        data['heating_type'] = np.random.choice(['Gas', 'Electric', 'Oil'], n_samples, 
                                              p=[0.6, 0.3, 0.1])
        
        # Quality ratings (1-10 scale)
        data['overall_condition'] = np.random.choice(range(1, 11), n_samples,
                                                   p=[0.02, 0.03, 0.05, 0.1, 0.15, 
                                                      0.25, 0.2, 0.12, 0.06, 0.02])
        
        data['kitchen_quality'] = np.random.choice(range(1, 11), n_samples,
                                                 p=[0.05, 0.05, 0.1, 0.15, 0.2, 
                                                    0.2, 0.15, 0.07, 0.02, 0.01])
        
        # Binary features
        data['has_pool'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        data['has_fireplace'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        data['has_basement'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        data['recently_renovated'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate realistic price based on features
        base_price = 100000  # Base price
        
        # Square footage impact (most important factor)
        price = base_price + (df['square_feet'] * 120);
        
        # Neighborhood multiplier
        neighborhood_mult = df['neighborhood'].map(dict(zip(neighborhoods, neighborhood_multipliers)));
        price = price * neighborhood_mult;
        
        # Bedrooms and bathrooms
        price += df['bedrooms'] * 15000;
        price += df['bathrooms'] * 10000;
        
        # Age depreciation
        price *= (1 - df['age_years'] * 0.005);
        
        # Quality factors
        price *= (0.7 + df['overall_condition'] * 0.03);
        price *= (0.9 + df['kitchen_quality'] * 0.01);
        
        # Property type adjustments
        type_multipliers = {'Single Family': 1.0, 'Condo': 0.85, 'Townhouse': 0.92}
        type_mult = df['property_type'].map(type_multipliers);
        price *= type_mult;
        
        # Additional features
        price += df['garage_spaces'] * 8000;
        price += df['lot_size_sqft'] * 2;
        price += df['has_pool'] * 25000;
        price += df['has_fireplace'] * 8000;
        price += df['has_basement'] * 12000;
        price += df['recently_renovated'] * 20000;
        
        # Add some noise
        noise = np.random.normal(0, price * 0.1);
        price += noise;
        
        # Ensure positive prices
        price = np.maximum(price, 50000);
        
        df['price'] = price.astype(int)
        
        self.data = df
        return df
    
    def perform_eda_regression(self):
        """Perform EDA for regression problem"""
        print("=== HOUSING DATA EXPLORATORY ANALYSIS ===")
        
        print(f"\n1. Dataset Overview:")
        print(f"   - Shape: {self.data.shape}")
        print(f"   - Missing values: {self.data.isnull().sum().sum()}")
        print(f"   - Price range: ${self.data['price'].min():,} - ${self.data['price'].max():,}")
        print(f"   - Median price: ${self.data['price'].median():,}")
        print(f"   - Mean price: ${self.data['price'].mean():,.0f}")
        
        # Numerical features analysis
        numerical_features = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 
                            'garage_spaces', 'lot_size_sqft', 'overall_condition', 
                            'kitchen_quality', 'price']
        
        print(f"\n2. Numerical Features Summary:")
        print(self.data[numerical_features].describe())
        
        # Correlation with price
        print(f"\n3. Correlation with Price:")
        price_correlations = self.data[numerical_features].corr()['price'].sort_values(ascending=False)
        for feature, corr in price_correlations.items():
            if feature != 'price':
                print(f"   - {feature}: {corr:.3f}")
        
        # Categorical features analysis
        categorical_features = ['neighborhood', 'property_type', 'heating_type']
        print(f"\n4. Price by Categories:")
        
        for feature in categorical_features:
            print(f"\n   {feature}:")
            price_by_category = self.data.groupby(feature)['price'].agg(['mean', 'median', 'count'])
            for category in price_by_category.index:
                mean_price = price_by_category.loc[category, 'mean']
                median_price = price_by_category.loc[category, 'median']
                count = price_by_category.loc[category, 'count']
                print(f"     {category}: Mean=${mean_price:,.0f}, Median=${median_price:,.0f}, Count={count}")
        
        return self._create_regression_eda_plots()
    
    def _create_regression_eda_plots(self):
        """Create EDA visualizations for regression"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        
        # 1. Price distribution
        axes[0,0].hist(self.data['price'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('House Price Distribution')
        axes[0,0].set_xlabel('Price ($)')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Log price distribution (often more normal)
        log_price = np.log(self.data['price'])
        axes[0,1].hist(log_price, bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Log Price Distribution')
        axes[0,1].set_xlabel('Log(Price)')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Square feet vs Price
        axes[0,2].scatter(self.data['square_feet'], self.data['price'], alpha=0.5)
        axes[0,2].set_xlabel('Square Feet')
        axes[0,2].set_ylabel('Price ($)')
        axes[0,2].set_title('Square Feet vs Price')
        
        # 4. Age vs Price
        axes[1,0].scatter(self.data['age_years'], self.data['price'], alpha=0.5)
        axes[1,0].set_xlabel('Age (Years)')
        axes[1,0].set_ylabel('Price ($)')
        axes[1,0].set_title('Age vs Price')
        
        # 5. Bedrooms vs Price (box plot)
        self.data.boxplot(column='price', by='bedrooms', ax=axes[1,1])
        axes[1,1].set_title('Price Distribution by Bedrooms')
        axes[1,1].set_xlabel('Number of Bedrooms')
        
        # 6. Neighborhood vs Price
        neighborhood_prices = self.data.groupby('neighborhood')['price'].mean().sort_values()
        axes[1,2].bar(neighborhood_prices.index, neighborhood_prices.values)
        axes[1,2].set_title('Average Price by Neighborhood')
        axes[1,2].set_xlabel('Neighborhood')
        axes[1,2].set_ylabel('Average Price ($)')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        # 7. Overall condition vs Price
        self.data.boxplot(column='price', by='overall_condition', ax=axes[2,0])
        axes[2,0].set_title('Price by Overall Condition')
        axes[2,0].set_xlabel('Overall Condition (1-10)')
        
        # 8. Correlation heatmap
        numerical_cols = ['square_feet', 'bedrooms', 'bathrooms', 'age_years', 
                         'garage_spaces', 'overall_condition', 'kitchen_quality', 'price']
        corr_matrix = self.data[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2,1])
        axes[2,1].set_title('Feature Correlation Matrix')
        
        # 9. Price vs. Multiple features
        # Create a composite score and plot against price
        self.data['quality_score'] = (self.data['overall_condition'] + self.data['kitchen_quality']) / 2
        axes[2,2].scatter(self.data['quality_score'], self.data['price'], alpha=0.5)
        axes[2,2].set_xlabel('Average Quality Score')
        axes[2,2].set_ylabel('Price ($)')
        axes[2,2].set_title('Quality Score vs Price')
        
        plt.tight_layout()
        return fig

    def preprocess_regression_data(self):
        """Preprocess data for regression modeling"""
        print("=== REGRESSION DATA PREPROCESSING ===")
        
        # Feature engineering for regression
        df_processed = self.data.copy()
        
        # 1. Create new features
        print("\n1. Feature Engineering:")
        
        # Price per square foot (for analysis, not modeling)
        df_processed['price_per_sqft'] = df_processed['price'] / df_processed['square_feet']
        
        # Total rooms
        df_processed['total_rooms'] = df_processed['bedrooms'] + df_processed['bathrooms']
        
        # Property age categories
        df_processed['age_category'] = pd.cut(df_processed['age_years'], 
                                            bins=[0, 5, 15, 30, 100], 
                                            labels=['New', 'Recent', 'Mature', 'Old'])
        
        # Size categories
        df_processed['size_category'] = pd.cut(df_processed['square_feet'],
                                             bins=[0, 1200, 2000, 3000, 10000],
                                             labels=['Small', 'Medium', 'Large', 'Luxury'])
        
        # Quality score
        df_processed['quality_score'] = (df_processed['overall_condition'] + df_processed['kitchen_quality']) / 2
        
        # Lot efficiency (house size relative to lot size)
        df_processed['lot_efficiency'] = df_processed['square_feet'] / df_processed['lot_size_sqft']
        
        print(f"   Created features: total_rooms, age_category, size_category, quality_score, lot_efficiency")
        
        # 2. Handle categorical variables
        print("\n2. Encoding categorical variables...")
        
        # One-hot encode categorical variables
        categorical_features = ['neighborhood', 'property_type', 'heating_type', 'age_category', 'size_category']
        df_encoded = pd.get_dummies(df_processed, columns=categorical_features, prefix=categorical_features)
        
        # 3. Select features for modeling
        # Exclude target and intermediate variables
        exclude_features = ['price', 'price_per_sqft']
        feature_columns = [col for col in df_encoded.columns if col not in exclude_features]
        
        X = df_encoded[feature_columns]
        y = df_encoded['price']
        
        print(f"\nFeatures for modeling: {len(feature_columns)} features")
        
        # 4. Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 5. Feature scaling
        print("\n3. Feature scaling...")
        scaler = StandardScaler()
        
        # Scale all features for regression
        self.X_train_scaled = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        self.X_test_scaled = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        self.preprocessor = {
            'scaler': scaler,
            'feature_columns': feature_columns
        }
        
        print(f"Training set: {self.X_train_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")
        print(f"Target variable statistics:")
        print(f"  Training mean: ${self.y_train.mean():,.0f}")
        print(f"  Training std: ${self.y_train.std():,.0f}")
        print(f"  Training range: ${self.y_train.min():,.0f} - ${self.y_train.max():,.0f}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def build_regression_models(self):
        """Build and evaluate regression models"""
        print("=== REGRESSION MODEL BUILDING ===")
        
        # Initialize model selector for regression
        model_selector = ModelSelector(problem_type='regression', cv_folds=5, scoring='r2')
        
        # Compare models
        print("\n1. Comparing regression models...")
        model_results = model_selector.compare_models(self.X_train_scaled, self.y_train)
        
        # Hyperparameter tuning
        print("\n2. Hyperparameter tuning...")
        best_model_search = model_selector.hyperparameter_tuning(
            self.X_train_scaled, self.y_train,
            model_name=None,
            search_type='random'
        )
        
        self.model = model_selector.best_model
        
        # Comprehensive evaluation
        print("\n3. Model evaluation...")
        evaluator = ModelEvaluator(problem_type='regression')
        evaluation_results = evaluator.comprehensive_evaluation(
            self.model, self.X_test_scaled, self.y_test,
            self.X_train_scaled, self.y_train
        )
        
        # Calculate additional metrics
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # Within percentage thresholds
        errors = np.abs(self.y_test - y_pred) / self.y_test
        within_5_pct = (errors <= 0.05).mean() * 100
        within_10_pct = (errors <= 0.10).mean() * 100
        within_20_pct = (errors <= 0.20).mean() * 100
        
        print(f"\nRegression Model Performance:")
        print(f"  R² Score: {evaluation_results['r2_score']:.4f}")
        print(f"  RMSE: ${evaluation_results['rmse']:,.0f}")
        print(f"  MAE: ${evaluation_results['mae']:,.0f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Predictions within 5%: {within_5_pct:.1f}%")
        print(f"  Predictions within 10%: {within_10_pct:.1f}%")
        print(f"  Predictions within 20%: {within_20_pct:.1f}%")
        
        evaluation_results['mape'] = mape
        evaluation_results['within_5_pct'] = within_5_pct
        evaluation_results['within_10_pct'] = within_10_pct
        evaluation_results['within_20_pct'] = within_20_pct
        
        return evaluation_results
```

---

## 8.4 Case Study 3: Customer Segmentation (Unsupervised Learning)

### 8.4.1 Problem Definition and Implementation

**Scenario:** An e-commerce company wants to segment their customers for targeted marketing campaigns, personalized recommendations, and inventory planning.

```python
class CustomerSegmentationProject:
    def __init__(self):
        self.data = None
        self.customer_features = None
        self.segmentation_model = None
        self.segment_profiles = {}
        
    def generate_ecommerce_data(self, n_customers=10000):
        """Generate realistic e-commerce customer data"""
        np.random.seed(42)
        
        # Customer demographics
        data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(40, 15, n_customers).astype(int),
            'registration_days': np.random.exponential(365, n_customers).astype(int)
        }
        
        # Clip age to reasonable range
        data['age'] = np.clip(data['age'], 18, 80)
        
        # Purchase behavior (with realistic correlations)
        # Create different customer archetypes
        customer_types = np.random.choice(['bargain_hunter', 'premium', 'occasional', 'frequent'], 
                                        n_customers, p=[0.3, 0.2, 0.3, 0.2])
        
        # Initialize arrays
        total_orders = np.zeros(n_customers)
        total_spent = np.zeros(n_customers)
        avg_order_value = np.zeros(n_customers)
        days_since_last_order = np.zeros(n_customers)
        
        for i in range(n_customers):
            if customer_types[i] == 'bargain_hunter':
                total_orders[i] = np.random.poisson(15)
                avg_order_value[i] = np.random.normal(25, 8)
                days_since_last_order[i] = np.random.exponential(30)
            elif customer_types[i] == 'premium':
                total_orders[i] = np.random.poisson(8)
                avg_order_value[i] = np.random.normal(150, 50)
                days_since_last_order[i] = np.random.exponential(45)
            elif customer_types[i] == 'occasional':
                total_orders[i] = np.random.poisson(3)
                avg_order_value[i] = np.random.normal(60, 20)
                days_since_last_order[i] = np.random.exponential(90)
            else:  # frequent
                total_orders[i] = np.random.poisson(25)
                avg_order_value[i] = np.random.normal(80, 25)
                days_since_last_order[i] = np.random.exponential(15)
        
        # Ensure positive values
        avg_order_value = np.maximum(avg_order_value, 10)
        total_spent = total_orders * avg_order_value
        days_since_last_order = np.maximum(days_since_last_order, 1)
        
        data.update({
            'total_orders': total_orders.astype(int),
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'days_since_last_order': days_since_last_order.astype(int)
        })
        
        # Category preferences
        categories = ['electronics', 'clothing', 'home', 'books', 'sports']
        for category in categories:
            data[f'{category}_orders'] = np.random.poisson(data['total_orders'] * np.random.uniform(0.1, 0.4, n_customers))
        
        # Engagement metrics
        data['website_visits'] = np.random.poisson(data['total_orders'] * np.random.uniform(3, 8, n_customers))
        data['email_opens'] = np.random.binomial(50, 0.3, n_customers)  # Assumes 50 emails sent
        data['social_media_clicks'] = np.random.poisson(5, n_customers)
        
        # Channel preferences
        data['mobile_orders'] = np.random.binomial(data['total_orders'], 0.6)
        data['desktop_orders'] = data['total_orders'] - data['mobile_orders']
        
        self.data = pd.DataFrame(data)
        
        # Add true customer type for validation (normally wouldn't have this)
        self.data['true_segment'] = customer_types
        
        return self.data
    
    def feature_engineering_unsupervised(self):
        """Create features for customer segmentation"""
        print("=== FEATURE ENGINEERING FOR SEGMENTATION ===")
        
        df = self.data.copy()
        
        # 1. RFM Features (Recency, Frequency, Monetary)
        df['recency'] = df['days_since_last_order']
        df['frequency'] = df['total_orders']
        df['monetary'] = df['total_spent']
        
        # 2. Behavioral features
        df['orders_per_day'] = df['total_orders'] / np.maximum(df['registration_days'], 1)
        df['avg_days_between_orders'] = df['registration_days'] / np.maximum(df['total_orders'], 1)
        
        # 3. Category diversity
        category_columns = ['electronics_orders', 'clothing_orders', 'home_orders', 'books_orders', 'sports_orders']
        df['category_diversity'] = (df[category_columns] > 0).sum(axis=1)
        
        # 4. Engagement metrics
        df['engagement_score'] = (
            (df['website_visits'] / np.maximum(df['total_orders'], 1)) * 0.3 +
            (df['email_opens'] / 50) * 0.4 +  # Normalize by emails sent
            (df['social_media_clicks'] / 10) * 0.3  # Normalize
        )
        
        # 5. Channel preference
        df['mobile_preference'] = df['mobile_orders'] / np.maximum(df['total_orders'], 1)
        
        # 6. Customer lifetime value approximation
        df['customer_lifetime_value'] = df['avg_order_value'] * df['total_orders']
        
        # Select features for clustering
        clustering_features = [
            'recency', 'frequency', 'monetary', 'avg_order_value',
            'orders_per_day', 'category_diversity', 'engagement_score',
            'mobile_preference', 'age'
        ]
        
        self.customer_features = df[clustering_features]
        
        print(f"Features for clustering: {clustering_features}")
        print(f"Feature matrix shape: {self.customer_features.shape}")
        
        # Display feature statistics
        print(f"\nFeature Statistics:")
        print(self.customer_features.describe())
        
        return self.customer_features
    
    def perform_customer_segmentation(self, n_clusters_range=range(2, 10)):
        """Perform customer segmentation using multiple methods"""
        print("=== CUSTOMER SEGMENTATION ===")
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.customer_features)
        features_scaled_df = pd.DataFrame(features_scaled, 
                                        columns=self.customer_features.columns,
                                        index=self.customer_features.index)
        
        # 1. Determine optimal number of clusters
        print("\n1. Determining optimal number of clusters...")
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        inertias = []
        silhouette_scores = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            inertias.append(kmeans.inertia_)
            if n_clusters > 1:
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            print(f"   {n_clusters} clusters: Inertia={kmeans.inertia_:.0f}, "
                  f"Silhouette={silhouette_score(features_scaled, cluster_labels):.3f}")
        
        # Choose optimal number of clusters (highest silhouette score)
        optimal_clusters = n_clusters_range[np.argmax(silhouette_scores) + 1]  # +1 because silhouette starts from 2
        print(f"\nOptimal number of clusters: {optimal_clusters} (highest silhouette score)")
        
        # 2. Final clustering
        print(f"\n2. Performing final clustering with {optimal_clusters} clusters...")
        
        self.segmentation_model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = self.segmentation_model.fit_predict(features_scaled)
        
        # Add cluster labels to data
        self.data['cluster'] = cluster_labels
        self.customer_features['cluster'] = cluster_labels
        
        # 3. Profile segments
        print(f"\n3. Creating segment profiles...")
        self.segment_profiles = self._create_segment_profiles()
        
        # 4. Validation against true segments (if available)
        if 'true_segment' in self.data.columns:
            self._validate_clustering()
        
        return cluster_labels, self.segment_profiles
    
    def _create_segment_profiles(self):
        """Create detailed profiles for each segment"""
        profiles = {}
        
        for cluster_id in sorted(self.data['cluster'].unique()):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.data) * 100,
                'demographics': {
                    'avg_age': cluster_data['age'].mean(),
                    'avg_registration_days': cluster_data['registration_days'].mean()
                },
                'rfm': {
                    'avg_recency': cluster_data['days_since_last_order'].mean(),
                    'avg_frequency': cluster_data['total_orders'].mean(),
                    'avg_monetary': cluster_data['total_spent'].mean()
                },
                'behavior': {
                    'avg_order_value': cluster_data['avg_order_value'].mean(),
                    'category_diversity': cluster_data[['electronics_orders', 'clothing_orders', 
                                                     'home_orders', 'books_orders', 'sports_orders']].mean(),
                    'mobile_preference': (cluster_data['mobile_orders'] / 
                                        np.maximum(cluster_data['total_orders'], 1)).mean(),
                    'engagement_score': (cluster_data['email_opens'] / 50).mean()
                }
            }
            
            profiles[f'Segment_{cluster_id}'] = profile
            
            # Print profile
            print(f"\n   Segment {cluster_id} ({profile['size']:,} customers, {profile['percentage']:.1f}%):")
            print(f"     Demographics: Age={profile['demographics']['avg_age']:.1f}, "
                  f"Days registered={profile['demographics']['avg_registration_days']:.0f}")
            print(f"     RFM: Recency={profile['rfm']['avg_recency']:.0f} days, "
                  f"Frequency={profile['rfm']['avg_frequency']:.1f} orders, "
                  f"Monetary=${profile['rfm']['avg_monetary']:.0f}")
            print(f"     Behavior: AOV=${profile['behavior']['avg_order_value']:.0f}, "
                  f"Mobile pref={profile['behavior']['mobile_preference']:.1%}")
        
        return profiles
    
    def recommend_marketing_strategies(self):
        """Recommend marketing strategies for each segment"""
        print("\n=== MARKETING STRATEGY RECOMMENDATIONS ===")
        
        strategies = {}
        
        for segment_name, profile in self.segment_profiles.items():
            cluster_id = segment_name.split('_')[1]
            
            # Analyze segment characteristics
            high_value = profile['rfm']['avg_monetary'] > self.data['total_spent'].median()
            frequent_buyer = profile['rfm']['avg_frequency'] > self.data['total_orders'].median()
            recent_activity = profile['rfm']['avg_recency'] < self.data['days_since_last_order'].median()
            high_aov = profile['behavior']['avg_order_value'] > self.data['avg_order_value'].median()
            
            # Generate strategy recommendations
            recommendations = []
            
            if high_value and frequent_buyer:
                recommendations.extend([
                    "VIP/Premium loyalty program",
                    "Early access to new products",
                    "Personalized shopping experiences",
                    "High-value product recommendations"
                ])
            
            if not recent_activity:
                recommendations.extend([
                    "Re-engagement campaigns",
                    "Win-back offers with discounts",
                    "Reminder emails about abandoned carts",
                    "Survey to understand satisfaction issues"
                ])
            
            if frequent_buyer and not high_value:
                recommendations.extend([
                    "Upselling campaigns",
                    "Bundle offers",
                    "Category expansion recommendations",
                    "Volume discount programs"
                ])
            
            if high_aov and not frequent_buyer:
                recommendations.extend([
                    "Frequency-building campaigns",
                    "Subscription/repeat purchase incentives",
                    "Cross-selling based on purchase history",
                    "Seasonal reminders"
                ])
            
            if profile['behavior']['mobile_preference'] > 0.7:
                recommendations.append("Mobile-optimized campaigns and app notifications")
            else:
                recommendations.append("Email and desktop-focused campaigns")
            
            strategies[segment_name] = {
                'characteristics': {
                    'high_value': high_value,
                    'frequent_buyer': frequent_buyer,
                    'recent_activity': recent_activity,
                    'high_aov': high_aov
                },
                'recommendations': recommendations
            }
            
            print(f"\n{segment_name} Strategy:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        return strategies

def run_customer_segmentation_project():
    """Run complete customer segmentation project"""
    print("CUSTOMER SEGMENTATION PROJECT")
    print("=" * 50)
    
    project = CustomerSegmentationProject()
    
    # Generate data
    data = project.generate_ecommerce_data(n_customers=5000)
    
    # Feature engineering
    features = project.feature_engineering_unsupervised()
    
    # Perform segmentation
    clusters, profiles = project.perform_customer_segmentation()
    
    # Generate marketing recommendations
    strategies = project.recommend_marketing_strategies()
    
    return project, profiles, strategies
```

---

## 8.5 Case Study 4: Fraud Detection (Imbalanced Classification)

### 8.5.1 Problem Setup and Specialized Techniques

**Scenario:** A financial services company needs to detect fraudulent transactions in real-time to minimize financial losses while maintaining customer satisfaction.

```python
class FraudDetectionProject:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.evaluation_results = {}
    
    def generate_fraud_dataset(self, n_transactions=100000, fraud_rate=0.02):
        """Generate realistic fraud detection dataset"""
        np.random.seed(42)
        
        # Transaction features
        data = {
            'transaction_id': range(1, n_transactions + 1),
            'amount': np.random.lognormal(3, 1.5, n_transactions),  # Log-normal distribution for amounts
            'hour': np.random.randint(0, 24, n_transactions),
            'day_of_week': np.random.randint(0, 7, n_transactions),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'atm'], 
                                                n_transactions, p=[0.25, 0.15, 0.20, 0.15, 0.20, 0.05]),
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], 
                                               n_transactions, p=[0.7, 0.2, 0.1])
        }
        
        # Customer behavior features
        data['customer_age'] = np.random.normal(45, 15, n_transactions).astype(int)
        data['customer_age'] = np.clip(data['customer_age'], 18, 90)
        
        data['account_age_days'] = np.random.exponential(1000, n_transactions).astype(int)
        data['transactions_last_30_days'] = np.random.poisson(20, n_transactions)
        data['avg_transaction_amount'] = np.random.lognormal(2.5, 1, n_transactions)
        
        # Location and device features
        data['same_city_last_transaction'] = np.random.choice([0, 1], n_transactions, p=[0.1, 0.9])
        data['same_device'] = np.random.choice([0, 1], n_transactions, p=[0.05, 0.95])
        data['international_transaction'] = np.random.choice([0, 1], n_transactions, p=[0.95, 0.05])
        
        # Time-based features
        data['weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['night_time'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
        
        df = pd.DataFrame(data)
        
        # Create realistic fraud patterns
        fraud_probability = np.full(n_transactions, 0.001)  # Base fraud rate
        
        # High-risk patterns increase fraud probability
        fraud_probability += np.where(df['amount'] > df['amount'].quantile(0.95), 0.15, 0)  # Very high amounts
        fraud_probability += np.where(df['night_time'] == 1, 0.02, 0)  # Night transactions
        fraud_probability += np.where(df['international_transaction'] == 1, 0.08, 0)  # International
        fraud_probability += np.where(df['same_device'] == 0, 0.05, 0)  # Different device
        fraud_probability += np.where(df['same_city_last_transaction'] == 0, 0.03, 0)  # Different location
        fraud_probability += np.where(df['merchant_category'] == 'atm', 0.03, 0)  # ATM transactions
        fraud_probability += np.where(df['account_age_days'] < 30, 0.04, 0)  # New accounts
        
        # Amount vs. customer history
        amount_ratio = df['amount'] / df['avg_transaction_amount']
        fraud_probability += np.where(amount_ratio > 5, 0.1, 0)  # Much larger than usual
        
        # Multiple transactions in short time (velocity)
        fraud_probability += np.where(df['transactions_last_30_days'] > 50, 0.03, 0)
        
        # Clip probabilities
        fraud_probability = np.clip(fraud_probability, 0, 0.5)
        
        # Generate fraud labels
        df['is_fraud'] = np.random.binomial(1, fraud_probability)
        
        # Adjust to target fraud rate
        actual_fraud_rate = df['is_fraud'].mean()
        if actual_fraud_rate > fraud_rate:
            # Randomly convert some frauds to normal
            fraud_indices = df[df['is_fraud'] == 1].index
            n_to_convert = int((actual_fraud_rate - fraud_rate) * len(df))
            convert_indices = np.random.choice(fraud_indices, size=min(n_to_convert, len(fraud_indices)), replace=False)
            df.loc[convert_indices, 'is_fraud'] = 0
        
        self.data = df
        
        print(f"Dataset created:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.2%})")
        print(f"  Normal transactions: {(df['is_fraud'] == 0).sum():,}")
        
        return df
    
    def analyze_fraud_patterns(self):
        """Analyze fraud patterns in the data"""
        print("=== FRAUD PATTERN ANALYSIS ===")
        
        fraud_data = self.data[self.data['is_fraud'] == 1]
        normal_data = self.data[self.data['is_fraud'] == 0]
        
        print(f"\n1. Transaction Amount Analysis:")
        print(f"   Fraud transactions - Mean: ${fraud_data['amount'].mean():.2f}, Median: ${fraud_data['amount'].median():.2f}")
        print(f"   Normal transactions - Mean: ${normal_data['amount'].mean():.2f}, Median: ${normal_data['amount'].median():.2f}")
        
        print(f"\n2. Timing Patterns:")
        fraud_night_pct = (fraud_data['night_time'] == 1).mean() * 100
        normal_night_pct = (normal_data['night_time'] == 1).mean() * 100
        print(f"   Night-time transactions - Fraud: {fraud_night_pct:.1f}%, Normal: {normal_night_pct:.1f}%")
        
        fraud_weekend_pct = (fraud_data['weekend'] == 1).mean() * 100
        normal_weekend_pct = (normal_data['weekend'] == 1).mean() * 100
        print(f"   Weekend transactions - Fraud: {fraud_weekend_pct:.1f}%, Normal: {normal_weekend_pct:.1f}%")
        
        print(f"\n3. Location and Device Patterns:")
        fraud_intl_pct = (fraud_data['international_transaction'] == 1).mean() * 100
        normal_intl_pct = (normal_data['international_transaction'] == 1).mean() * 100
        print(f"   International - Fraud: {fraud_intl_pct:.1f}%, Normal: {normal_intl_pct:.1f}%")
        
        fraud_diff_device_pct = (fraud_data['same_device'] == 0).mean() * 100
        normal_diff_device_pct = (normal_data['same_device'] == 0).mean() * 100
        print(f"   Different device - Fraud: {fraud_diff_device_pct:.1f}%, Normal: {normal_diff_device_pct:.1f}%")
        
        return self._create_fraud_analysis_plots()
    
    def _create_fraud_analysis_plots(self):
        """Create visualizations for fraud analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Fraud distribution by amount
        sns.histplot(self.data[self.data['is_fraud'] == 1]['amount'], bins=50, kde=True, ax=axes[0,0])
        axes[0,0].set_title('Fraudulent Transactions Amount Distribution')
        axes[0,0].set_xlabel('Amount')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Transaction time analysis
        fraud_times = self.data[self.data['is_fraud'] == 1]['hour']
        sns.histplot(fraud_times, bins=24, kde=True, ax=axes[0,1])
        axes[0,1].set_title('Fraudulent Transactions by Hour of Day')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Day of week analysis
        fraud_days = self.data[self.data['is_fraud'] == 1]['day_of_week']
        sns.histplot(fraud_days, bins=7, kde=True, ax=axes[1,0])
        axes[1,0].set_title('Fraudulent Transactions by Day of Week')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Frequency')
        
        # 4. Correlation heatmap for fraud data
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Feature Correlation Matrix (Fraud Data)')
        
        plt.tight_layout()
        return fig
    
    def handle_class_imbalance(self):
        """Implement techniques to handle class imbalance"""
        print("=== HANDLING CLASS IMBALANCE ===")
        
        # Prepare features
        feature_columns = [col for col in self.data.columns 
                          if col not in ['transaction_id', 'is_fraud']]
        
        # Encode categorical variables
        df_encoded = pd.get_dummies(self.data[feature_columns + ['is_fraud']], 
                                  columns=['merchant_category', 'transaction_type'])
        
        X = df_encoded.drop('is_fraud', axis=1)
        y = df_encoded['is_fraud']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Original class distribution:")
        print(f"  Training: {self.y_train.value_counts().to_dict()}")
        print(f"  Testing: {self.y_test.value_counts().to_dict()}")
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        # Implement different sampling strategies
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTETomek
        
        sampling_strategies = {
            'original': (self.X_train_scaled, self.y_train),
            'smote': SMOTE(random_state=42),
            'adasyn': ADASYN(random_state=42),
            'undersampling': RandomUnderSampler(random_state=42),
            'smote_tomek': SMOTETomek(random_state=42)
        }
        
        self.resampled_datasets = {}
        
        for strategy_name, strategy in sampling_strategies.items():
            if strategy_name == 'original':
                self.resampled_datasets[strategy_name] = strategy
            else:
                X_resampled, y_resampled = strategy.fit_resample(self.X_train_scaled, self.y_train)
                self.resampled_datasets[strategy_name] = (X_resampled, y_resampled)
                
                print(f"\n{strategy_name.upper()} resampling:")
                print(f"  New shape: {X_resampled.shape}")
                print(f"  Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return self.resampled_datasets

    def train_fraud_models(self):
        """Train models with different sampling strategies and algorithms"""
        print("=== TRAINING FRAUD DETECTION MODELS ===")
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import xgboost as xgb
        
        # Models to test
        models = {
            'logistic_regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'random_forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'xgboost': xgb.XGBClassifier(random_state=42, scale_pos_weight=10),  # Handle imbalance
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Evaluation metrics for imbalanced datasets
        def evaluate_fraud_model(model, X_test, y_test, model_name, sampling_strategy):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            from sklearn.metrics import (classification_report, confusion_matrix, 
                                       roc_auc_score, precision_recall_fscore_support,
                                       average_precision_score)
            
            # Standard metrics
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Business metrics
            cost_per_fraud_missed = 1000  # Average loss per fraud
            cost_per_false_alarm = 10    # Cost to investigate false positive
            
            total_cost = (fn * cost_per_fraud_missed) + (fp * cost_per_false_alarm)
            
            return {
                'model': model_name,
                'sampling_strategy': sampling_strategy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'total_cost': total_cost
            }
        
        # Train and evaluate all combinations
        results = []
        
        for sampling_name, (X_train_resampled, y_train_resampled) in self.resampled_datasets.items():
            print(f"\nTesting sampling strategy: {sampling_name}")
            
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_resampled, y_train_resampled)
                    
                    # Evaluate
                    result = evaluate_fraud_model(model, self.X_test_scaled, self.y_test, 
                                                model_name, sampling_name)
                    results.append(result)
                    
                    print(f"  {model_name}: Precision={result['precision']:.3f}, "
                          f"Recall={result['recall']:.3f}, PR-AUC={result['pr_auc']:.3f}")
                    
                except Exception as e:
                    print(f"  {model_name}: Error - {str(e)}")
        
        # Convert to DataFrame for analysis
        self.evaluation_results = pd.DataFrame(results)
        
        # Find best model based on PR-AUC (better for imbalanced datasets)
        best_model_idx = self.evaluation_results['pr_auc'].idxmax()
        best_result = self.evaluation_results.iloc[best_model_idx]
        
        print(f"\n=== BEST MODEL PERFORMANCE ===")
        print(f"Model: {best_result['model']} with {best_result['sampling_strategy']}")
        print(f"Precision: {best_result['precision']:.3f}")
        print(f"Recall: {best_result['recall']:.3f}")
        print(f"F1-Score: {best_result['f1_score']:.3f}")
        print(f"ROC-AUC: {best_result['roc_auc']:.3f}")
        print(f"PR-AUC: {best_result['pr_auc']:.3f}")
        print(f"Business cost: ${best_result['total_cost']:,.0f}")
        
        return self.evaluation_results

def run_fraud_detection_project():
    """Run complete fraud detection project"""
    print("FRAUD DETECTION PROJECT")
    print("=" * 50)
    
    project = FraudDetectionProject()
    
    # Generate data
    data = project.generate_fraud_dataset(n_transactions=50000, fraud_rate=0.02)
    
    # Analyze patterns
    analysis_fig = project.analyze_fraud_patterns()
    
    # Handle imbalance
    resampled_datasets = project.handle_class_imbalance()
    
    # Train models
    results = project.train_fraud_models()
    
    return project, results
```

---

## 8.6 Practical Labs

### 8.6.1 Lab 1: End-to-End Pipeline Implementation

**Objective:** Build a complete ML pipeline from data ingestion to model deployment.

```python
# Lab 1: Complete ML Pipeline
def lab_ml_pipeline():
    """
    Lab Exercise: Build a complete ML pipeline for predicting employee attrition
    
    Tasks:
    1. Data loading and initial exploration
    2. Data preprocessing and feature engineering
    3. Model selection and hyperparameter tuning
    4. Model evaluation and interpretation
    5. Deployment preparation
    """
    
    print("LAB 1: COMPLETE ML PIPELINE")
    print("=" * 40)
    
    # Step 1: Generate employee attrition data
    np.random.seed(42)
    n_employees = 2000
    
    employee_data = {
        'employee_id': range(1, n_employees + 1),
        'age': np.random.normal(35, 8, n_employees).astype(int),
        'years_at_company': np.random.exponential(5, n_employees).astype(int),
        'salary': np.random.normal(70000, 20000, n_employees),
        'satisfaction_score': np.random.uniform(1, 10, n_employees),
        'performance_rating': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.05, 0.15, 0.6, 0.15, 0.05]),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 
                                     n_employees, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'remote_work': np.random.choice([0, 1], n_employees, p=[0.7, 0.3]),
        'overtime_hours': np.random.exponential(5, n_employees),
        'commute_distance': np.random.exponential(10, n_employees)
    }
    
    df = pd.DataFrame(employee_data)
    
    # Create realistic attrition patterns
    attrition_prob = 0.1  # Base probability
    
    # Factors increasing attrition
    prob_adjustments = np.zeros(n_employees)
    prob_adjustments += np.where(df['satisfaction_score'] < 5, 0.2, 0)
    prob_adjustments += np.where(df['years_at_company'] < 2, 0.15, 0)
    prob_adjustments += np.where(df['salary'] < 50000, 0.1, 0)
    prob_adjustments += np.where(df['overtime_hours'] > 10, 0.12, 0)
    prob_adjustments += np.where(df['performance_rating'] <= 2, 0.15, 0)
    prob_adjustments += np.where(df['commute_distance'] > 20, 0.08, 0)
    
    final_prob = np.clip(attrition_prob + prob_adjustments, 0, 0.8)
    df['attrition'] = np.random.binomial(1, final_prob)
    
    # Task instructions for students
    tasks = """
    TODO: Complete the following tasks:
    
    1. DATA EXPLORATION
       - Analyze target variable distribution
       - Identify numerical vs categorical features
       - Check for missing values and outliers
       - Calculate correlation with target
    
    2. FEATURE ENGINEERING
       - Create new features (e.g., tenure categories, salary bands)
       - Handle categorical variables
       - Scale numerical features
    
    3. MODEL BUILDING
       - Split data into train/validation/test
       - Try multiple algorithms
       - Perform hyperparameter tuning
    
    4. EVALUATION
       - Calculate comprehensive metrics
       - Create confusion matrix and ROC curve
       - Analyze feature importance
    
    5. DEPLOYMENT PREP
       - Create prediction pipeline
       - Validate on holdout test set
       - Document model performance
    """
    
    print(tasks)
    
    # Provide starter code structure
    starter_code = """
    # Starter code structure:
    
    # 1. Load and explore data
    print("Dataset shape:", df.shape)
    print("Attrition rate:", df['attrition'].mean())
    
    # 2. EDA - Add your analysis here
    # TODO: Implement exploratory data analysis
    
    # 3. Preprocessing - Add your preprocessing here  
    # TODO: Feature engineering and preprocessing
    
    # 4. Model training - Add your models here
    # TODO: Train and evaluate multiple models
    
    # 5. Final evaluation - Add evaluation code here
    # TODO: Comprehensive model evaluation
    """
    
    print(starter_code)
    return df
```

### 8.6.2 Lab 2: Model Interpretability and Explainability

def lab_model_interpretability():
    """
    Lab Exercise: Implement model interpretability techniques
    
    Covers:
    - SHAP values
    - LIME explanations  
    - Feature importance analysis
    - Partial dependence plots
    """
    
    print("LAB 2: MODEL INTERPRETABILITY")
    print("=" * 40)
    
    tasks = """
    INTERPRETABILITY LAB TASKS:
    
    1. GLOBAL INTERPRETABILITY
       - Calculate and plot feature importance
       - Create partial dependence plots
       - Analyze feature interactions
    
    2. LOCAL INTERPRETABILITY  
       - Implement SHAP explanations
       - Use LIME for individual predictions
       - Create explanation dashboards
    
    3. MODEL COMPARISON
       - Compare interpretability across model types
       - Analyze trade-offs between accuracy and interpretability
    
    4. BUSINESS INSIGHTS
       - Translate technical insights to business language
       - Identify actionable insights
       - Create executive summary
    """
    
    print(tasks)
    
    # Sample interpretability code
    sample_code = """
    # Sample interpretability implementation:
    
    import shap
    from lime import lime_tabular
    import matplotlib.pyplot as plt
    
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    
    # LIME explanations
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names, mode='classification'
    )
    explanation = lime_explainer.explain_instance(
        X_test.iloc[0], model.predict_proba
    )
    
    # Partial dependence plots
    from sklearn.inspection import plot_partial_dependence
    plot_partial_dependence(model, X_train, features=[0, 1, (0, 1)])
    """
    
    print(sample_code)

### 8.6.3 Lab 3: MLOps Pipeline Implementation

def lab_mlops_pipeline():
    """
    Lab Exercise: Implement MLOps pipeline with monitoring and deployment
    
    Covers:
    - Model versioning
    - Automated retraining
    - Performance monitoring
    - A/B testing setup
    """
    
    print("LAB 3: MLOPS PIPELINE")
    print("=" * 40)
    
    tasks = """
    MLOPS PIPELINE TASKS:
    
    1. VERSION CONTROL
       - Set up model versioning system
       - Track experiments and parameters
       - Implement model registry
    
    2. AUTOMATED PIPELINE
       - Create training pipeline
       - Implement validation checks
       - Set up automated deployment
    
    3. MONITORING SETUP
       - Implement data drift detection
       - Set up performance monitoring
       - Create alerting system
    
    4. A/B TESTING
       - Design A/B testing framework
       - Implement traffic splitting
       - Set up metrics collection
    """
    
    print(tasks)
    
    pipeline_template = """
    # MLOps Pipeline Template:
    
    class MLOpsPipeline:
        def __init__(self, model_name, version):
            self.model_name = model_name
            self.version = version
            self.model_registry = {}
        
        def train_pipeline(self, data_path):
            # Load data
            # Preprocess
            # Train model
            # Validate performance
            # Register model if valid
            pass
        
        def deploy_model(self, model_version):
            # Load model from registry
            # Create deployment package
            # Deploy to production
            # Update monitoring
            pass
        
        def monitor_performance(self):
            # Check data drift
            # Monitor accuracy
            # Check for anomalies
            # Send alerts if needed
            pass
    """
    
    print(pipeline_template)
```

---

## 8.7 Best Practices and Common Pitfalls

### 8.7.1 Data Science Best Practices

**1. Data Quality Assurance**
- Always validate data quality before modeling
- Document data sources and collection methods
- Implement data validation checks in production
- Monitor for data drift and quality degradation

**2. Reproducibility**
- Set random seeds for all stochastic processes
- Version control all code and configuration
- Document environment dependencies
- Use containerization for deployment consistency

**3. Model Validation**
- Use appropriate cross-validation strategies
- Hold out a final test set for unbiased evaluation
- Validate on out-of-time data when applicable
- Test model robustness with adversarial examples

### 8.7.2 Common Pitfalls and How to Avoid Them

```python
class MLProjectPitfalls:
    """Common pitfalls in ML projects and how to avoid them"""
    
    @staticmethod
    def data_leakage_examples():
        """Examples of data leakage and prevention"""
        
        pitfalls = {
            "Future Information Leakage": {
                "description": "Using information that wouldn't be available at prediction time",
                "example": "Including 'days_since_last_transaction' in a fraud detection model where you're trying to predict fraud in real-time",
                "solution": "Carefully review features for temporal consistency"
            },
            
            "Target Leakage": {
                "description": "Including features that are direct derivatives of the target",
                "example": "Using 'approved_loan_amount' to predict loan approval",
                "solution": "Remove features that are consequences of the target variable"
            },
            
            "Train-Test Contamination": {
                "description": "Information from test set influencing training",
                "example": "Scaling features using statistics from entire dataset before splitting",
                "solution": "Always split data before any preprocessing that involves statistics"
            }
        }
        
        return pitfalls
    
    @staticmethod
    def sampling_bias_prevention():
        """Prevent sampling and selection bias"""
        
        prevention_strategies = {
            "Temporal Splits": "Use time-based splits for time series data",
            "Stratified Sampling": "Maintain class distributions across splits", 
            "Representative Sampling": "Ensure test data represents production distribution",
            "Cross-Validation": "Use appropriate CV strategy for your data type"
        }
        
        return prevention_strategies
    
    @staticmethod
    def overfitting_prevention():
        """Comprehensive overfitting prevention strategies"""
        
        strategies = {
            "Regularization": "Use L1/L2 regularization or dropout",
            "Early Stopping": "Stop training when validation performance degrades",
            "Feature Selection": "Remove irrelevant/redundant features",
            "Cross-Validation": "Use proper validation to assess generalization",
            "Ensemble Methods": "Combine multiple models to reduce variance",
            "Data Augmentation": "Increase training data diversity when possible"
        }
        
        return strategies

# Checklist for ML Project Success
ml_project_checklist = {
    "Business Understanding": [
        "✓ Clear problem definition",
        "✓ Success metrics defined", 
        "✓ Stakeholder alignment",
        "✓ Resource constraints identified"
    ],
    
    "Data Preparation": [
        "✓ Data quality assessed",
        "✓ Missing value strategy defined",
        "✓ Feature engineering completed",
        "✓ Data leakage prevented"
    ],
    
    "Modeling": [
        "✓ Baseline model established",
        "✓ Multiple algorithms tested",
        "✓ Hyperparameters optimized",
        "✓ Cross-validation performed"
    ],
    
    "Evaluation": [
        "✓ Appropriate metrics selected",
        "✓ Business impact calculated",
        "✓ Model interpretability assessed",
        "✓ Bias and fairness evaluated"
    ],
    
    "Deployment": [
        "✓ Production pipeline designed",
        "✓ Monitoring strategy implemented",
        "✓ Rollback plan prepared",
        "✓ Documentation completed"
    ]
}
```

---

## 8.8 Chapter Summary

This chapter provided comprehensive coverage of end-to-end machine learning projects through the CRISP-DM methodology and four detailed case studies:

### Key Learnings:

1. **CRISP-DM Methodology**: Structured approach to ML projects with six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

2. **Case Studies Completed**:
   - **Customer Churn Prediction**: Binary classification with business impact analysis
   - **House Price Prediction**: Regression modeling with feature engineering
   - **Customer Segmentation**: Unsupervised learning for marketing insights  
   - **Fraud Detection**: Handling severely imbalanced datasets

3. **Technical Implementation**: Complete code examples for data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation.

4. **Business Integration**: Frameworks for translating technical results into business value and actionable insights.

5. **MLOps Considerations**: Model deployment, monitoring, and maintenance strategies for production systems.

### Best Practices Emphasized:
- Systematic approach to problem-solving
- Comprehensive data quality assessment
- Appropriate handling of different data types and challenges
- Business-focused evaluation and interpretation
- Deployment readiness assessment

### Next Steps:
The next chapter will focus on Model Selection and Evaluation techniques, diving deeper into advanced evaluation methodologies, cross-validation strategies, and model comparison frameworks that build upon the foundation established in this chapter.

---

## Exercises

### Exercise 8.1: CRISP-DM Implementation
Apply the CRISP-DM methodology to a new domain (e.g., healthcare, retail, manufacturing). Document each phase and identify domain-specific challenges.

### Exercise 8.2: Feature Engineering Workshop  
Given a raw dataset, implement comprehensive feature engineering including:
- Temporal features
- Interaction terms
- Domain-specific transformations
- Dimensionality reduction

### Exercise 8.3: Model Interpretation
Take one of the case study models and implement multiple interpretability techniques:
- SHAP values
- LIME explanations
- Permutation importance
- Partial dependence plots

### Exercise 8.4: Deployment Pipeline
Design and implement a complete deployment pipeline including:
- Model packaging
- API creation
- Monitoring setup
- A/B testing framework

### Exercise 8.5: Business Impact Analysis
For each case study, perform detailed business impact analysis including:
- ROI calculations
- Cost-benefit analysis
- Risk assessment
- Implementation timeline
