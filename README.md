# Heart Stroke Prediction Analysis

A comprehensive machine learning project for predicting stroke risk using healthcare data. This project implements multiple ensemble learning algorithms, performs extensive hyperparameter tuning, and delivers a production-ready model for stroke prediction.

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Code Snippets and Significance](#code-snippets-and-significance)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

##  Overview

This project aims to predict the likelihood of stroke occurrence based on various health indicators including hypertension, heart disease, average glucose level, BMI, and smoking status. The project follows a complete machine learning pipeline from data exploration to model deployment.

**Key Objectives:**
- Analyze healthcare data to identify stroke risk factors
- Compare multiple machine learning algorithms
- Optimize model performance through hyperparameter tuning
- Deploy the best-performing model for predictions

##  Features

- **Comprehensive EDA**: Exploratory data analysis with visualizations
- **Data Preprocessing**: Handles missing values, categorical encoding, and data type conversions
- **Multiple Models**: Compares Random Forest, XGBoost, and Gradient Boosting classifiers
- **Hyperparameter Tuning**: Uses RandomizedSearchCV for optimal parameter selection
- **Model Persistence**: Saves the best model for future predictions
- **Stratified Splitting**: Ensures balanced train-test distribution

##  File Structure📁

```
Heart-Stroke/
│
├── README.md                          # Project documentation
├── requirements.txt                    # Python dependencies
│
└── HEART-STROKE-ANALYSIS-/
    │
    ├── Heartstroke.ipynb              # Main Jupyter notebook with complete analysis
    │
    ├── Best_model/
    │   └── TheHonouredOne.pkl         # Saved best-performing model (XGBoost)
    │
    ├── healthcare-dataset-stroke-data.csv          # Processed dataset
    ├── healthcare-dataset-stroke-data(raw).csv     # Raw dataset
    │
    └── venv_heart/                    # Virtual environment (not tracked in git)
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Heart-Stroke
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open the notebook**
   - Navigate to `HEART-STROKE-ANALYSIS-/Heartstroke.ipynb`

## Code Snippets and Significance

### 1. Data Loading and Initial Exploration

```python
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()
```

**Significance**: Loads the healthcare dataset and displays the first few rows to understand the data structure and features.

---

### 2. Data Cleaning - Fixing Redundant Values

```python
df['smoking_status'] = df['smoking_status'].str.replace('never smokes', 'never smoked')
df['smoking_status'].value_counts()
```

**Significance**: Standardizes categorical values by replacing the redundant "never smokes" with "never smoked" to ensure data consistency, which is crucial for proper model training.

---

### 3. Feature Selection

```python
feature_cols = ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
target_col = 'stroke'

X = df[feature_cols].copy()   
y = df[target_col].copy().astype(int)
```

**Significance**: Selects relevant features for prediction. These features are medically significant indicators of stroke risk. The target variable is converted to integer type for classification.

---

### 4. Stratified Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=40
)
```

**Significance**: 
- Splits data into 75% training and 25% testing sets
- `stratify=y` ensures both sets maintain the same class distribution (important for imbalanced datasets)
- `random_state=40` ensures reproducibility

---

### 5. Categorical Encoding

```python
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train.loc[:, col] = X_train.loc[:, col].fillna("UNKNOWN")
    X_test.loc[:, col] = X_test.loc[:, col].fillna("UNKNOWN")
    X_train.loc[:, col] = le.fit_transform(X_train.loc[:, col])
    X_test.loc[:, col] = le.transform(X_test.loc[:, col])
    encoders[col] = le
```

**Significance**: 
- Converts categorical variables (like `smoking_status`) to numerical format
- Handles missing values by filling with "UNKNOWN"
- Uses LabelEncoder to maintain ordinal relationships
- Stores encoders for future predictions on new data

---

### 6. Numerical Feature Imputation

```python
num_cols = [c for c in feature_cols if c not in cat_cols]
for c in num_cols:
    X_train.loc[:, c] = pd.to_numeric(X_train.loc[:, c], errors='coerce')
    X_test.loc[:, c] = pd.to_numeric(X_test.loc[:, c], errors='coerce')
    med = X_train.loc[:, c].median()
    X_train.loc[:, c] = X_train.loc[:, c].fillna(med)
    X_test.loc[:, c] = X_test.loc[:, c].fillna(med)
```

**Significance**: 
- Handles missing values in numerical features using median imputation (robust to outliers)
- Converts to numeric type, handling any conversion errors gracefully
- Uses training set median to avoid data leakage

---

### 7. Model Comparison Function

```python
def Model(models : dict) -> list:
    model_w_accuracy = {}
    the_honored_one = []

    for model in range(len(list(models))):
        estimator = list(models.values())[model]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        print(f"{list(models.keys())[model]}")
        print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        model_w_accuracy[estimator] = accuracy
    
    # Select best model based on accuracy
    for model in range(len(list(models))):
        if (list(model_w_accuracy.values()))[model] == max(list(model_w_accuracy.values())):
            the_honored_one.append(list(model_w_accuracy.keys())[model])
            
    return the_honored_one[0]
```

**Significance**: 
- Systematically compares multiple models (Random Forest, XGBoost, Gradient Boosting)
- Automatically selects the best-performing model based on accuracy
- Provides detailed classification reports for each model
- Returns the "honored one" (best model) for further optimization

---

### 8. Hyperparameter Tuning with RandomizedSearchCV

```python
def gridSearch(grid : dict, best_model : tuple):
    randomCV_model = RandomizedSearchCV(
        estimator = model_use,
        param_distributions=grid,
        cv = 5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    
    randomCV_model.fit(X_train, y_train)
    return randomCV_model
```

**Significance**: 
- Uses RandomizedSearchCV for efficient hyperparameter optimization
- Tests multiple parameter combinations across 5-fold cross-validation
- `n_jobs=-1` utilizes all CPU cores for faster computation
- Finds optimal hyperparameters to maximize model accuracy

---

### 9. Model Persistence

```python
def prediction(new_test_data : pd.DataFrame, label_mapping : dict, final_model):
    xg_model = final_model
    y_new_pred = xg_model.predict(new_test_data)
    y_pred_category = np.array([label_mapping[pred] for pred in y_new_pred])
    
    pickle.dump(xg_model, open('Best_model/TheHonouredOne.pkl', "wb"))
    return sorted(y_pred_category, reverse=True)
```

**Significance**: 
- Saves the trained model using pickle for future use
- Converts numerical predictions to human-readable labels ("Stroke" or "No Stroke")
- Enables model deployment without retraining
- The model file `TheHonouredOne.pkl` can be loaded later for predictions

---

## Model Performance📊

### Model Comparison Results

| Model | Accuracy | Precision (Class 0) | Precision (Class 1) | Recall (Class 0) | Recall (Class 1) |
|-------|----------|---------------------|---------------------|------------------|------------------|
| **XGBoost** | **95%** | 0.95 | 0.36 | 0.99 | 0.06 |
| Random Forest | 94% | 0.95 | 0.09 | 0.99 | 0.02 |
| Gradient Boosting | 94% | 0.95 | 0.00 | 0.99 | 0.00 |

**Note**: The dataset is highly imbalanced (fewer stroke cases), which explains the lower recall for the positive class. The XGBoost model was selected as the best performer and further optimized through hyperparameter tuning.

### Optimized XGBoost Performance

After hyperparameter tuning with RandomizedSearchCV:
- **Cross-Validation Accuracy**: 94.81%
- **Best Parameters**: 
  - `n_estimators`: 400
  - `max_depth`: 4
  - `learning_rate`: 0.1
  - `subsample`: 0.7
  - `colsample_bytree`: 0.9
  - And other optimized parameters

## Dataset

The dataset contains healthcare information with the following features:

- **age**: Patient age
- **hypertension**: Binary indicator (0 = No, 1 = Yes)
- **heart_disease**: Binary indicator (0 = No, 1 = Yes)
- **avg_glucose_level**: Average glucose level in blood
- **bmi**: Body Mass Index
- **smoking_status**: Categorical (formerly smoked, never smoked, smokes)
- **stroke**: Target variable (0 = No Stroke, 1 = Stroke)

**Dataset Statistics:**
- Total samples: 5,110
- Training samples: 3,832 (75%)
- Testing samples: 1,278 (25%)
- Class distribution: Highly imbalanced (majority class: No Stroke)

## 🛠 Technologies Used

- **Python 3.x**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and utilities
  - RandomForestClassifier
  - GradientBoostingClassifier
  - LabelEncoder
  - train_test_split
  - RandomizedSearchCV
  - Classification metrics
- **XGBoost**: Gradient boosting framework
- **Jupyter Notebook**: Interactive development environment
- **Pickle**: Model serialization

## Usage

### Running the Complete Analysis

1. Open `HEART-STROKE-ANALYSIS-/Heartstroke.ipynb` in Jupyter Notebook
2. Run all cells sequentially to:
   - Load and explore the data
   - Perform data preprocessing
   - Train and compare models
   - Optimize hyperparameters
   - Save the best model

### Making Predictions with Saved Model

```python
import pickle
import pandas as pd

# Load the saved model
with open('HEART-STROKE-ANALYSIS-/Best_model/TheHonouredOne.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare new data (example)
new_data = pd.DataFrame({
    'hypertension': [0],
    'heart_disease': [1],
    'avg_glucose_level': [150.0],
    'bmi': [28.5],
    'smoking_status': [1]  # Encoded value
})

# Make prediction
prediction = model.predict(new_data)
print("Stroke Risk:", "High" if prediction[0] == 1 else "Low")
```

##  Key Insights

1. **Feature Importance**: The model identifies hypertension, heart disease, and glucose levels as significant predictors
2. **Class Imbalance**: The dataset is imbalanced, which affects recall for the minority class (stroke cases)
3. **Model Selection**: XGBoost outperformed other ensemble methods, likely due to its ability to handle imbalanced data better
4. **Outlier Handling**: High glucose levels were retained as they may indicate diabetic patients, which is medically relevant

##  Important Notes

- **Medical Disclaimer**: This model is for educational purposes only and should not be used as a substitute for professional medical advice
- **Data Imbalance**: The dataset has significantly fewer stroke cases, which may affect model performance on rare cases
- **Feature Engineering**: Consider adding more features or feature interactions for improved performance
- **Model Validation**: For production use, consider additional validation techniques like k-fold cross-validation and external validation datasets

##  License

This project is open source and available for educational purposes.

---

**Author1**: [Sujal-G-Sanyasi](https://github.com/Sujal-G-Sanyasi) **and** [vyshnavireddy12](https://github.com/vyshnavireddy12)          
**Date**: 2024  
**Version**: 1.0
