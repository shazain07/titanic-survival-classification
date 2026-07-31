# Titanic Survival Prediction — Logistic Regression Model

A Machine Learning classification project built using `scikit-learn` and `pandas` to predict passenger survival on the Titanic.

## 📌 Problem Overview
The objective is to binary-classify passengers as either **Survived (1)** or **Did Not Survive (0)** based on demographic and voyage attributes such as age, sex, passenger class, fare, and embarkation port.

---

## 🛠️ Machine Learning Workflow

1. **Data Preprocessing & Imputation:**
   - Handled missing values in `Age` and `Fare` using median imputation.
   - Imputed missing categorical entries in `Embarked` using the mode.
2. **Feature Engineering & One-Hot Encoding:**
   - Selected features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
   - Converted categorical strings into binary numerical indicators using `pd.get_dummies(drop_first=True)` to prevent dummy variable redundancy.
3. **Dataset Splitting & Scaling:**
   - Split dataset into **80% training** and **20% testing** sets with class stratification (`stratify=y`).
   - Scaled continuous features using `StandardScaler` to ensure optimal gradient convergence.
4. **Model Training:**
   - Trained a baseline `LogisticRegression` classifier.

---

## 📊 Results & Performance Evaluation

* **Test Accuracy:** **80.45%**

### Confusion Matrix Breakdown
