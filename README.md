Titanic Survival Prediction — Logistic Regression Model
A Machine Learning classification project built using scikit-learn and pandas to predict passenger survival on the Titanic dataset.

📌 Problem Overview
The objective is to binary-classify passengers as either Survived (1) or Did Not Survive (0) based on demographic and voyage attributes such as age, sex, passenger class, fare, and embarkation port.

🛠️ Machine Learning Workflow
Data Preprocessing & Imputation:

Handled missing values in Age and Fare using median imputation.

Imputed missing categorical entries in Embarked using the mode.

Feature Engineering & One-Hot Encoding:

Selected predictive features: Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

Converted categorical strings into binary numerical indicators using pd.get_dummies(drop_first=True) to prevent dummy variable redundancy (multicollinearity).

Dataset Splitting & Scaling:

Split dataset into 80% training and 20% testing sets with class stratification (stratify=y) to maintain target distribution.

Scaled continuous features using StandardScaler to optimize model convergence.

Model Training:

Trained a baseline LogisticRegression classifier on the preprocessed training set.

📊 Results & Performance Evaluation
Test Accuracy: 80.45%

Confusion Matrix Breakdown
[[98 12]
 [23 46]]
True Negatives (98): Passengers who did not survive, correctly predicted as non-survivors.

False Positives (12): Passengers who did not survive, incorrectly predicted as survivors (Type I Error).

False Negatives (23): Passengers who survived, incorrectly predicted as non-survivors (Type II Error).

True Positives (46): Passengers who survived, correctly predicted as survivors.

Classification Metrics
Class 0 (Did Not Survive): Precision = 0.81, Recall = 0.89, F1-Score = 0.85

Class 1 (Survived): Precision = 0.79, Recall = 0.67, F1-Score = 0.72

Key Takeaways
The model shows higher recall for non-survivors (89% vs 67%) due to the underlying class imbalance in the training data.

Gender (Sex_male) and socio-economic status (Pclass) were the dominant signals driving passenger survival predictions.
