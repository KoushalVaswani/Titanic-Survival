# Titanic Survival Prediction 🚢

This repository contains a comprehensive Machine Learning project that predicts passenger survival on the Titanic using the classic Titanic dataset from Seaborn. The project covers the entire data science pipeline: exploratory data cleaning, preprocessing, feature scaling, automated hyperparameter tuning, and comparing multiple classification models using both a standard Train-Test Split and 5-Fold Cross-Validation.

---

## 📌 Project Overview
The primary objective of this project is to analyze passenger data from the Titanic disaster and build a machine learning model capable of predicting whether a passenger survived (`1`) or not (`0`). 

---

## 🛠️ Data Preprocessing & Cleaning
To ensure high-quality inputs for our machine learning algorithms, the following preprocessing steps were performed:
- **Feature Selection:** Dropped redundant, repetitive, or low-correlation columns to prevent overfitting:
  - `deck`, `who`, `embark_town`, `alive`, `class`, and `adult_male`.
- **Missing Value Imputation:** 
  - Filled missing values in the `age` column using the dataset's **mean age**.
  - Dropped rows with missing values in the `embarked` column since they were negligible.
- **Categorical Encoding:** Converted text-based categorical columns (`sex` and `embarked`) into numerical format using `LabelEncoder`.
- **Data Type Standardization:** Converted the final dataframe to integer type for consistency.
- **Feature Scaling:** Applied `StandardScaler` to normalize features, which is strictly required for distance-based and boundary-based models like KNN and SVM.

---

## 🤖 Models Evaluated
We trained and evaluated five different classification algorithms to find the most optimal solution:
1. **Logistic Regression** (Baseline statistical model)
2. **Gaussian Naive Bayes (GaussianNB)** (Probabilistic classifier)
3. **Decision Tree Classifier** (Tree-based non-linear model)
4. **K-Nearest Neighbors (KNN)** (Distance-based local learner)
5. **Support Vector Machine (SVM)** (Boundary-based global learner)
6. **Random Forest Classifier** (Ensemble tree-based learner)

---

## ⚙️ Hyperparameter Tuning (GridSearchCV)
To squeeze out the maximum performance from our best-performing algorithm, we implemented **GridSearchCV** for automated hyperparameter optimization on the Support Vector Classifier. 

The grid search evaluated various combinations of regularization parameters (`C`), kernel coefficients (`gamma`), and kernel types, revealing the most optimal configuration:
* **Best Parameters Found:** `{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}`

---

## 📊 Final Model Performance Comparison

To understand the true generalizability of our models and eliminate the element of "luck" from a single random split, we applied **5-Fold Cross-Validation** alongside automated tuning.

Here is the complete breakdown of how the models performed:

| Model Name | Single Split Accuracy (80-20) | 5-Fold CV Mean Accuracy | Status / Remark |
| :--- | :---: | :---: | :--- |
| **Optimized SVM (Tuned via GridSearch)** | **82.02%** | **82.79%** | 🏆 **Winner** (Highly stable, optimized boundary) |
| **Support Vector Machine (Baseline)** | 81.46% | 82.79% | Robust global learner |
| **K-Nearest Neighbors (KNN)** | 81.46% | 80.31% | 📉 Accuracy dropped due to local noise sensitivity |
| **Logistic Regression** | 80.33% | *Not Evaluated* | Standard baseline performance |
| **Decision Tree Classifier** | 80.33% | *Not Evaluated* | Standard baseline performance |
| **Random Forest Classifier** | 80.33% | *Not Evaluated* | Default ensemble baseline |
| **Gaussian Naive Bayes** | 77.52% | *Not Evaluated* | Lowest baseline performance |

---

## 🔍 Key Insights & Technical Takeaways

1. **The Train-Test Split Illusion:** On a simple 80-20 random split, both **KNN (with $K=35$)** and **SVM** scored an identical **81.46%**. This made them look equally good initially.
2. **Why KNN Dropped on Cross-Validation ($81.46\% \rightarrow 80.31\%$):** KNN is a *local learner* that relies strictly on the closest data points (neighbors). The Titanic dataset contains noise (e.g., wealthy passengers who didn't survive or lower-class passengers who did). When tested across different folds, KNN's local boundaries fluctuated significantly due to this noise, revealing its true lower average accuracy.
3. **Why SVM Won ($81.46\% \rightarrow 82.79\%$):** SVM creates a *global decision boundary (hyperplane)* using an `rbf` kernel. Instead of getting distracted by individual noisy points, it focuses on the most critical boundary points (Support Vectors). This allowed it to remain highly stable and perform even better when evaluated across all folds.
4. **The Power of Tuning ($81.46\% \rightarrow 82.02\%$):** By utilizing `GridSearchCV` to systematically find the sweet spot for `C=1` and `gamma=0.1`, the test accuracy of the final deployment-ready model successfully pushed past the baseline to hit a peak of **82.02%**.

---

## 🚀 How to Run the Notebook
1. Clone this repository to your local machine or open it directly in Google Colab.
2. Ensure you have the required Python libraries installed:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
