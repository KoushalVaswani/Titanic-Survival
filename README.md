# Titanic - Survival Prediction & Model Comparison 🚢

This repository contains a Machine Learning project focused on predicting passenger survival on the Titanic shipwreck. The core objective of this project is to perform comprehensive **Data Preprocessing** and **Feature Scaling**, and then **compare the accuracy scores of multiple classification models** to find the best performing algorithm.

---

## 📌 Project Workflow

The project is structured into the following distinct phases:
1. **Exploratory Data Analysis (EDA):** Inspecting shapes, data types, and missing values using `sns.load_dataset("titanic")`.
2. **Data Cleaning & Feature Selection:** * Dropped redundant or leaky features (`deck`, `who`, `embark_town`, `alive`, `class`, `adult_male`).
   * Imputed missing values in the `age` column using its mean.
   * Dropped rows with missing values in the `embarked` column.
3. **Categorical Encoding:** Converted non-numeric columns (`sex`, `embarked`) into numerical format using `LabelEncoder`.
4. **Feature Scaling:** Applied `StandardScaler` to ensure distance-based models (like KNN and SVM) perform optimally.
5. **Model Training & Comparison:** Evaluated 5 different classification algorithms on the exact same test split ($20\%$).

---

## 📊 Model Evaluation & Comparison

Here is the comparative analysis of all the trained models based on their **Test Accuracy Scores**:

| Machine Learning Model | Feature Scaling Applied? | Test Accuracy | Status |
| :--- | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | Yes (`n_neighbors=35`) | **81.46%** | 🏆 **Best Performer** |
| **Support Vector Classifier (SVC)** | Yes (`kernel='rbf'`) | **81.46%** | 🏆 **Best Performer** |
| **Logistic Regression** | No | **80.33%** | Baseline Model |
| **Decision Tree Classifier** | Yes | **80.33%** | Baseline Model |
| **Gaussian Naive Bayes** | No | **77.52%** | Lowest Accuracy |

### 🔍 Key Insights from Comparison:
* **The Power of Tuning:** For the **KNN** model, trial and error was conducted for the hyperparameter `n_neighbors`. Setting it to `35` yielded the highest accuracy score of **81.46%**.
* **Distance & Margin Classifiers:** Both **KNN** and **SVM** (using the RBF kernel) tied for the highest accuracy ($81.46\%$), proving that scaling the features significantly helped distance-based boundaries.
* **Tree vs. Linear:** **Logistic Regression** and **Decision Tree** performed identically on this dataset, scoring **80.33%**.

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3
* **Platform:** Google Colab
* **Libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`

## 🚀 How to Run the Notebook
1. Clone this repository or download the `Titanic_Survival.ipynb` file.
2. Open the file in **Google Colab** or Jupyter Notebook.
3. Ensure you have `seaborn` and `scikit-learn` installed.
4. Run all the cells (`Runtime > Run all`) to reproduce the comparative results.
