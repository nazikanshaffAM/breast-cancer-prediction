# 🩺 Breast Cancer Mortality & Survival Prediction using Machine Learning

An end-to-end machine learning project aimed at predicting **mortality status** (alive/dead) and **survival months** of breast cancer patients using real-world clinical data. This project applies **classification and regression techniques** to support early intervention and personalized treatment planning in healthcare.

---

## 🎯 Project Objectives

- Predict **mortality status** of patients using classification algorithms.
- Predict **survival time (in months)** using regression models.
- Apply interpretable and accurate ML models to support **data-driven clinical decisions**.
- Evaluate the performance of models using appropriate metrics for healthcare use cases.

---

## 📌 Dataset

- Source: [SEER Breast Cancer Dataset](https://ieee-dataport.org/open-access/seer-breast-cancer-data) (via IEEE DataPort)
- Contains clinical and demographic information on breast cancer patients, including tumor stage, receptor status, and outcomes.

---

## 🧠 ML Models Applied

### Mortality Status Prediction (Classification)
- Logistic Regression (best model)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Voting Ensemble Classifier (LR + NB)

### Survival Months Prediction (Regression)
- Decision Tree Regressor (DT-1: fully grown, DT-2: pruned)

---

## 🔎 CRISP-DM Pipeline

1. **Business Understanding**  
   - Support clinical predictions using ML to improve care quality.

2. **Data Understanding**  
   - Identified key clinical and demographic predictors (e.g., tumor size, receptor status).

3. **Data Preparation**  
   - Cleaned data by handling nulls, standardizing categorical values, removing outliers using IQR.

4. **Modeling**  
   - Tuned hyperparameters using `GridSearchCV`, applied stratified train-test split for class balance.

5. **Evaluation**  
   - Classification: Accuracy, F1-score, AUC-ROC  
   - Regression: MAE, MSE, R²  
   - Interpretability and ethical considerations were addressed.

---

## 📊 Model Performance Summary

### 🔹 Classification (Mortality)

| Model           | Accuracy | F1-Score (Class 1) | AUC-ROC | Notes                          |
|----------------|----------|--------------------|---------|--------------------------------|
| Logistic Reg.   | 71%      | 0.42               | 0.74    | ✅ Best balanced performer      |
| Naive Bayes     | 73%      | 0.41               | 0.73    | High accuracy, lower recall    |
| KNN             | 85%      | 0.23               | 0.82    | Poor recall for mortality      |
| Ensemble (LR+NB)| 77%      | 0.37               | 0.76    | Improved minority class recall |

> **Logistic Regression** was selected as the best individual model due to higher recall and AUC-ROC for critical Class 1 (deceased).

---

### 🔹 Regression (Survival Months)

| Model         | MAE   | MSE    | R²     | Notes                                  |
|---------------|-------|--------|--------|----------------------------------------|
| DT-1 (Full)   | 26.35 | 1087.67| -1.14  | Overfitting; large error margins       |
| DT-2 (Pruned) | 18.95 | 591.89 | -0.16  | ✅ Best generalization; interpretable   |

> DT-2 (pruned) was selected due to lower MAE, simplicity, and better generalization.

---


---

## 📈 Key Highlights

- ✅ Applied **IQR-based outlier detection** and **label encoding**.
- ✅ Used **stratified train-test split** and **random_state** for reproducibility.
- ✅ Built **ensemble voting classifiers** to combine model strengths.
- ✅ Interpreted tree-based models to explain survival outcomes.
- ✅ Evaluated model reliability for high-stakes domains like healthcare.

---

## 📌 Limitations & Ethics

- Class imbalance impacted minority class performance.
- Logistic Regression offers interpretability but may oversimplify relationships.
- R² was low for regression due to missing latent clinical factors.
- Ethical implications around mortality prediction were acknowledged and addressed in the report.

---

## 🔧 Tools Used

- Python, Jupyter Notebook  
- Pandas, NumPy, Scikit-learn  
- Matplotlib, Seaborn  

---

## 📈 Visual Examples

*(Include screenshots or graphs here if uploading images)*

---

## 🚀 Future Work

- Deploy best models using **Flask**.
- Improve regression accuracy with additional clinical features.
- Integrate **SHAP/LIME** for explainability.
- Build a dashboard for healthcare professionals.

---

## 🙋‍♂️ Author

**Anshaff Ameer**  

---
