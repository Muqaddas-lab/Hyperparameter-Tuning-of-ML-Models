# 🧠 Hyperparameter Tuning with Decision Tree (GUI Based)

This project showcases a graphical interface to perform **Hyperparameter Tuning** on a **Decision Tree Classifier** using both **Grid Search** and **Randomized Search**, with live result display and accuracy comparison graph. Built using Python and Tkinter.

---

## 📁 Dataset

We use the built-in **Breast Cancer Wisconsin Dataset** from `sklearn.datasets`:
- 569 samples
- 30 numeric features (e.g., radius, texture)
- Binary target: Malignant (0) or Benign (1)
- Automatically loaded in GUI (no need to upload manually)

---

## 🎯 Features

| Feature                       | Status  |
|------------------------------|---------|
| GUI using Tkinter            | ✅       |
| Decision Tree Classifier     | ✅       |
| GridSearchCV & RandomizedSearchCV | ✅ |
| Accuracy Comparison Graph    | ✅       |
| Real-time result display     | ✅       |

---

## 🔍 Hyperparameters Tuned

- `criterion`: `'gini'` or `'entropy'`
- `max_depth`: Various depth values for tree
- `min_samples_split`: Minimum samples needed for a split

---
