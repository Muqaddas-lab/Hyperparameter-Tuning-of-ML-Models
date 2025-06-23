# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the model
model = DecisionTreeClassifier()

# Step 5: Grid Search Parameters
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 4, 6]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
acc_grid = accuracy_score(y_test, y_pred)

# Step 6: Randomized Search Parameters
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(2, 20)),
    'min_samples_split': list(range(2, 10))
}
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=15, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)
y_pred_random = random_search.predict(X_test)
acc_random = accuracy_score(y_test, y_pred_random)

# Step 7: Print Results
print("Best Parameters (Grid Search):", grid_search.best_params_)
print("Grid Search Accuracy:", acc_grid)
print("Best Parameters (Random Search):", random_search.best_params_)
print("Random Search Accuracy:", acc_random)

# Step 8: Plot
methods = ['Grid Search', 'Random Search']
accuracies = [acc_grid, acc_random]

plt.figure(figsize=(6, 4))
plt.bar(methods, accuracies, color=['orange', 'green'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Breast Cancer Dataset: Hyperparameter Tuning Accuracy')
plt.grid(axis='y')
plt.show()
