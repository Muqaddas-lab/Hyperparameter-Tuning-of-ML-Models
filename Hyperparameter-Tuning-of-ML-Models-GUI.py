import tkinter as tk
from tkinter import messagebox
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Global variables
acc_grid = 0
acc_random = 0
best_grid = {}
best_random = {}

# Load dataset (auto)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GUI Window
root = tk.Tk()
root.title("Hyperparameter Tuning GUI - Decision Tree")
root.geometry("700x500")
root.configure(bg="white")

result_label = tk.Label(root, text="", bg="white", font=("Arial", 12), justify="left")
result_label.pack(pady=20)

def run_tuning():
    global acc_grid, acc_random, best_grid, best_random

    model = DecisionTreeClassifier()

    # Grid Search
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, 15],
        'min_samples_split': [2, 4, 6]
    }
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    best_grid = grid.best_params_
    acc_grid = accuracy_score(y_test, grid.predict(X_test))

    # Randomized Search
    param_dist = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 20)),
        'min_samples_split': list(range(2, 10))
    }
    random = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=15, cv=5, random_state=42)
    random.fit(X_train, y_train)
    best_random = random.best_params_
    acc_random = accuracy_score(y_test, random.predict(X_test))

    result = f"🔍 Grid Search:\nBest: {best_grid}\nAccuracy: {acc_grid:.4f}\n\n"
    result += f"🎲 Random Search:\nBest: {best_random}\nAccuracy: {acc_random:.4f}"
    result_label.config(text=result)

def show_graph():
    methods = ['Grid Search', 'Random Search']
    accuracies = [acc_grid, acc_random]

    plt.figure(figsize=(6, 4))
    plt.bar(methods, accuracies, color=['skyblue', 'lightgreen'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Grid vs Random Search')
    plt.grid(axis='y')
    plt.show()

# Buttons
tk.Button(root, text="🚀 Run Tuning", command=run_tuning, bg="lightblue", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="📊 Show Accuracy Graph", command=show_graph, bg="lightgreen", font=("Arial", 12)).pack(pady=10)

root.mainloop()
