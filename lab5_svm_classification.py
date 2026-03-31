# Lab 5 - SVM Classification with GridSearchCV
# Dataset: Wholesale Customers Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ── Load Data ──────────────────────────────────────────────────────────────────
column_names = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery',
                'Frozen', 'Detergents_Paper', 'Delicassen']

df = pd.read_csv("Wholesale_customers_data.csv", names=column_names)

# Create binary target: 0 = Horeca, 1 = Retail
df['target'] = (df['Channel'] > 1).astype(int)

print(f"Dataset shape: {df.shape}")
print(df.head())

# ── Data Preprocessing ────────────────────────────────────────────────────────
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

df = df.dropna()
print(f"Dataset shape after dropping missing values: {df.shape}")

# ── Target Distribution ───────────────────────────────────────────────────────
print("Target distribution:")
print(df['target'].value_counts())

df.info()

# ── Exploratory Data Analysis ─────────────────────────────────────────────────
sns.set_style('whitegrid')
sns.countplot(x='target', data=df)
plt.title('Channel Distribution (0=Horeca, 1=Retail)')
plt.xlabel('Target (0=Horeca, 1=Retail)')
plt.ylabel('Count')
plt.show()

# Box plots of all features by target
feature_cols = [col for col in df.columns if col not in ['target', 'Channel']]
for feature in feature_cols:
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f'{feature} by Channel')
    plt.show()

# Scatter plots
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
ax1.scatter(df['Fresh'], df['target'])
ax1.set_title("Channel vs Fresh Produce", fontsize=15)
ax1.set_xlabel('Fresh')
ax1.set_ylabel('Target')

ax2.scatter(df['Milk'], df['target'])
ax2.set_title("Channel vs Milk", fontsize=15)
ax2.set_xlabel('Milk')
plt.show()

# ── Train / Test Split ────────────────────────────────────────────────────────
df_feat = df.drop(['target', 'Channel'], axis=1)
df_target = df['target']

print("Features shape:", df_feat.shape)
print(df_feat.head())

X_train, X_test, y_train, y_test = train_test_split(
    df_feat, df_target, test_size=0.30, random_state=101)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ── Train Support Vector Classifier ──────────────────────────────────────────
model = SVC()
model.fit(X_train, y_train)

# ── Predictions and Evaluation ────────────────────────────────────────────────
predictions = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report:")
print(classification_report(y_test, predictions))

print("Misclassification error rate:",
      round(np.mean(predictions != y_test), 3))

# ── GridSearchCV — Round 1 ────────────────────────────────────────────────────
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best estimator:", grid.best_estimator_)

grid_predictions = grid.predict(X_test)

print("Confusion Matrix (after GridSearch):")
print(confusion_matrix(y_test, grid_predictions))

print("Classification Report (after GridSearch):")
print(classification_report(y_test, grid_predictions))

print("Misclassification error rate (after GridSearch):",
      round(np.mean(grid_predictions != y_test), 3))

# ── GridSearchCV — Round 2 (Refined Parameters) ───────────────────────────────
param_grid = {'C': [50, 75, 100, 125, 150],
              'gamma': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(tol=1e-5), param_grid, refit=True, verbose=1)
grid.fit(X_train, y_train)

print("Best estimator (refined):", grid.best_estimator_)

grid_predictions = grid.predict(X_test)

print("Confusion Matrix (refined GridSearch):")
print(confusion_matrix(y_test, grid_predictions))

print("\nClassification Report (refined GridSearch):")
print(classification_report(y_test, grid_predictions))
