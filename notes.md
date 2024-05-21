KNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, recall_score, f1_score

# Cargar un conjunto de datos de ejemplo
data = load_iris()
X = data.data
y = data.target

# Lista de valores de k a probar
k_values = range(1, 31)

# Definir los scorers para recall y F1 Score
recall_scorer = make_scorer(recall_score, average='macro')
f1_scorer = make_scorer(f1_score, average='macro')

# Diccionarios para almacenar las métricas
cv_scores_5_recall = []
cv_scores_10_recall = []
cv_scores_5_f1 = []
cv_scores_10_f1 = []

# Realizar validación cruzada para cada valor de k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    scores_recall_5 = cross_val_score(knn, X, y, cv=5, scoring=recall_scorer)
    cv_scores_5_recall.append(scores_recall_5.mean())
    
    scores_recall_10 = cross_val_score(knn, X, y, cv=10, scoring=recall_scorer)
    cv_scores_10_recall.append(scores_recall_10.mean())
    
    scores_f1_5 = cross_val_score(knn, X, y, cv=5, scoring=f1_scorer)
    cv_scores_5_f1.append(scores_f1_5.mean())
    
    scores_f1_10 = cross_val_score(knn, X, y, cv=10, scoring=f1_scorer)
    cv_scores_10_f1.append(scores_f1_10.mean())

# Graficar las métricas para cada valor de k
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(k_values, cv_scores_5_recall, marker='o', label='Recall (CV=5)')
plt.plot(k_values, cv_scores_10_recall, marker='o', label='Recall (CV=10)')
plt.xlabel('Número de vecinos K')
plt.ylabel('Recall')
plt.title('Recall para diferentes valores de K')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(k_values, cv_scores_5_f1, marker='o', label='F1 Score (CV=5)')
plt.plot(k_values, cv_scores_10_f1, marker='o', label='F1 Score (CV=10)')
plt.xlabel('Número de vecinos K')
plt.ylabel('F1 Score')
plt.title('F1 Score para diferentes valores de K')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


===========


Random forest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Cargar un conjunto de datos de ejemplo
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el modelo
rf = RandomForestClassifier(random_state=42)

# Definir la rejilla de hiperparámetros para buscar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Configurar la búsqueda de la rejilla con validación cruzada
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Ajustar el modelo
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Evaluar el modelo con los mejores parámetros en el conjunto de prueba
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Informe de clasificación:\n", classification_report(y_test, y_pred))





========
xgboost

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Cargar un conjunto de datos de ejemplo
data = load_iris()
X = data.data
y = data.target

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el modelo
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Definir la rejilla de hiperparámetros para buscar
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Configurar la búsqueda de la rejilla con validación cruzada
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Ajustar el modelo
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Evaluar el modelo con los mejores parámetros en el conjunto de prueba
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)
print("Informe de clasificación:\n", classification_report(y_test, y_pred))