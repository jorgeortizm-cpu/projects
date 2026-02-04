"""
Entrenamiento y Evaluacion de Clasificadores - Social Network Ads
Modelos:
  1. Arbol de Decision
  2. SVM (con ajuste de kernel y C)
  3. Random Forest
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV

# --- Configuracion ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

sns.set_theme(style="whitegrid")

# --- Carga de datos preprocesados y escalados ---
FEATURES = ["Gender", "Age", "EstimatedSalary"]

train_df = pd.read_csv(os.path.join(RESULTS_DIR, "train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(RESULTS_DIR, "test_data_scaled.csv"))

X_train_scaled = train_df[FEATURES].values
y_train = train_df["Purchased"]
X_test_scaled = test_df[FEATURES].values
y_test = test_df["Purchased"]

print("=" * 60)
print("ENTRENAMIENTO Y EVALUACION DE CLASIFICADORES")
print("=" * 60)
print(f"Train: {X_train_scaled.shape[0]} muestras | Test: {X_test_scaled.shape[0]} muestras")


# =============================================
# FUNCION AUXILIAR: Evaluar y graficar modelo
# =============================================
def evaluar_modelo(nombre, modelo, y_real, y_pred, archivo_prefix):
    """Evalua un modelo y guarda la matriz de confusion."""
    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred)
    rec = recall_score(y_real, y_pred)
    f1 = f1_score(y_real, y_pred)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\n  Reporte de Clasificacion:")
    print(classification_report(y_real, y_pred, target_names=["No Compra", "Compra"]))

    # Matriz de confusion
    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Compra", "Compra"],
                yticklabels=["No Compra", "Compra"], ax=ax)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor Real")
    ax.set_title(f"Matriz de Confusion - {nombre}")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{archivo_prefix}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> {path}")

    return {"Modelo": nombre, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1-Score": f1}


# =============================================
# MODELO 1: ARBOL DE DECISION
# =============================================
print("\n" + "=" * 60)
print("MODELO 1: ARBOL DE DECISION")
print("=" * 60)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

res_dt = evaluar_modelo("Arbol de Decision", dt, y_test, y_pred_dt, "09_arbol_decision")

# Importancia de caracteristicas
print("  Importancia de caracteristicas:")
for feat, imp in zip(FEATURES, dt.feature_importances_):
    print(f"    {feat}: {imp:.4f}")

# Visualizacion del arbol
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt, feature_names=FEATURES,
          class_names=["No Compra", "Compra"],
          filled=True, rounded=True, ax=ax, fontsize=10)
ax.set_title("Arbol de Decision", fontsize=14)
plt.tight_layout()
tree_path = os.path.join(RESULTS_DIR, "10_arbol_decision_estructura.png")
plt.savefig(tree_path, dpi=150)
plt.close()
print(f"  -> {tree_path}")


# =============================================
# MODELO 2: SVM (con ajuste de kernel y C)
# =============================================
print("\n" + "=" * 60)
print("MODELO 2: SVM (con ajuste de kernel y C)")
print("=" * 60)

print("\n  Ejecutando GridSearchCV para encontrar mejores hiperparametros...")
param_grid_svm = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 100]
}

grid_svm = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid_svm, cv=5, scoring="accuracy", n_jobs=-1
)
grid_svm.fit(X_train_scaled, y_train)

print(f"\n  Mejores hiperparametros: {grid_svm.best_params_}")
print(f"  Mejor accuracy (CV):    {grid_svm.best_score_:.4f}")

# Resultados del GridSearch
print("\n  Resultados de todas las combinaciones:")
cv_results = pd.DataFrame(grid_svm.cv_results_)
for _, row in cv_results.iterrows():
    print(f"    kernel={row['param_kernel']:<8} C={str(row['param_C']):<6} "
          f"-> accuracy={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")

svm_best = grid_svm.best_estimator_
y_pred_svm = svm_best.predict(X_test_scaled)

res_svm = evaluar_modelo("SVM", svm_best, y_test, y_pred_svm, "11_svm")


# =============================================
# MODELO 3: RANDOM FOREST
# =============================================
print("\n" + "=" * 60)
print("MODELO 3: RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

res_rf = evaluar_modelo("Random Forest", rf, y_test, y_pred_rf, "12_random_forest")

# Importancia de caracteristicas
print("  Importancia de caracteristicas:")
for feat, imp in zip(FEATURES, rf.feature_importances_):
    print(f"    {feat}: {imp:.4f}")

# Grafico de importancia
fig, ax = plt.subplots(figsize=(8, 5))
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
importances.plot(kind="barh", color="#3498db", edgecolor="black", ax=ax)
ax.set_title("Importancia de Caracteristicas - Random Forest")
ax.set_xlabel("Importancia")
plt.tight_layout()
imp_path = os.path.join(RESULTS_DIR, "13_random_forest_importancia.png")
plt.savefig(imp_path, dpi=150)
plt.close()
print(f"  -> {imp_path}")


# =============================================
# COMPARACION FINAL DE MODELOS
# =============================================
print("\n" + "=" * 60)
print("COMPARACION FINAL DE MODELOS")
print("=" * 60)

resultados = pd.DataFrame([res_dt, res_svm, res_rf])
resultados = resultados.set_index("Modelo")
print(f"\n{resultados.to_string()}")

# Grafico comparativo
fig, ax = plt.subplots(figsize=(10, 6))
resultados_plot = resultados.reset_index()
metricas = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metricas))
width = 0.25

for i, (_, row) in enumerate(resultados_plot.iterrows()):
    valores = [row[m] for m in metricas]
    ax.bar(x + i * width, valores, width, label=row["Modelo"], edgecolor="black")

ax.set_ylabel("Score")
ax.set_title("Comparacion de Modelos - Metricas de Evaluacion")
ax.set_xticks(x + width)
ax.set_xticklabels(metricas)
ax.set_ylim(0.5, 1.05)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
comp_path = os.path.join(RESULTS_DIR, "14_comparacion_modelos.png")
plt.savefig(comp_path, dpi=150)
plt.close()
print(f"\n  -> {comp_path}")

# Curvas ROC
fig, ax = plt.subplots(figsize=(8, 6))
modelos_roc = [
    ("Arbol de Decision", dt),
    ("SVM", svm_best),
    ("Random Forest", rf)
]
for nombre, modelo in modelos_roc:
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = modelo.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curvas ROC - Comparacion de Modelos")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "15_curvas_roc.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"  -> {roc_path}")

# Mejor modelo
mejor = resultados["F1-Score"].idxmax()
print(f"\n  Mejor modelo segun F1-Score: {mejor} ({resultados.loc[mejor, 'F1-Score']:.4f})")

print("\n" + "=" * 60)
print("ENTRENAMIENTO Y EVALUACION COMPLETADOS")
print("=" * 60)
