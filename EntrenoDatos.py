# ================================
# 1. Importar librerías necesarias
# ================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# ================================
# 2. Cargar el dataset
# ================================
df = pd.read_csv('data/merged_dummies.csv')

# ================================
# 3. Definir variables predictoras (X) y variable objetivo (y)
# ================================
X = df[['relapse', 'tumor_size', 'early_detection', 'sex', 'inflammatory_bowel_disease']]
y = df['survival_prediction']

# ================================
# 4. Dividir los datos en entrenamiento y prueba
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 5. Definir y entrenar el modelo XGBoost
# ================================
modelo = XGBClassifier(
    n_estimators=500,          # número de árboles
    max_depth=5,               # profundidad máxima de cada árbol
    learning_rate=0.05,        # tasa de aprendizaje
    subsample=0.8,             # fracción de muestras a usar en cada árbol
    colsample_bytree=0.8,      # fracción de características a usar por árbol
    gamma=1,                   # complejidad mínima para dividir un nodo
    min_child_weight=5,        # peso mínimo de muestra en hoja
    use_label_encoder=False,   # no usar codificador antiguo de etiquetas
    eval_metric='logloss',     # función de evaluación
    random_state=42            # semilla para reproducibilidad
)

modelo.fit(X_train, y_train)

# ================================
# 6. Evaluar el modelo
# ================================
y_pred = modelo.predict(X_test)

print("\n🎯 Resultados finales:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# ================================
# 7. Guardar el modelo entrenado
# ================================
joblib.dump(modelo, 'modelo_datos.pkl')
print("\n✅ Modelo guardado como 'modelo_datos.pkl'")