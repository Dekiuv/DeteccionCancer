# 1. Librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# 2. Cargar el dataset
df = pd.read_csv('data\merged_dummies.csv')

# 3. Definir X e y usando solo las mejores variables
X = df[['relapse', 'tumor_size', 'early_detection', 'sex', 'inflammatory_bowel_disease']]
y = df['survival_prediction']

# 4. Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Definir y entrenar el modelo
modelo = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

modelo.fit(X_train, y_train)

# 6. Evaluar
y_pred = modelo.predict(X_test)

print("\n🎯 Resultados finales:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 7. Guardar el modelo
joblib.dump(modelo, 'modelo_datos.pkl')
print("\n✅ Modelo guardado como 'modelo_cancer_colon_simple.pkl'")