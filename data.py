import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Cargar los datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Separar características (X) y la variable objetivo (y)
# Se excluyen 'id_cliente' y 'mes' porque son identificadores, no predictores.
X = train.drop(['id_cliente', 'mes', 'objetivo'], axis=1)
y = train['objetivo']

# Alinear las columnas de prueba
X_test = test.drop(['id_cliente', 'mes'], axis=1, errors='ignore')

# 3. Preprocesamiento básico: Codificar variables categóricas a números
categorical_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
encoder = LabelEncoder()

for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])
    X_test[col] = encoder.transform(X_test[col])

# 4. División de validación local (80% entrenamiento, 20% validación)
# stratify=y asegura que la proporción de conversiones se mantenga en ambas partes
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Definir y entrenar el modelo base
# n_jobs=-1 usa todos los núcleos del procesador para mayor velocidad
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)

# 6. Evaluar el modelo utilizando la métrica de la competencia (Gini)
# predict_proba[:, 1] obtiene la probabilidad de la clase 1 (conversión)
pred_val = modelo.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, pred_val)
gini = 2 * auc - 1
print(f"Gini de validación local: {gini:.4f}")

# 7. Generar las predicciones para test.csv y crear el archivo de entrega
test['prediccion'] = modelo.predict_proba(X_test)[:, 1]
submission = test[['id_cliente', 'prediccion']]
submission.to_csv('mi_primer_modelo.csv', index=False)