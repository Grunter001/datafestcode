import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Carga de datos
train = pd.read_csv('train.csv')
y = train['objetivo']

# 2. Preprocesamiento lineal: One-Hot Encoding para categorías
categorical_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
train_encoded = pd.get_dummies(train, columns=categorical_cols, drop_first=True)

# --- ESCENARIO A: EXCLUYENDO 'mes' ---
X_sin_mes = train_encoded.drop(['id_cliente', 'mes', 'objetivo'], axis=1)
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(
    X_sin_mes, y, test_size=0.2, random_state=42, stratify=y
)

# Nivelación de escalas numéricas
scaler_sm = StandardScaler()
X_train_sm_scaled = scaler_sm.fit_transform(X_train_sm)
X_val_sm_scaled = scaler_sm.transform(X_val_sm)

# max_iter=1000 asegura que la ecuación matemática converja sin errores
lr_sm = LogisticRegression(max_iter=1000, random_state=42)
lr_sm.fit(X_train_sm_scaled, y_train_sm)
pred_sm = lr_sm.predict_proba(X_val_sm_scaled)[:, 1]
gini_sin_mes = 2 * roc_auc_score(y_val_sm, pred_sm) - 1

# --- ESCENARIO B: INCLUYENDO 'mes' ---
X_con_mes = train_encoded.drop(['id_cliente', 'objetivo'], axis=1)
X_train_cm, X_val_cm, y_train_cm, y_val_cm = train_test_split(
    X_con_mes, y, test_size=0.2, random_state=42, stratify=y
)

scaler_cm = StandardScaler()
X_train_cm_scaled = scaler_cm.fit_transform(X_train_cm)
X_val_cm_scaled = scaler_cm.transform(X_val_cm)

lr_cm = LogisticRegression(max_iter=1000, random_state=42)
lr_cm.fit(X_train_cm_scaled, y_train_cm)
pred_cm = lr_cm.predict_proba(X_val_cm_scaled)[:, 1]
gini_con_mes = 2 * roc_auc_score(y_val_cm, pred_cm) - 1

# --- RESULTADOS ---
print(f"Gini Regresión Logística (Excluyendo 'mes'): {gini_sin_mes:.4f}")
print(f"Gini Regresión Logística (Incluyendo 'mes'): {gini_con_mes:.4f}")