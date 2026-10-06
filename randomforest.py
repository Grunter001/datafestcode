import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Carga y preprocesamiento de datos base
train = pd.read_csv('train.csv')

categorical_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col])

# Variable objetivo compartida
y = train['objetivo']

# --- ESCENARIO A: EXCLUYENDO LA COLUMNA 'mes' ---
X_sin_mes = train.drop(['id_cliente', 'mes', 'objetivo'], axis=1)

X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(
    X_sin_mes, y, test_size=0.2, random_state=42, stratify=y
)

rf_sin_mes = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sin_mes.fit(X_train_sm, y_train_sm)
pred_sm = rf_sin_mes.predict_proba(X_val_sm)[:, 1]
gini_sin_mes = 2 * roc_auc_score(y_val_sm, pred_sm) - 1

# --- ESCENARIO B: INCLUYENDO LA COLUMNA 'mes' ---
# Solo se excluyen el identificador y la respuesta matemática
X_con_mes = train.drop(['id_cliente', 'objetivo'], axis=1)

X_train_cm, X_val_cm, y_train_cm, y_val_cm = train_test_split(
    X_con_mes, y, test_size=0.2, random_state=42, stratify=y
)

rf_con_mes = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_con_mes.fit(X_train_cm, y_train_cm)
pred_cm = rf_con_mes.predict_proba(X_val_cm)[:, 1]
gini_con_mes = 2 * roc_auc_score(y_val_cm, pred_cm) - 1

# --- RESULTADOS ---
print(f"Gini Random Forest (Excluyendo 'mes'): {gini_sin_mes:.4f}")
print(f"Gini Random Forest (Incluyendo 'mes'): {gini_con_mes:.4f}")