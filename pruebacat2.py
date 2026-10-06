import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# 1. Carga de datos
train = pd.read_csv('train.csv')
y = train['objetivo']

# 2. Definir variables categóricas
categorical_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
for col in categorical_cols:
    train[col] = train[col].astype(str)

# --- ESCENARIO A: EXCLUYENDO 'mes' ---
X_sin_mes = train.drop(['id_cliente', 'mes', 'objetivo'], axis=1)
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(
    X_sin_mes, y, test_size=0.2, random_state=42, stratify=y
)

# Incrementamos a 1000 iteraciones
cb_opt_sm = CatBoostClassifier(iterations=1000, random_state=42, thread_count=-1, verbose=False)
cb_opt_sm.fit(X_train_sm, y_train_sm, cat_features=categorical_cols)
pred_sm = cb_opt_sm.predict_proba(X_val_sm)[:, 1]
gini_sin_mes = 2 * roc_auc_score(y_val_sm, pred_sm) - 1

# --- ESCENARIO B: INCLUYENDO 'mes' ---
X_con_mes = train.drop(['id_cliente', 'objetivo'], axis=1)
X_train_cm, X_val_cm, y_train_cm, y_val_cm = train_test_split(
    X_con_mes, y, test_size=0.2, random_state=42, stratify=y
)

cb_opt_cm = CatBoostClassifier(iterations=1000, random_state=42, thread_count=-1, verbose=False)
cb_opt_cm.fit(X_train_cm, y_train_cm, cat_features=categorical_cols)
pred_cm = cb_opt_cm.predict_proba(X_val_cm)[:, 1]
gini_con_mes = 2 * roc_auc_score(y_val_cm, pred_cm) - 1

# --- RESULTADOS ---
print(f"Gini CatBoost Optimizado (Excluyendo 'mes'): {gini_sin_mes:.4f}")
print(f"Gini CatBoost Optimizado (Incluyendo 'mes'): {gini_con_mes:.4f}")