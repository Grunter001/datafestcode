import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 1. Carga y preprocesamiento
train = pd.read_csv('train.csv')
categorical_cols = ['ocupacion', 'region', 'canal_adquisicion', 'banda_riesgo', 'dispositivo_principal']
encoder = LabelEncoder()
for col in categorical_cols:
    train[col] = encoder.fit_transform(train[col])

y = train['objetivo']

# --- ESCENARIO A: EXCLUYENDO 'mes' ---
X_sin_mes = train.drop(['id_cliente', 'mes', 'objetivo'], axis=1)
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(
    X_sin_mes, y, test_size=0.2, random_state=42, stratify=y
)

# LightGBM requiere especificar parámetros básicos para optimizar su velocidad
modelo_lgb_sm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
modelo_lgb_sm.fit(X_train_sm, y_train_sm)
pred_sm = modelo_lgb_sm.predict_proba(X_val_sm)[:, 1]
gini_sin_mes = 2 * roc_auc_score(y_val_sm, pred_sm) - 1

# --- ESCENARIO B: INCLUYENDO 'mes' ---
X_con_mes = train.drop(['id_cliente', 'objetivo'], axis=1)
X_train_cm, X_val_cm, y_train_cm, y_val_cm = train_test_split(
    X_con_mes, y, test_size=0.2, random_state=42, stratify=y
)

modelo_lgb_cm = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
modelo_lgb_cm.fit(X_train_cm, y_train_cm)
pred_cm = modelo_lgb_cm.predict_proba(X_val_cm)[:, 1]
gini_con_mes = 2 * roc_auc_score(y_val_cm, pred_cm) - 1

# --- RESULTADOS ---
print(f"Gini LightGBM (Excluyendo 'mes'): {gini_sin_mes:.4f}")
print(f"Gini LightGBM (Incluyendo 'mes'): {gini_con_mes:.4f}")