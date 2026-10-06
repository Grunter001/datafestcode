# Documentación Técnica: Predicción de Conversión de Clientes

Este documento detalla la metodología, experimentación y resultados comparativos de los modelos de *Machine Learning* evaluados para el proyecto de clasificación binaria, desarrollado para la asignatura de Análisis y Diseño de Algoritmos.

## 1. Metodología de Evaluación

Para garantizar la fiabilidad de las predicciones y evitar el sobreajuste (*overfitting*), el entrenamiento de todos los algoritmos se sometió a las siguientes reglas estandarizadas:

*   **Periodo de Análisis:** Los modelos fueron entrenados utilizando observaciones históricas registradas entre enero y noviembre de 2026.
*   **Estrategia de Partición (80/20):** Se utilizó la función `train_test_split` para aislar el 20% de los datos como conjunto de validación (simulación de examen), reservando el 80% para la fase de aprendizaje. La selección de las filas fue estrictamente aleatoria entre los 11 meses disponibles.
*   **Estratificación (`stratify=y`):** Se garantizó que la proporción de conversiones reales (clase minoritaria) se mantuviera matemáticamente idéntica tanto en el bloque de entrenamiento como en el de validación.
*   **Métrica de Rendimiento:** Coeficiente de Gini, derivado del Área Bajo la Curva ROC (`Gini = 2 * AUC - 1`).
*   **Variable de Estacionalidad:** Cada modelo fue evaluado en dos escenarios distintos: excluyendo e incluyendo la columna temporal `mes` para medir su impacto en la capacidad de generalización hacia el mes objetivo de diciembre de 2026.

## 2. Resumen Comparativo de Rendimiento

| Modelo Predictivo | Gini (Sin 'mes') | Gini (Con 'mes') | Diagnóstico Técnico |
| :--- | :--- | :--- | :--- |
| **Random Forest (100 árboles)** | 0.2112 | 0.2981 | Lidera la métrica, pero el salto drástico al incluir el mes indica un alto riesgo de sobreajuste al memorizar el periodo de entrenamiento. |
| **LightGBM (100 árboles)** | 0.2390 | **0.2411** | El modelo más estable y balanceado. Resiste el ruido temporal y ofrece la mejor capacidad de generalización segura. |
| **CatBoost Optimizado (1000 iteraciones)** | 0.2282 | 0.2391 | Muy competitivo tras corregir el subajuste, pero requiere mayor costo computacional sin superar a LightGBM. |
| **Regresión Logística (1000 iteraciones)** | 0.2287 | 0.2302 | Línea base matemática lineal; confirma la necesidad de usar modelos basados en árboles para capturar relaciones complejas. |
| **CatBoost (100 iteraciones)** | 0.2092 | 0.2194 | Subajuste severo (*underfitting*) debido a la falta de iteraciones para que el algoritmo secuencial optimice sus pesos. |

## 3. Fichas Técnicas de Experimentación

### Modelo 1: Random Forest Classifier
Ensamble paralelo basado en la construcción de múltiples árboles de decisión independientes.
*   **Preprocesamiento:** Transformación de variables textuales mediante `LabelEncoder`.
*   **Columnas excluidas:** `id_cliente` y `objetivo` (iterando la exclusión/inclusión de `mes`).
*   **Conjunto de Entrenamiento:** 80% de muestra aleatoria estratificada (Enero-Nov 2026).
*   **Configuración:** `n_estimators=100`, `random_state=42`.
*   **Gini Obtenido:** 0.2112 (Sin mes) / 0.2981 (Con mes).

### Modelo 2: LightGBM (LGBMClassifier)
Algoritmo de *Gradient Boosting* optimizado para ejecución rápida sobre datos tabulares.
*   **Preprocesamiento:** Transformación de variables textuales mediante `LabelEncoder`.
*   **Columnas excluidas:** `id_cliente` y `objetivo` (iterando la exclusión/inclusión de `mes`).
*   **Conjunto de Entrenamiento:** 80% de muestra aleatoria estratificada (Enero-Nov 2026).
*   **Configuración:** `n_estimators=100`, `random_state=42`.
*   **Gini Obtenido:** 0.2390 (Sin mes) / 0.2411 (Con mes).

### Modelo 3: Regresión Logística
Modelo paramétrico lineal utilizado como línea base algorítmica.
*   **Preprocesamiento:** Estandarización de escalas numéricas con `StandardScaler` y binarización de categorías de texto con `One-Hot Encoding` (`pd.get_dummies`).
*   **Columnas excluidas:** `id_cliente` y `objetivo` (iterando la exclusión/inclusión de `mes`).
*   **Conjunto de Entrenamiento:** 80% de muestra aleatoria estratificada (Enero-Nov 2026).
*   **Configuración:** `max_iter=1000`, `random_state=42`.
*   **Gini Obtenido:** 0.2287 (Sin mes) / 0.2302 (Con mes).

### Modelo 4: CatBoost Classifier (Inicial)
Ensamble secuencial diseñado para el manejo nativo de variables categóricas.
*   **Preprocesamiento:** Lectura nativa de columnas categóricas como *strings*, sin necesidad de codificadores externos.
*   **Columnas excluidas:** `id_cliente` y `objetivo` (iterando la exclusión/inclusión de `mes`).
*   **Conjunto de Entrenamiento:** 80% de muestra aleatoria estratificada (Enero-Nov 2026).
*   **Configuración:** `iterations=100`, `random_state=42`.
*   **Gini Obtenido:** 0.2092 (Sin mes) / 0.2194 (Con mes).

### Modelo 5: CatBoost Classifier (Optimizado)
Re-evaluación del algoritmo CatBoost incrementando el margen de aprendizaje.
*   **Preprocesamiento:** Lectura nativa de columnas categóricas como *strings*.
*   **Columnas excluidas:** `id_cliente` y `objetivo` (iterando la exclusión/inclusión de `mes`).
*   **Conjunto de Entrenamiento:** 80% de muestra aleatoria estratificada (Enero-Nov 2026).
*   **Configuración:** `iterations=1000`, `random_state=42`.
*   **Gini Obtenido:** 0.2282 (Sin mes) / 0.2391 (Con mes).