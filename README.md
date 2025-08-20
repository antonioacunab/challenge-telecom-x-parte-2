# Telecom X - Parte 2: Modelado Predictivo de Churn 📊

Este proyecto se enfoca en el análisis y la creación de modelos de Machine Learning para predecir la cancelación de clientes (churn) en una empresa de telecomunicaciones. El objetivo es identificar los factores clave que influyen en la decisión de un cliente de abandonar el servicio y construir un modelo que pueda anticipar este comportamiento de manera proactiva.

## 1\. Propósito del Análisis 🎯

El objetivo principal de este análisis es **reducir la pérdida de clientes** mediante la identificación temprana de aquellos con alta probabilidad de cancelar su contrato. Al comprender los patrones y las variables que preceden al churn, la empresa puede diseñar e implementar **estrategias de retención** más efectivas y personalizadas, optimizando así los recursos y mejorando la lealtad del cliente.

## 2\. Estructura del Proyecto 📂

El proyecto está organizado de la siguiente manera para facilitar su comprensión y reproducibilidad:

```
telecom-x-parte-2/
│
├── 📄 TelecomX_Parte2.ipynb        # Cuaderno principal con todo el análisis y modelado.
├── 📊 datos_tratados.csv           # Conjunto de datos limpios y preprocesados.
├── 🖼️ coeficientes_rl.png/         # Carpeta para guardar los gráficos generados.
├── 🖼️ corrs_churn.png/             # Carpeta para guardar los gráficos generados.
├── 🖼️ importances_rf.png/          # Carpeta para guardar los gráficos generados.
├── 🖼️ matriz_confusion_rl.png/     # Carpeta para guardar los gráficos generados.
└── 📖 README.md                    # Este archivo.
```

## 3\. Proceso de Preparación de Datos ⚙️

La preparación de los datos fue una etapa crucial para asegurar el rendimiento y la validez de los modelos.

#### Clasificación y Codificación de Variables

  - Las variables se clasificaron en **numéricas** (como `Tenure`, `MonthlyCharges`) y **categóricas** (como `Contract`, `PaymentMethod`).
  - Las variables categóricas con valores binarios (Sí/No) o múltiples categorías fueron transformadas a un formato numérico mediante **One-Hot Encoding** con la función `pd.get_dummies`. Esto es esencial para que los algoritmos de Machine Learning puedan procesarlas.

#### Balanceo de Clases (SMOTE)

  - El análisis exploratorio reveló un fuerte desbalance de clases: solo el **26.5%** de los clientes habían cancelado, frente a un **73.5%** que permanecían activos.
  - Para evitar que los modelos se sesgaran hacia la clase mayoritaria, se aplicó la técnica de sobremuestreo **SMOTE** (Synthetic Minority Over-sampling TEchnique) **únicamente sobre el conjunto de entrenamiento**.

#### Estandarización de Datos

  - Para el modelo de Regresión Logística, que es sensible a la escala de las variables, se aplicó la **estandarización** (`StandardScaler`). Este proceso transforma los datos para que tengan una media de 0 y una desviación estándar de 1, asegurando que ninguna variable domine a las demás solo por su magnitud.

#### Separación en Conjuntos de Entrenamiento y Prueba

  - Los datos se dividieron en un conjunto de **entrenamiento (70%)** y uno de **prueba (30%)**.
  - Se utilizó la estratificación (`stratify=y`) para garantizar que la proporción de clientes cancelados y activos fuera la misma en ambos conjuntos, permitiendo una evaluación más robusta del modelo.

## 4\. Justificaciones de Modelado 🧠

Se eligieron dos modelos con características diferentes para abordar el problema desde distintos ángulos:

1.  **Regresión Logística:** Se seleccionó por su alta interpretabilidad. Sus coeficientes nos permiten entender claramente el impacto (positivo o negativo) de cada variable en la probabilidad de cancelación. Requiere datos estandarizados.
  ![Correlaciones Modelo de Regresión Logística](https://github.com/antonioacunab/challenge-telecom-x-parte-2/blob/main/coeficientes_rl.png?raw=true)
  
2.  **Random Forest:** Se eligió por su alto rendimiento y robustez. Al ser un modelo basado en árboles, no requiere estandarización y es capaz de capturar relaciones no lineales complejas entre las variables.
  ![Importancias de las variables en modelo Random Forest](https://github.com/antonioacunab/challenge-telecom-x-parte-2/blob/main/importances_rf.png?raw=true)

El modelo final recomendado fue la **Regresión Logística** debido a su superior **Recall (76%)**, lo que lo hace más efectivo para el objetivo de negocio de identificar a la mayor cantidad posible de clientes en riesgo.
![Matriz de confusión de Regresión Logística](https://github.com/antonioacunab/challenge-telecom-x-parte-2/blob/main/matriz_confusion_rl.png?raw=true)

## 5\. Insights del Análisis Exploratorio (EDA) ✨

El análisis exploratorio reveló patrones claros y consistentes, confirmados posteriormente por los modelos.

  ![Correlaciones Modelo de Regresión Logística](https://github.com/antonioacunab/challenge-telecom-x-parte-2/blob/main/corrs_churn.png?raw=true)

  - **La Antigüedad es Clave:** Los clientes que cancelan tienden a hacerlo en los primeros meses de su contrato. La probabilidad de cancelación disminuye drásticamente a medida que aumenta la antigüedad.

  - **El Tipo de Contrato es Decisivo:** Los clientes con contratos mes a mes (`MonthToMonth`) son, por mucho, los más propensos a cancelar. Los contratos anuales o de dos años actúan como un fuerte factor de retención.

## 6\. Instrucciones de Ejecución 🚀

Para ejecutar este proyecto en un entorno local, sigue los siguientes pasos:

#### 6.1. Prerrequisitos (Bibliotecas)

Asegúrate de tener instaladas las siguientes bibliotecas de Python. Puedes instalarlas usando pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

#### 6.2. Ejecución

1.  Clona o descarga este repositorio en tu máquina local.
2.  Abre el cuaderno `telecom_churn_prediction.ipynb` en un entorno como Jupyter Notebook, JupyterLab o Google Colab.
3.  El cuaderno carga los datos directamente desde una URL, por lo que no es necesario descargar el archivo `datos_tratados.csv` por separado.
4.  Ejecuta las celdas del cuaderno en orden secuencial para replicar todo el análisis, desde la carga de datos hasta la evaluación final de los modelos.
