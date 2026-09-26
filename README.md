# MLOps SGSI - Gobierno Regional de Pasco

## Descripción

Proyecto de Machine Learning orientado al Sistema de Gestión de
Seguridad de la Información (SGSI) del Gobierno Regional de Pasco.

El modelo permite estimar la posible materialización de un riesgo
en un incidente de seguridad de la información a partir de
características relacionadas con activos, amenazas, vulnerabilidades,
probabilidad, impacto y tratamiento.

## Objetivo

Desarrollar un modelo de Machine Learning que, a partir de información
de activos, riesgos, amenazas, vulnerabilidades y tratamientos, permita
predecir la posible materialización de un riesgo en un incidente de
seguridad de la información.

## Variable objetivo

incidente_365_dias

- 0: No incidente
- 1: Incidente

## Variables de entrada

- asset_category
- asset_classification
- asset_criticality
- asset_tipo
- threat
- vulnerability
- probability
- impact
- treatment

## Modelos evaluados

- Regresión Logística
- K-Nearest Neighbors (KNN)
- Árbol de Decisión
- Random Forest

## Modelo seleccionado

Regresión Logística optimizada.

Hiperparámetros principales:

- C = 0.01
- class_weight = balanced

La selección considera especialmente Recall y F1-score debido a la
importancia de reducir falsos negativos en la identificación de
posibles incidentes de seguridad.

## Flujo MLOps

1. Entrada de datos.
2. Validación.
3. Preprocesamiento.
4. Ejecución del modelo.
5. Generación de predicción.
6. Almacenamiento de resultados.
7. Registro de ejecución mediante logs.
8. Documentación y reproducibilidad.

## Estructura del proyecto

mlops-sgsi/
- config/: configuración y metadatos del modelo.
- data/: dataset utilizado.
- logs/: registros de ejecución.
- metrics/: métricas de evaluación.
- models/: modelo serializado.
- output/: resultados de inferencia.

## Modelo serializado

models/modelo_sgsi_incidentes.joblib

El archivo contiene el Pipeline completo de Scikit-learn,
incluyendo el preprocesamiento y el clasificador.

## Resultado de la evaluación final

Modelo seleccionado: Regresión Logística

- Accuracy: 0.6457
- Precision: 0.6135
- Recall: 0.6306
- F1-score: 0.6219
- ROC-AUC: 0.6771

## Tecnologías

- Python
- Pandas
- Scikit-learn
- Joblib
- Google Colab
- Google Drive

## Organización objetivo

Gobierno Regional de Pasco (GRP).

## Autor

Proyecto académico de Automatización de Procesos.
