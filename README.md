# Impacto causal de las políticas de trabajo remoto en la satisfacción laboral

Este repositorio contiene el TFM para el Máster Universitario en Analítica de Negocion y Big Data, impartido por la universidad de Alcalá (UAH).

---

## Objetivo

El objetivo del proyecto es estudiar el efecto del teletrabajo sobre la satisfacción laboral desde una perspectiva causal, analizando:

- Efecto medio del tratamiento (**ATE**)
- Efectos heterogéneos individuales (**CATE**)
- Mecanismos causales intermedios
- Identificación de perfiles de trabajadores

---

## Metodología

El trabajo se estructura en tres bloques principales:

### Estimación del ATE
Se utilizan métodos de inferencia causal basados en:

- DoWhy (identificación causal mediante DAG)
- EconML (Double Machine Learning)
- Random Forest como modelos base

Tras los cuales se realizan una serie de validaciones.

---

### Estimación del CATE
Se emplea:

- **Causal Forest (EconML)**

Esto permite estimar efectos individuales del tratamiento en función de las características del trabajador.

A continuación se realiza una estructuración de variables en: variables de control, variables de heterogeneidad y mecanismos causales. 
A partir de aquí se realizan diversos tests y validaciones.

---

### Clustering de perfiles
Se aplica:

- K-Means clustering
- Selección de número de clusters mediante silhouette score
- Análisis de perfiles por grupo
- Validación estadística con Kruskal-Wallis

---

## 📈 Principales resultados

- El teletrabajo presenta un **efecto medio positivo moderado**
- Existe **heterogeneidad causal significativa entre trabajadores**
- La **autonomía** es el principal mecanismo explicativo
- Se identifican **perfiles diferenciados de impacto**
- Los resultados son robustos a múltiples validaciones

Para una explicación más detallada de la metodología, el desarrollo del modelo y la interpretación completa de los resultados, se recomienda consultar la memoria del proyecto.
