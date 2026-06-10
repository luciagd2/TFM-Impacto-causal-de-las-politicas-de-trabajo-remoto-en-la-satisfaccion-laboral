"""
4. Creacion del modelo de Causal Forest y calculo del CATE

Se organizan las variables del modelo en tratamiento (T), resultado (Y), covariables de ajuste (W) y variables de heterogeneidad (X),
manteniendo separadas las variables mediadoras (M) para su análisis posterior. 

Se divide la muestra en conjuntos de entrenamiento y test para evaluar la estabilidad del modelo, y se estima el propensity score 
con el objetivo de comprobar el supuesto de soporte común mediante el análisis del solapamiento entre grupos tratados y de control. 

Finalmente, el efecto causal heterogéneo se estima mediante un modelo de Causal Forest, que permite capturar variaciones individuales 
en el efecto del tratamiento, y se analizan los resultados del CATE a través de estadísticos descriptivos (media, desviación estándar, 
mínimo y máximo) para evaluar la magnitud y dispersión de la heterogeneidad en la muestra.
"""

from CATE.preparacion_datos_CATE import df_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


# VARIABLES DEL MODELO

# Variable de tratamiento
T_col = "teletrabajo"

# Variable resultado
Y_col = "satisfaccion"

# Variables de ajuste, utilizadas para controlar la confusion
W_cols = [
    "sexo",
    "educacion",
    "ocupacion",
    "tam_empresa",
    "trabajo_fisico",
    "horario_habitual",
    "horario_fin_semana"
]

# Variables de heterogeneidad, utilizadas para descrubrir perfiles
X_cols = [
    "edad",
    "antiguedad",
    "tiene_hijos"
]

# Mediadores
M_cols = [
    "autonomia",
    "digitalizacion",
    "horas_trabajadas",
    "horas_extra",
    "satisfaccion_ingresos",
    "seguridad_laboral",
    "paga_rendimiento"
]

# CREACION DEL PIPELINE

# Matrices
Y = df_model[Y_col].values
T = df_model[T_col].values

X = df_model[X_cols]
W = df_model[W_cols]

# Train/Test
(
    X_train,
    X_test,
    W_train,
    W_test,
    Y_train,
    Y_test,
    T_train,
    T_test
) = train_test_split(
    X,
    W,
    Y,
    T,
    test_size=0.3,
    random_state=42
)

# INFORMACION
print("Tamaño train:", len(X_train))
print("Tamaño test:", len(X_test))

print("\nProporción teletrabajo (train):")
print(pd.Series(T_train).value_counts(normalize=True))

print("\nProporción teletrabajo (test):")
print(pd.Series(T_test).value_counts(normalize=True))


# PROPENSITY SCORE
ps_model = LogisticRegression(max_iter=1000)

ps_model.fit(W_train, T_train)

# Probabilidad de teletrabajo
ps_train = ps_model.predict_proba(W_train)[:, 1]
ps_test = ps_model.predict_proba(W_test)[:, 1]

print("Propensity score train: ", ps_train)
print("Propensity score test: ", ps_test)

# OVERLAP
plt.figure()

plt.hist(ps_train[T_train == 1], alpha=0.5, label="Tratados (train)")
plt.hist(ps_train[T_train == 0], alpha=0.5, label="Control (train)")

plt.title("Overlap del propensity score (train)")
plt.legend()
plt.show()
plt.close()

# REVISION DE LOS DATOS DE SOPORTE COMUN
print("Min propensity tratados:", ps_train[T_train == 1].min())
print("Max propensity control:", ps_train[T_train == 0].max())

# MODELO DE CAUSAL FOREST
cf_model = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=20),
    model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=20),
    
    n_estimators=500,
    min_samples_leaf=20,
    max_depth=10,
    random_state=42,
    
    # IMPORTANTES
    discrete_treatment=True,
    cv=3
)
# Entrenamiento
cf_model.fit(
    Y=Y_train,
    T=T_train,
    X=X_train,
    W=W_train
)


# Comprobacion basica
cate_test = cf_model.effect(X_test)

print("CATE_test:", cate_test[:10])
print("CATE medio:", np.mean(cate_test))

print("Std CATE:", np.std(cate_test))

print("Min:", np.min(cate_test))
print("Max:", np.max(cate_test))