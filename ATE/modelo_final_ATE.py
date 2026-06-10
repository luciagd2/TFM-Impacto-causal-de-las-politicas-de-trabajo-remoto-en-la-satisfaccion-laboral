"""
4. ATE

En este script se implementa la estimación del Average Treatment Effect (ATE) del teletrabajo sobre la satisfacción laboral. 

Para ello se construye un modelo causal basado en el DAG definido previamente, se verifica el cumplimiento de los principales 
supuestos de inferencia causal observacional y se estima el efecto medio mediante técnicas de Double Machine Learning (DML). 
Finalmente, se realizan validaciones y pruebas de robustez para evaluar la estabilidad y consistencia de los resultados obtenidos.
"""
from ATE.preparacion_datos import df_model

from dowhy import CausalModel
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import random

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Definicion del DAG
grafo_causal = """
digraph {
    teletrabajo -> salario
    teletrabajo -> autonomia
    
    edad -> salario
    edad -> satisfaccion

    horas_trabajadas -> salario
    horas_trabajadas -> satisfaccion
    
    tam_empresa -> salario
    tam_empresa -> autonomia
    tam_empresa -> puesto

    autonomia -> horas_trabajadas
    autonomia -> satisfaccion
    autonomia -> salario

    sexo -> satisfaccion
    sexo -> teletrabajo

    puesto -> autonomia
    puesto -> horas_trabajadas

    ocupacion -> teletrabajo
    ocupacion -> salario
     
    sector -> teletrabajo
    sector -> salario
    sector -> ocupacion
}
"""

# Preparacion de variables
tratamiento = "teletrabajo"
resultado = "satisfaccion"
variables_ajuste = [
    "edad",
    "sexo",
    "sector",
    "tam_empresa",
    "puesto"
]

# Semilla aleatorio
np.random.seed(42)
random.seed(42)


# Calculo del propensity score
X = df_model[variables_ajuste]
T = df_model["teletrabajo"]

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)

df_model["propensity"] = ps_model.predict_proba(X)[:, 1]


# Visualizacion del overlap
import matplotlib.pyplot as plt

plt.hist(df_model[df_model["teletrabajo"]==1]["propensity"], alpha=0.5, label="Teletrabajo")
plt.hist(df_model[df_model["teletrabajo"]==0]["propensity"], alpha=0.5, label="No teletrabajo")

plt.legend()
plt.title("Overlap del propensity score")
plt.show()

# Trimming - Eliminacion de regiones sin soporte comun
lower = max(
    df_model[df_model["teletrabajo"]==1]["propensity"].quantile(0.05),
    df_model[df_model["teletrabajo"]==0]["propensity"].quantile(0.05)
)

upper = min(
    df_model[df_model["teletrabajo"]==1]["propensity"].quantile(0.95),
    df_model[df_model["teletrabajo"]==0]["propensity"].quantile(0.95)
)

df_trim = df_model[
    (df_model["propensity"] >= lower) &
    (df_model["propensity"] <= upper)
]


# MODELO DOWHY
# Creacion del objeto casual
model = CausalModel(
    data=df_trim,
    treatment=tratamiento,
    outcome=resultado,
    common_causes=variables_ajuste,
    graph=grafo_causal
)

# Identificacion casual
identified_estimand = model.identify_effect()
### print(identified_estimand)


# Estimacion del efecto
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    method_params={
        "init_params": {
            "model_y": RandomForestRegressor(n_estimators=100, random_state=42),
            "model_t": RandomForestClassifier(n_estimators=100, random_state=42),
            "discrete_treatment": True,
        },
        "fit_params": {}
    }
)

print("ATE:", estimate.value)
### print(estimate)

## VALIDACIONES

# TEST PLACEBO
print("---------- TEST PLACEBO ----------")
placebo = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter"
)

print(placebo)

# SOLAPAMIENTO - PROPENSITY/SUPPORT
print("---------- PROPENSITY ----------")
X = df_model[variables_ajuste]
T = df_model["teletrabajo"]

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, T)

ps = ps_model.predict_proba(X)[:, 1]

print("Propensity score min treated:", ps[T==1].min())
print("Propensity score max control:", ps[T==0].max())

# TEST DE BALANCE
print("---------- TEST BALANCE ----------")
def smd(x_treated, x_control):
    mean_t = np.mean(x_treated)
    mean_c = np.mean(x_control)

    std_t = np.std(x_treated, ddof=1)
    std_c = np.std(x_control, ddof=1)

    pooled_std = np.sqrt((std_t**2 + std_c**2) / 2)

    return abs(mean_t - mean_c) / pooled_std

print("--- SMD ANTES DE TRIMMING ---")

for col in variables_ajuste:
    treated = df_model[df_model["teletrabajo"] == 1][col]
    control = df_model[df_model["teletrabajo"] == 0][col]

    print(col, round(smd(treated, control), 3))

print("--- SMD DESPUÉS DE TRIMMING ---")

for col in variables_ajuste:
    treated = df_trim[df_trim["teletrabajo"] == 1][col]
    control = df_trim[df_trim["teletrabajo"] == 0][col]

    print(col, round(smd(treated, control), 3))


# TEST DE SENSIBILIDAD
print("---------- TEST SENSIBILIDAD ----------")
estimate2 = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.LinearDML",
    method_params={
        "init_params": {
            "model_y": LassoCV(),
            "model_t": LassoCV()
        },
        "fit_params": {}
    }
)
print("ATE Lasso:", estimate2.value)


# MUESTRA DIVIDIDA
print("---------- MUESTRA DIVIDIDA ----------")
# Dividir la muestra
df_train, df_test = train_test_split(
    df_model,
    test_size=0.5,
    random_state=42
)
print("Tamaño train:", df_train.shape)
print("Tamaño test:", df_test.shape)


# Funcion para estimar el ATE
def estimate_ate(data):
    model = CausalModel(
        data=data,
        treatment=tratamiento,
        outcome=resultado,
        common_causes=variables_ajuste,
        graph=grafo_causal
    )
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        method_params={
            "init_params": {
                "model_y": RandomForestRegressor(n_estimators=100, random_state=42),
                "model_t": RandomForestRegressor(n_estimators=100, random_state=42)
            },
            "fit_params": {}
        }
    )
    return estimate.value

# Estimar en ambas muestras
ate_train = estimate_ate(df_train)
ate_test = estimate_ate(df_test)

# Resultados
print("ATE (train):", ate_train)
print("ATE (test):", ate_test)
print("Diferencia absoluta:", abs(ate_train - ate_test))



## BOOTSTRAP
# Configuracion
B = 30  # numero de bootstrap
ate_list = []

# Bootstrap Loop
for i in range(B):
    # 1. Resampleo Bootstrap
    df_b = df_trim.sample(frac=1, replace=True, random_state=i)

    Y = df_b[resultado].values.ravel()
    T = df_b[tratamiento].values.ravel()
    X = df_b[variables_ajuste]

    # 2. MModelo Causal
    est = LinearDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestClassifier(n_estimators=100, random_state=42),
    discrete_treatment=True,
    random_state=42
)

    est.fit(Y, T, X=X)

    # 3. Guardar ATE
    ate = est.ate(X)
    ate_list.append(ate)

    print(f"Iteración {i+1}/{B} - ATE: {ate:.4f}")

# RESULTADOS
ate_array = np.array(ate_list)

ate_mean = np.mean(ate_array)
ate_std = np.std(ate_array)

ci_lower = np.percentile(ate_array, 2.5)
ci_upper = np.percentile(ate_array, 97.5)

print("\n=========================")
print("RESULTADO BOOTSTRAP")
print("=========================")

print(f"ATE medio: {ate_mean:.4f}")
print(f"Std: {ate_std:.4f}")
print(f"IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")