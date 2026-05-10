### CAMBIOS RESPECTO AL MODELO 3
# SE FIJA UNA SEMILLA

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
causal_graph_anterior = """
digraph {
    edad -> teletrabajo
    sexo -> teletrabajo
    educacion -> teletrabajo
    sector -> teletrabajo
    tam_empresa -> teletrabajo
    horas_trabajadas -> teletrabajo
    ocupacion -> teletrabajo
    salario -> teletrabajo
    contrato -> teletrabajo

    edad -> satisfaccion
    sexo -> satisfaccion
    educacion -> satisfaccion
    sector -> satisfaccion
    tam_empresa -> satisfaccion
    horas_trabajadas -> satisfaccion
    ocupacion -> satisfaccion
    salario -> satisfaccion
    contrato -> satisfaccion

    teletrabajo -> satisfaccion
}
"""

causal_graph = """
digraph {
    edad -> teletrabajo
    sexo -> teletrabajo
    educacion -> teletrabajo
    sector -> teletrabajo
    tam_empresa -> teletrabajo
    horas_trabajadas -> teletrabajo

    edad -> satisfaccion
    sexo -> satisfaccion
    educacion -> satisfaccion
    sector -> satisfaccion
    tam_empresa -> satisfaccion
    horas_trabajadas -> satisfaccion

    teletrabajo -> satisfaccion
}
"""

# Preparacion de variables
treatment = "teletrabajo"
outcome = "satisfaccion"
common_causes = [
    "edad",
    "sexo",
    "educacion",
    "horas_trabajadas",
    "sector",
    "tam_empresa",
    "puesto"
]

# SEMILLA
np.random.seed(42)
random.seed(42)








# Calculo del propensity score
X = df_model[common_causes]
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
lower = df_model[df_model["teletrabajo"]==1]["propensity"].quantile(0.05)
upper = df_model[df_model["teletrabajo"]==0]["propensity"].quantile(0.95)

df_trim = df_model[
    (df_model["propensity"] >= lower) &
    (df_model["propensity"] <= upper)
]











# MODELO DOWHY

# Creacion del objeto casual
model = CausalModel(
    data=df_trim,
    treatment=treatment,
    outcome=outcome,
    common_causes=common_causes,
    graph=causal_graph
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
            "model_t": RandomForestRegressor(n_estimators=100, random_state=42)
###            "model_t": RandomForestClassifier(n_estimators=100, random_state=42)

        },
        "fit_params": {}
    }
)

print("ATE:", estimate.value)
### print(estimate)




## BOOTSTRAP
B = 30  # número de bootstrap
ate_list = []

Y_col = "satisfaccion"
T_col = "teletrabajo"

X_cols = [
    "edad",
    "sexo",
    "educacion",
    "horas_trabajadas",
    "sector",
    "tam_empresa",
    "puesto",
]


# BUCLE
for i in range(B):

    # 1. resampleo bootstrap
    df_b = df_trim.sample(frac=1, replace=True, random_state=i)

    Y = df_b[Y_col].values.ravel()
    T = df_b[T_col].values.ravel()
    X = df_b[X_cols]

    # 2. modelo causal
    est = LinearDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestClassifier(n_estimators=100, random_state=42),
    discrete_treatment=True,
    random_state=42
)

    est.fit(Y, T, X=X)

    # 3. guardar ATE
    ate = est.ate(X)
    ate_list.append(ate)

    print(f"Iteración {i+1}/{B} - ATE: {ate:.4f}")

# RESULTADOS
ate_array = np.array(ate_list)

ate_mean = np.mean(ate_array)
ate_std = np.std(ate_array)

ci_lower = np.percentile(ate_array, 2.5)
ci_upper = np.percentile(ate_array, 97.5)

print("----- RESULTADO BOOTSTRAP -----")
print(f"ATE medio: {ate_mean:.4f}")
print(f"Std: {ate_std:.4f}")
print(f"IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")