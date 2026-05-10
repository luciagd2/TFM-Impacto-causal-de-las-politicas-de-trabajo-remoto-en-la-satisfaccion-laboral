import numpy as np
import pandas as pd

from ATE.preparacion_datos import df_model

from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIGURACIÓN
# =========================
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

# =========================
# BOOTSTRAP LOOP
# =========================
for i in range(B):

    # 1. resampleo bootstrap
    df_b = df_model.sample(frac=1, replace=True, random_state=i)

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

# =========================
# RESULTADOS FINALES
# =========================
ate_array = np.array(ate_list)

ate_mean = np.mean(ate_array)
ate_std = np.std(ate_array)

ci_lower = np.percentile(ate_array, 2.5)
ci_upper = np.percentile(ate_array, 97.5)

print("\n=========================")
print("RESULTADO FINAL (ROBUSTO)")
print("=========================")

print(f"ATE medio: {ate_mean:.4f}")
print(f"Std: {ate_std:.4f}")
print(f"IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")





