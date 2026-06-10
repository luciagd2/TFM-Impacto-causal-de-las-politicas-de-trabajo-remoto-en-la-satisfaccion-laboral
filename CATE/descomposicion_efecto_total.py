"""
4.4. Descomposicion del efecto total
"""

from CATE.preparacion_datos_CATE import df_model
from CATE.CATE import cf_model
from CATE.CATE import X_cols
from CATE.CATE import W_cols
from CATE.CATE import M_cols
from CATE.CATE import X_test

import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# DESCOMPOSICION DEL EFECTO TOTAL (TE)
# Modelo base (sin mediadores)
te_base = cf_model.efecto(X_test).mean()

# Modelo con todos los mediadores
med_model = CausalForestDML(
    model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=20),
    model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=20),
    n_estimators=500,
    min_samples_leaf=20,
    random_state=42,
    discrete_treatment=True,
    cv=3
)

med_model.fit(
    Y=df_model["satisfaccion"],
    T=df_model["teletrabajo"],
    X=df_model[X_cols],
    W=df_model[W_cols + M_cols]
)

te_con_m = med_model.efecto(X_test).mean()

# Descomposicion simple
efecto_directo = te_con_m
efecto_indirecto = te_base - te_con_m
print("----- DESCOMPOSICION DEL EFECTO -----")
print("TE (sin mediadores):", te_base)
print("TE (con mediadores):", te_con_m)
print("Efecto mediado:", efecto_indirecto)
print("Efecto directo:", efecto_directo)

# Descomposicion por mediador
efecto_mediadores = {}

for mediador in M_cols:
    # todos menos uno
    mediadores_reducido = [m for m in M_cols if m != mediador]

    model = CausalForestDML(
        model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=20),
        model_y=RandomForestRegressor(n_estimators=200, min_samples_leaf=20),
        n_estimators=500,
        min_samples_leaf=20,
        random_state=42,
        discrete_treatment=True,
        cv=3
    )

    model.fit(
        Y=df_model["satisfaccion"],
        T=df_model["teletrabajo"],
        X=df_model[X_cols],
        W=df_model[W_cols + mediadores_reducido]
    )

    efecto = model.efecto(X_test).mean()

    efecto_mediadores[mediador] = efecto - efecto_directo
print("----- DESCOMPOSICION DEL EFECTO POR MEDIADOR -----")
print(efecto_mediadores)

nombres = list(efecto_mediadores.keys())
valores = list(efecto_mediadores.valores())

plt.figure(figsize=(10,5))
plt.bar(nombres, valores)

plt.axhline(0, linestyle="--")

plt.title("Contribución de cada mediador")
plt.ylabel("Contribución al efecto causal")

plt.xticks(rotation=45)
plt.show()
