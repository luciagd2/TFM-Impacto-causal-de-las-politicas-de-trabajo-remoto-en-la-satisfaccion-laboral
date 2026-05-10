from CATE.CATE import M_cols
from CATE.CATE import X_cols
from CATE.CATE import W_cols
from CATE.CATE import df_model
from CATE.CATE import cf_model
from CATE.CATE import X_test

import numpy as np

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# TEST DE ESTABILIDAD DEL EFECTO CAUSAL
effects = []

cf_model.fit(
    Y=df_model["satisfaccion"],
    T=df_model["teletrabajo"],
    X=df_model[X_cols],
    W=df_model[W_cols]
)
for i in range(30):  # baja a 30, suficiente
    df_b = resample(df_model, replace=True)

    effect = cf_model.effect(
        df_b[X_cols]
    ).mean()

    effects.append(effect)

print("----- TEST DE ESTABILIDAD DEL EFECTO CAUSAL -----")
print(np.mean(effects), np.std(effects))

# TEST DE ESTABILIDAD MEDIACION PARCIAL
mediators = M_cols

results = {}

for m in mediators:
    M_subset = [x for x in M_cols if x != m]
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
        W=df_model[W_cols + M_subset]
    )
    effect = model.effect(X_test).mean()
    results[m] = effect
print("----- TEST DE MEDIACION PARCIAL -----")
print(results)