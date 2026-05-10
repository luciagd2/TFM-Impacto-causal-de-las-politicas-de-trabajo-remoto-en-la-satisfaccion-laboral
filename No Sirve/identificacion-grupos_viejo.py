from CATE.preparacion_datos_CATE import df_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# VARIABLES DEL MODELO

# Variable de tratamiento
T_col = "teletrabajo"

# Variable resultado
Y_col = "satisfaccion"

# Variables de ajuste, utilizadas para controlar la confusion
W_cols = [
###    "edad",
###    "sexo",
###    "educacion",
###    "sector",
    "ocupacion",
###    "tam_empresa",
    "puesto",
    "antiguedad",
    "contrato_perm"
]

# Variables de heterogeneidad, utilizadas para descrubrir perfiles
X_cols = [
    "edad",
    "sexo",
    "num_hijos",
    "sector",
    "tam_empresa",
    "educacion"
]

# Mediadores
M_cols = [
    "autonomia",
    "satisfaccion_ingresos",
    "digitalizacion",
    "horas_trabajadas",
    "horas_extra",
    "trabajo_fisico",
    "seguridad_laboral",
    "motivacion_trabajo"
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
    Y=df_model["satisfaccion"],
    T=df_model["teletrabajo"],
    X=df_model[X_cols],         # heterogeneidad
    W=df_model[W_cols]          # confusores
)


# Comprobacion basica
cate_test = cf_model.effect(X_test)
print("CATE_test:", cate_test[:10])

print("CATE medio:", np.mean(cate_test))
print("Std CATE:", np.std(cate_test))

print("Min:", np.min(cate_test))
print("Max:", np.max(cate_test))


# FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "variable": X_train.columns,
    "importance": cf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance)

feature_importance.plot(
    x="variable",
    y="importance",
    kind="bar"
)

plt.title("Importancia causal de las variables")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.show()

# HETEROGENEIDAD CAUSAL

# Añadir el CATE al dataframe de test
df_test = X_test.copy()
### df_test = pd.concat([df_test, W_test], axis=1)

df_test["teletrabajo"] = T_test
df_test["satisfaccion"] = Y_test
df_test["CATE"] = cate_test



# Definir variables para clusters

# Variables heterogeneas principales
cluster_vars = [
    "tam_empresa",
    "educacion",
    "sector",
    "sexo",
    "num_hijos"
]

mechanism_vars = [
    "autonomia",
    "horas_extra",
    "horas_trabajadas",
    "digitalizacion",
    "trabajo_fisico",
    "seguridad_laboral",
    "motivacion_trabajo"
]


#-------------------------------------------------------------
# Estandarizar
scaler = StandardScaler()

X_cluster = scaler.fit_transform(df_test[cluster_vars])

#  Crear clusters
kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

df_test["cluster"] = kmeans.fit_predict(X_cluster)

# Analizar CATE por cluster
cluster_cate = (
    df_test
    .groupby("cluster")["CATE"]
    .agg(["mean", "std", "count"])
)

### print(cluster_cate)

#-------------------------------------------------------------

# Crear dataframe base
###df_test = X_test.copy()
###df_test = df_test.join(W_test)
df_test = df_model.loc[X_test.index, X_cols + W_cols + M_cols].copy()
df_test["CATE"] = cate_test
print(df_test.head())

X_cluster = df_test[cluster_vars]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42)
df_test["cluster"] = kmeans.fit_predict(X_scaled)

# CATE POR CLUSTER
cate_cluster = df_test.groupby("cluster")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_cluster)

# PERFIL DE CLUSTERS
perfil = df_test.groupby("cluster")[cluster_vars + mechanism_vars + ["CATE"]].mean()
print(perfil)