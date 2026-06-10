"""
1. Creacion de clusters y definicion de perfiles

A partir de las estimaciones individuales del CATE, se aplica un análisis de heterogeneidad mediante clustering. 

Para ello:
- Se estandarizan las variables de heterogeneidad (X)
- Se determina el número óptimo de grupos mediante el silhouette score
- Se utiliza un modelo de K-means para segmentar la muestra y se analizan los clusters obtenidos en función del CATE,
  las variables sociodemográficas y los mecanismos, para caracterizar perfiles diferenciados de impacto del teletrabajo.
"""

from CATE.CATE import M_cols
from CATE.CATE import X_cols
from CATE.CATE import W_cols
from CATE.CATE import df_model
from CATE.CATE import T_test
from CATE.CATE import Y_test
from CATE.CATE import X_test
from CATE.CATE import cate_test

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# HETEROGENEIDAD CAUSAL

# Definir variables para clusters
cluster_vars = X_cols
mechanism_vars = M_cols

# Crear dataframe base
df_test = df_model.loc[X_test.index, X_cols + W_cols + M_cols].copy()
df_test["teletrabajo"] = T_test
df_test["satisfaccion"] = Y_test
df_test["CATE"] = cate_test
print(df_test.head())

# Escalar
scaler = StandardScaler()
X_cluster = df_test[cluster_vars]
X_scaled = scaler.fit_transform(X_cluster)

scores = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append((k, score))
print("---------- Silhouette score ----------")
print(scores)

# CLUSTERING
kmeans = KMeans(
    n_clusters=4, 
    random_state=42,
    n_init=20       # evitar inestabilidad
)
df_test["cluster"] = kmeans.fit_predict(X_scaled)
print("---------- CLUSTERING ----------")
print(df_test.groupby("cluster")[cluster_vars].std())

# CATE POR CLUSTER
cate_cluster = df_test.groupby("cluster")["CATE"].agg(
    ["mean", "std", "count"]
)
print("---------- CATE POR CLUSTER ----------")
print(cate_cluster)

# PERFIL DE CLUSTERS
perfil = df_test.groupby("cluster")[cluster_vars + mechanism_vars + ["CATE"]].mean()
print("---------- PERFIL DE CLUSTERS ----------")
print(perfil)

# PERFIL DE MECANISMOS POR CLUSTES
mechanism_summary = df_test.groupby("cluster")[
    M_cols + ["CATE"]
].mean()
print("---------- PERFIL DE MECANISMOS POR CLUSTER ----------")
print(mechanism_summary)

# Correlacion entre mecanismos y CATE
corr = df_test[M_cols + ["CATE"]].corr()["CATE"]
print("---------- CORRELACION ENTRE MECANISMOS Y CATE ----------")
print(corr.sort_values(ascending=False))

# Comparacion entre treated y control
comparacion_cluster = df_test.groupby(
    ["cluster", "teletrabajo"]
)[M_cols + ["satisfaccion"]].mean()
print("---------- COMPARACION ENTRE TREATED Y CONTROL ----------")
print(comparacion_cluster)