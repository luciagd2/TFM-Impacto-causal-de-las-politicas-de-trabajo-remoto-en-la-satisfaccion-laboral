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
    "edad",
    "sexo",
    
    "sector",
    "ocupacion",
    "puesto",
    "tam_empresa",
    
    "horas_trabajadas",
    
    "antiguedad",
    
    "educacion",
    
    "num_hijos",
    
    "contrato_perm",
    
    "digitalizacion"
]

# Variables de heterogeneidad, utilizadas para descrubrir perfiles
X_cols = [
    "edad",
    "sexo",
    "tiene_hijos",
    "num_hijos",
    
    "sector",
    "ocupacion",
    "puesto",
    "tam_empresa",
    
    "horas_trabajadas",
    "horas_extra",
    
    "antiguedad",
    
    "educacion",
    
    "satisfaccion_ingresos",
    
###    "estres_laboral",
###    "motivacion_trabajo",
    
    "digitalizacion",

    "autonomia",

    "trabajo_fisico",   # 1: muy, 2: bastante, 3: poco, 4: nada fisico
    "paga_rendimiento",   # Pay includes performance related pay 
    "horario_habitual",  # horario en el que se trabaja normalmente
    "horario_fin_semana",  # normalmente trabaja findes de semana
    "seguridad_laboral", 
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

    # Modelos nuisance
    model_y=RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    ),

    model_t=RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    ),

    # Tratamiento binario
    discrete_treatment=True,

    # Configuración causal forest
    n_estimators=500,
    min_samples_leaf=20,
    max_depth=None,

    random_state=42
)

# Entrenamiento
cf_model.fit(
    Y_train,
    T_train,
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
df_test = pd.concat([df_test, W_test], axis=1)

df_test["teletrabajo"] = T_test
df_test["satisfaccion"] = Y_test
df_test["CATE"] = cate_test

# ANÁLISIS DE AUTONOMIA
# Grupos
df_test["grupo_autonomia"] = pd.qcut(
    df_test["autonomia"],
    q=3,
    labels=["Baja", "Media", "Alta"]
)
# CATE medio por grupo
cate_autonomia = df_test.groupby(
    "grupo_autonomia"
)["CATE"].agg(["mean", "std", "count"])
# Resultados
print(cate_autonomia)

#cate_autonomia["mean"].plot(kind="bar")
#plt.title("CATE medio por nivel de autonomía")
# plt.ylabel("CATE")
# plt.show()

# ANÁLISIS DE OCUPACION
# Comprobamos las ocupaciones con mas frecuencia
df_test = df_test.loc[:, ~df_test.columns.duplicated()]
df_test["ocupacion"].value_counts().head(10)

# Cogemos solo las cinco primeras
top_ocup = df_test["ocupacion"].value_counts().nlargest(5).index

df_test["ocupacion_simple"] = df_test["ocupacion"].apply(
    lambda x: x if x in top_ocup else "Otros"
)
# CATE medio por ocupacion
cate_ocup = df_test.groupby("ocupacion_simple")["CATE"].agg(
    ["mean", "std", "count"]
).sort_values("mean")
# Resultados
print(cate_autonomia)


# ANÁLISIS DE EDAD
# Grupos
df_test["grupo_edad"] = pd.qcut(
    df_test["edad"],
    q=3,
    labels=["Joven", "Mediana edad", "Mayor"]
)
# CATE medio por grupo
cate_edad = df_test.groupby("grupo_edad")["CATE"].agg(
    ["mean", "std", "count"]
)
# Resultados
print(cate_edad)


# ANALISIS DE TAMAÑO DE EMPRESA
# Grupos
df_test["grupo_empresa"] = pd.cut(
    df_test["tam_empresa"],
    bins=[0, 10, 50, 250, 100000],
    labels=["Micro", "Pequeña", "Mediana", "Grande"]
)
# CATE medio por grupo
cate_empresa = df_test.groupby("grupo_empresa")["CATE"].agg(
    ["mean", "std", "count"]
)
# Resultados
print(cate_empresa)


# ANALISIS DE ANTIGUEDAD
df_test["grupo_antiguedad"] = pd.qcut(
    df_test["antiguedad"],
    q=3,
    labels=["Baja", "Media", "Alta"]
)
cate_antiguedad = df_test.groupby("grupo_antiguedad")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_antiguedad)

# ANALISIS DE SATISFACCION INGRESOS
df_test["grupo_sat_ingresos"] = pd.qcut(
    df_test["satisfaccion_ingresos"],
    q=3,
    labels=["Baja", "Media", "Alta"]
)
cate_sat_ingresos = df_test.groupby("grupo_sat_ingresos")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_sat_ingresos)

# ANALISIS DE EDUCACION
df_test["grupo_educacion"] = df_test["educacion"].map({
    1: "Baja",
    2: "Media",
    3: "Alta"
})
cate_educacion = df_test.groupby("grupo_educacion")["CATE"].agg(["mean", "std", "count"])
print(cate_educacion)

# ANALISIS DE HIJOS
df_test["grupo_tiene_hijos"] = df_test["tiene_hijos"].apply(
    lambda x: "Con hijos" if x > 0 else "Sin hijos"
)
cate_tiene_hijos = df_test.groupby("grupo_tiene_hijos")["CATE"].agg(["mean", "std", "count"])
print(cate_tiene_hijos)

df_test["grupo_num_hijos"] = df_test["num_hijos"].map({
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4 o más"
})
cate_num_hijos = df_test.groupby("grupo_num_hijos")["CATE"].agg(["mean", "std", "count"])
print(cate_num_hijos) 

# ANALISIS DE SEXO
df_test["grupo_sexo"] = df_test["sexo"].map({
    1: "Hombre",
    2: "Mujer"
})
cate_sexo = df_test.groupby("grupo_sexo")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_sexo)

# ANALISIS DEL SECTOR
df_test["grupo_sector"] = df_test["sector"].map({
    1: "Privado",
    0: "Público"
})
cate_sector = df_test.groupby("grupo_sector")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_sector)

# ANALISIS DE CONTRATO PERMANENTE
df_test["grupo_contrato"] = df_test["contrato_perm"].map({
    1: "Permanente",
    0: "Temporal"
})
cate_contrato = df_test.groupby("grupo_contrato")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_contrato)

# ANALISIS DE PUESTO
df_test["grupo_puesto"] = df_test["puesto"].apply(
    lambda x: "Manager" if x == 1 else "No manager"
)
cate_puesto = df_test.groupby("grupo_puesto")["CATE"].agg(
    ["mean", "std", "count"]
)
print(cate_puesto)

# HORAS TRABAJADAS
df_test["grupo_horas_trabajadas"] = pd.qcut(
    df_test["horas_trabajadas"],
    q=3,
    labels=["Pocas", "Medias", "Muchas"]
)
cate_horas_trabajadas = df_test.groupby("grupo_horas_trabajadas")["CATE"].agg(["mean", "std", "count"])
print(cate_horas_trabajadas)

# HORAS EXTRA
bins = [-0.1, 0, 5, np.inf]
labels = ["No hace horas extra", "Pocas", "Muchas"]
df_test["grupo_horas_extra"] = pd.cut(
    df_test["horas_extra"],
    bins=bins,
    labels=labels
)
cate_horas_extra = df_test.groupby("grupo_horas_extra")["CATE"].agg(["mean", "std", "count"])
print(cate_horas_extra)

# DIGITALIZACION
df_test["grupo_digitalizacion"] = df_test["digitalizacion"].apply(
    lambda x: "No" if x == 0 else "Si"
)
cate_digitalizacion = df_test.groupby("grupo_digitalizacion")["CATE"].agg(["mean", "std", "count"])
print(cate_digitalizacion)

# TRABAJO FISICO
df_test["grupo_trabajo_fisico"] = df_test["trabajo_fisico"].map({
    1: "Muy físico",
    2: "Bastante físico",
    3: "Poco físico",
    4: "Nada físico"
})
cate_trabajo_fisico = df_test.groupby("grupo_trabajo_fisico")["CATE"].agg(["mean", "std", "count"])
print(cate_trabajo_fisico) 


# PAGA RELACIONADA CON EL RENDIMIENTO
df_test["grupo_paga_rendimiento"] = df_test["paga_rendimiento"].apply(
    lambda x: "Si" if x == 1 else "No"
)
cate_paga_rendimiento = df_test.groupby("grupo_paga_rendimiento")["CATE"].agg(
    ["mean", "std", "count"]
)
print("cate_paga_rendimiento")

# HORARIO HABITUAL
df_test["grupo_horario_habitual"] = df_test["horario_habitual"].map({
    1: "Diurno",
    2: "Nocturno",
    3: "Variable/turnos",
    4: "Otros"
})
cate_horario_habitual = df_test.groupby("grupo_horario_habitual")["CATE"].agg(["mean", "std", "count"])
print(cate_horario_habitual) 

# TRABAJA EN FIN DE SEMANA
df_test["grupo_horario_fin_semana"] = df_test["horario_fin_semana"].map({
    0: "Nunca",
    1: "Algunos",
    2: "Casi todos"
})
cate_horario_fin_semana = df_test.groupby("grupo_horario_fin_semana")["CATE"].agg(["mean", "std", "count"])
print(cate_horario_fin_semana) 

# SEGURIDAD LABORAL
df_test["grupo_seguridad_laboral"] = df_test["seguridad_laboral"].map({
    1: "Nada seguro",
    2: "Poco seguro",
    3: "Bastante seguro",
    4: "Muy seguro"
})
cate_seguridad_laboral = df_test.groupby("grupo_seguridad_laboral")["CATE"].agg(["mean", "std", "count"])
print(cate_seguridad_laboral) 