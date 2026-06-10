"""
1.2. Análisis exploratorio de datos (EDA).

En este script se realiza una exploración inicial del conjunto de datos con el objetivo de comprender 
la distribución de las variables, detectar posibles anomalías y obtener una primera caracterización 
de la muestra estudiada.
"""

from ATE.preparacion_datos import df_model


# ANALISIS EXPLORATORIO DE DATOS (EDA)

print(df_model.head())
print(df_model.describe())

# Comprobar cuantas personas teletrabajan
print(df_model["teletrabajo"].value_counts())
print(df_model["teletrabajo"].value_counts(normalize=True))

# Distribucion
print(df_model["satisfaccion"].describe())

# Satisfaccion
print(df_model.groupby("teletrabajo")["satisfaccion"].mean())

# Comparacion de compatibilidad de covariables
def balance_summary(df):
    summary = df.groupby("teletrabajo").mean(numeric_only=True).T
    summary["diff"] = summary[1] - summary[0]
    return summary
print(balance_summary(df_model))
