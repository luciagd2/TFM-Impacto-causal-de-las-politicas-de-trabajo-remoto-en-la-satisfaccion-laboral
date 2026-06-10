"""
1. Preparación de datos para el análisis del ATE.

En este script se realizan las tareas necesarias para construir el conjunto de datos.
Entre ellas se incluyen:
- La carga de los datos
- La selección de variables relevantes
- El tratamiento de valores perdidos
- La recodificación de variables categóricas 

El objetivo es obtener una muestra consistente y adecuada para la estimación del efecto causal medio del teletrabajo sobre la satisfacción laboral,
reduciendo posibles problemas de calidad de datos y garantizando la reproducibilidad del análisis posterior.
"""

import pandas as pd
import numpy as np

## CARGA DE DATOS 
df = pd.read_stata(
    r"C:\Users\Lucia\OneDrive\Documentos\Uni-Pc_Lucia\Master\TFM\Dataset\UKDA-6614-stata\stata\stata14_se\ukhls\n_indresp.dta", 
    convert_categoricals=False
    )


## SELECCIÓN DE VARIABLES

# Variables que formaran las columnas del dataset
df_model = df[[
    # Tratamiento - Teletrabajo
    "n_wkhome",
    
    # Outcome - Satisfaccion laboral global
    "n_jbsat",
    
    # Variables demográficas
    "n_pdvage",    # edad
    "n_sex",       # sexo
    
    # Estructura laboral
    "n_jbhrs",     # horas trabajadas a la semana
    "n_jbsect",    # sector público o privado
    "n_jbsize",    # tamaño de la empresa
    "n_jbmngr",    # puesto de trabajo
    "n_jbseg_dv",  # grupo socioeconomico al que pertenece (ocupacion)
    "n_basrate",   # salario por hora

    # Variables de autonomia
    "n_wkaut1",    # autonomia sobre las tareas a realizar
    "n_wkaut2",    # autonomia sobre el ritmo de trabajo
    "n_wkaut3",    # autonomia sobre como hacer el trabajo
    "n_wkaut4",    # autonomia sobre el orden para realizar las tareas
    "n_wkaut5",    # autonomia sobre las horas de trabajo

    # Variables descartadas
###    "n_qfhigh",    # nivel de educacion mas alto completado -> demasiada perdida de observaciones
###    "n_paytyp",    # tipo de contrato -> demasiada perdida de observaciones
       # Flexibilidad horaria -> redundantes o demasiado relacionadas con la variable de tratamiento (teletrabajo)
###    "n_jbflex7",   # horarios flexibles: trabajar desde casa regularmente
###    "n_jbfxuse7",  # uso del trabajo flexible
       # Variables de tecnologia -> demasiado relacionadas con la variable de tratamiento (teletrabajo)
###    "n_wktechinet1",    # uso de ordenadores de mesa en el trabajo
###    "n_wktechinet2",    # uso deportatiles en el trabajo
###    "n_wktechinet3",    # uso de telefonos en el trabajo
###    "n_wktechinet4",    # uso de tablets en el trabajo
###    "n_wktechinet5",    # uso de 'feature phone/non-touchscreen mobilephone' en el trabajo
###    "n_wktechinet6",    # uso de 'handheld device/PDA' en el trabajo
###    "n_wkaut6",    # autonomia sobre el lugar de trabajo -> redundante con la variable de tratamiento (teletrabajo)
]]


## RENOMBRADO DE VARIABLES
df_model = df_model.rename(columns={
    "n_wkhome": "teletrabajo",
    "n_jbsat":  "satisfaccion",
    "n_pdvage": "edad",
    "n_sex":    "sexo",
    "n_jbhrs":  "horas_trabajadas",
    "n_jbsect": "sector",
    "n_jbsize": "tam_empresa",
    "n_jbmngr": "puesto", 
    
    "n_wkaut1": "aut_tareas", 
    "n_wkaut2": "aut_ritmo", 
    "n_wkaut3": "aut_metodo",
    "n_wkaut4": "aut_orden",
    "n_wkaut5": "aut_horas", 

    "n_jbseg_dv": "ocupacion",
    "n_basrate":  "salario"
})


## LIMPIEZA DE VALORES MISSING

# -9 = missing, -8 = inapplicable, -7 = proxy, -2 = refusal, -1 = don't know
missing_values = [-9, -8, -7, -2, -1]

# Conversion a float
df_model = df_model.astype(float)

# Reemplazar valores por nan
df_model = df_model.replace(missing_values, np.nan)



## TRATAMIENTO DE VARIABLES

# TELETRABAJO (TRATAMIENTO)

# Eliminacion de los datos con valor wkhome = 8 (Variable) ya que es demasiado ambigüa
df_model = df_model[df_model["teletrabajo"] != 8]

# Convertir a binaria: 1 = teletrabaja, 0 = no o muy poco
df_model["teletrabajo"] = df_model["teletrabajo"].apply(
    lambda x: 1 if x >= 3 else 0
)


# SATISFACCIÓN (RESULTADO)
df_model["satisfaccion"] = df_model["satisfaccion"].astype(float)


# SECTOR
# 1 sector privado, 0 sector publico
df_model["sector"] = df_model["sector"].apply(
    lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
)


# PUESTO DE MANAGER
df_model["puesto"] = df_model["puesto"].apply(
    lambda x: 1 if x == 1 else (0 if x in [2,3] else np.nan)
)


# AUTONOMÍA
aut_cols = [
    "aut_tareas",
    "aut_ritmo",
    "aut_metodo",
    "aut_orden",
    "aut_horas"
]
# skipna=True permite calcular la media ignorando valores nulos parciales, evitando la perdida de observaciones
df_model["autonomia"] = df_model[aut_cols].mean(axis=1, skipna=True) 
# Eliminacion de las columnas usadas para crear esta ultima
df_model = df_model.drop(columns=aut_cols)


# TAMAÑO EMPRESA
# Asegurar el tipo de datos
df_model["tam_empresa"] = df_model["tam_empresa"].astype(float)


# SALARIO
# Eliminacion de outliers extremos
df_model = df_model[df_model["salario"] < df_model["salario"].quantile(0.99)]


## ELIMINACIÓN DE NULOS
print("# de observaciones antes de replace missing:", len(df_model))
print(df_model.isnull().sum().sort_values(ascending=False))
df_model = df_model.dropna()
print("# de observaciones tras replace missing:", len(df_model))

print(df_model.columns.tolist())