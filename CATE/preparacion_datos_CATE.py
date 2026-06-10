"""
1. Preparación de datos para el análisis del CATE.

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

    "n_wkaut1",     # autonomia sobre las tareas a realizar
    "n_wkaut2",     # autonomia sobre el ritmo de trabajo
    "n_wkaut3",     # autonomia sobre como hacer el trabajo
    "n_wkaut4",     # autonomia sobre el orden para realizar las tareas
    "n_wkaut5",     # autonomia sobre las horas de trabajo
    "n_jbseg_dv",   # grupo socioeconomico al que pertenece (ocupacion)
    "n_pdvage",     # edad
    "n_sex",        # sexo
    "n_nchunder16", # num.hijos menores de 16
    "n_jbsize",     # tamaño de la empresa
    "n_isced11_dv", # educacion
    "n_jbbgy",
    "n_sclfsat2",   # satisfaccion con los ingresos percibidos

    "n_wktechinet1", # digitalizacion
    "n_wktechinet2",
    "n_wktechinet3",
    "n_wktechinet4",
    "n_wktechinet5",
    "n_wktechinet6",

    "n_cjbatt",      # motivacion para seguir en el trabajo

    "n_jbhrs",       # horas trabajadas a la semana
    "n_jbot",        # horas_extra
    "n_wkphys",      # physicality of job
    "n_jbperfp",     # Pay includes performance related pay 
    "n_wktime",      # horario en el que se trabaja normalmente
    "n_wkends",      # normalmente trabaja findes de semana
    "n_jbsec",       # job security

# Variables descartadas -> Por redundancia con otras variables o por perder demasiadas observaciones
###    "n_jbsect",   # tipo de organizacion/empresa para la que trabaja
###    "n_jbmngr",   # puesto de trabajo
###    "n_jbterm1",  # tipo de contrato
###    "n_mlstat",   # estado civil --- 25535 missing

###    "n_depenth1", # estres laboral
###    "n_depenth2",
###    "n_depenth3",
###    "n_depenth4",
###    "n_depenth5",
###    "n_depenth6",

###    "n_basrate",  # salario por hora --- 29848 missing
###    "n_jbotpd"    # horas_extra_pagadas --- 27819 missing
###    "n_jbttwt",   # Minutos perdidos en ir al trabajo
###    "n_wktrvfar", # Tipo de transporte usado para llegar al trabajo
###    "n_workdis",  # work distancce --- 8424
###    "n_journeysat", # satisfaccion con commute --- 8711
]]

## RENOMBRADO DE VARIABLES
df_model = df_model.rename(columns={
    "n_wkhome": "teletrabajo",
    "n_jbsat":  "satisfaccion",
    
    "n_wkaut1": "aut_tareas", 
    "n_wkaut2": "aut_ritmo", 
    "n_wkaut3": "aut_metodo",
    "n_wkaut4": "aut_orden",
    "n_wkaut5": "aut_horas", 
    
    "n_jbseg_dv": "ocupacion",
    "n_nchunder16": "num_hijos",

    "n_pdvage": "edad",
    "n_sex":    "sexo",
    
    "n_jbsize": "tam_empresa",

    "n_isced11_dv": "educacion",

    "n_sclfsat2": "satisfaccion_ingresos",

    "n_cjbatt": "motivacion_trabajo",

    "n_jbhrs":  "horas_trabajadas",
    "n_jbot": "horas_extra",

    "n_jbttwt": "tiempo_viaje_trabajo",

    "n_wkphys": "trabajo_fisico",   # 1: muy, 2: bastante, 3: poco, 4: nada fisico

    "n_jbperfp": "paga_rendimiento",   # Pay includes performance related pay 

    "n_wktime": "horario_habitual",  # horario en el que se trabaja normalmente

    "n_wkends": "horario_fin_semana",  # normalmente trabaja findes de semana

    "n_jbsec": "seguridad_laboral",   # job security

})
    
## ---------- LIMPIEZA DE VALORES MISSING ----------
# Conversion a float
df_model = df_model.astype(float)

# Reemplazar valores correctamente
missing_values = [-9, -8, -7, -2, -1]
df_model = df_model.replace(missing_values, np.nan)


## ---------- TRATAMIENTO DE VARIABLES ----------
# TELETRABAJO (TRATAMIENTO)

# Eliminacion de los datos con valor wkhome = 8 (Variable) ya que es demasiado ambigüa
df_model = df_model[df_model["teletrabajo"] != 8]

# Convertir a binaria: 1 = teletrabaja, 0 = no o muy poco
df_model["teletrabajo"] = df_model["teletrabajo"].apply(
    lambda x: 1 if x >= 3 else 0
)

# SATISFACCIÓN (RESULTADO)
df_model["satisfaccion"] = df_model["satisfaccion"].astype(float)

# AUTONOMÍA
aut_cols = [
    "aut_tareas",
    "aut_ritmo",
    "aut_metodo",
    "aut_orden",
    "aut_horas",
]
df_model["autonomia"] = df_model[aut_cols].mean(axis=1, skipna=True)
# Se invierte la escala para que los valores vayan desde 1-nada hasta 4-mucha autonomia
df_model["autonomia"] = 5 - df_model["autonomia"]
# Eliminacion de las columnas usadas para crear esta ultima
df_model = df_model.drop(columns=aut_cols)

# HIJOS (¿es responsable de hijos menores de 15? Si = 1, no = 0)
df_model["tiene_hijos"] = df_model["num_hijos"].apply(
    lambda x: 1 if x > 0 else 0
)

# NUMERO DE HIJOS
def recode_num_hijos(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x == 2:
        return 2
    elif x == 3:
        return 3
    else:
        return 4

df_model["num_hijos"] = df_model["num_hijos"].apply(recode_num_hijos)

# AÑOS EN EL PUESTO
df_model["antiguedad"] = 2026 - df_model["n_jbbgy"]
df_model = df_model.drop(columns="n_jbbgy")

# EDUCACIÓN
# Simplificar en niveles (1 bajo, 2 medio ,3 alto)
def recode_education(x):
    if x in [1, 2, 20, 21, 24]:
        return 3  # universitario
    elif x in [3, 4, 5, 19, 22, 23]:
        return 2  # técnico / profesional
    elif x in [6,7,8,9,10,11,12,13,14,15,16,17,18]:
        return 1  # secundaria o menos
    else:
        return np.nan

df_model["educacion"] = df_model["educacion"].apply(recode_education)

# HORARIO HABITUAL
def recode_horario(x):

    if x in [1,2,3,10,6]:
        return 1  # horario diurno

    elif x in [4,5]:
        return 2  # nocturno

    elif x in [8,9]:
        return 3  # variable / turnos

    else:
        return np.nan  # otros
    
df_model["horario_habitual"] = df_model["horario_habitual"].apply(recode_horario)

# TRABAJA EN FIN DE SEMANA
def recode_weekend(x):
    if x == 3:
        return 0  # nunca

    elif x == 2:
        return 1  # algunos

    elif x == 1:
        return 2  # casi todos

    else:
        return np.nan

df_model["horario_fin_semana"] = df_model["horario_fin_semana"].apply(recode_horario)

# SEGURIDAD LABORAL
def recode_seguridad(x):
# mayor valor, mayor seguridad laboral
    if x == 1:
        return 4

    elif x == 2:
        return 3

    elif x == 3:
        return 2

    elif x == 4:
        return 1

    else:
        return np.nan

df_model["seguridad_laboral"] = df_model["seguridad_laboral"].apply(recode_horario)

# PAGA RELACIONADA CON EL RENDIMIENTO 
df_model["paga_rendimiento"] = df_model["paga_rendimiento"].apply(
    lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
)

# DIGITALIZACION
tech_cols = [
    "n_wktechinet1",
    "n_wktechinet2",
    "n_wktechinet3",
    "n_wktechinet4",
    "n_wktechinet5",
    "n_wktechinet6"
]
# Normalizacion, si hace uso de al menos un dispositivo 1, si no 0
df_model["digitalizacion"] = (    
    df_model[tech_cols]    
    .fillna(0)    
    .sum(axis=1) > 0
).astype(int)
# Eliminacion de las columnas usadas para crear esta ultima
df_model = df_model.drop(columns=tech_cols)


# MOTIVACION LABORAL
def recode_motivacion(x):
    if x in [11]:
        return 1 # flexibilidad
    elif x in [1]:
        return 2 # dinero
    elif x in [5, 6, 8]:
        return 3 # intrinseco
    elif x in [2, 3, 4]:
        return 4 #progresion_seguridad
    else:
        return 5 # otro
df_model["motivacion_trabajo"] = df_model["motivacion_trabajo"].apply(recode_motivacion)


## ---------- ELIMINACIÓN DE NULOS ----------
print("Antes de replace missing:", len(df_model))
print(df_model.isnull().sum().sort_values(ascending=False))
df_model = df_model.dropna()
print("Tras replace missing:", len(df_model))

print(df_model.columns.tolist())