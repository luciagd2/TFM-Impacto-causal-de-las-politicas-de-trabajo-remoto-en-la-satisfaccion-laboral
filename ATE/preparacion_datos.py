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
###    "n_qfhigh",    # nivel de educacion mas alto completado
    
    # Estructura laboral
    "n_jbhrs",      # horas trabajadas a la semana
###    "n_jbsectpub",
    "n_jbsize",    # tamaño de la empresa
    "n_jbmngr",    # puesto de trabajo
    "n_jbseg_dv",  # grupo socioeconomico al que pertenece (ocupacion)
    "n_basrate",   # salario por hora
###    "n_paytyp",    # tipo de contrato

    # Flexibilidad horaria
###    "n_jbflex7",   # horarios flexibles: trabajar desde casa regularmente
###    "n_jbfxuse7",  # uso del trabajo flexible

    # Variables de tecnologia
###    "n_wktechinet1",    # uso de ordenadores de mesa en el trabajo
###    "n_wktechinet2",    # uso deportatiles en el trabajo
###    "n_wktechinet3",    # uso de telefonos en el trabajo
###    "n_wktechinet4",    # uso de tablets en el trabajo
###    "n_wktechinet5",    # uso de 'feature phone/non-touchscreen mobilephone' en el trabajo
###    "n_wktechinet6",    # uso de 'handheld device/PDA' en el trabajo

    # Variables de autonomia
    "n_wkaut1",    # autonomia sobre las tareas a realizar
    "n_wkaut2",    # autonomia sobre el ritmo de trabajo
    "n_wkaut3",    # autonomia sobre como hacer el trabajo
    "n_wkaut4",    # autonomia sobre el orden para realizar las tareas
    "n_wkaut5",    # autonomia sobre las horas de trabajo
###    "n_wkaut6",    # autonomia sobre el lugar de trabajo -> redundante con teletrabajo
]]


## RENOMBRADO DE VARIABLES
df_model = df_model.rename(columns={
    "n_wkhome": "teletrabajo",
    "n_jbsat":  "satisfaccion",
    "n_pdvage": "edad",
    "n_sex":    "sexo",
###    "n_qfhigh": "educacion",
    "n_jbhrs":  "horas_trabajadas",
    "n_jbsect": "sector",
###    "n_jbsectpub": "sector_pub",
    "n_jbsize": "tam_empresa",
    "n_jbmngr": "puesto", 
    
    "n_wkaut1": "aut_tareas", 
    "n_wkaut2": "aut_ritmo", 
    "n_wkaut3": "aut_metodo",
    "n_wkaut4": "aut_orden",
    "n_wkaut5": "aut_horas", 
###    "n_wkaut6": "aut_lugar",
###    "n_jbflex7": "horario_flexible",
###    "n_jbfxuse7": "uso_hflex",

    "n_jbseg_dv": "ocupacion",
    "n_basrate":  "salario"
###    "n_paytyp":   "contrato"
})



## LIMPIEZA DE VALORES MISSING

# -9 = missing, -8 = inapplicable, -7 = proxy, -2 = refusal, -1 = don't know
missing_values = [-9, -8, -7, -2, -1]

# Conversion a float
df_model = df_model.astype(float)

# Reemplazar valores correctamente
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



# EDUCACIÓN
# Simplificar en niveles (1 bajo, 2 medio ,3 alto)
###def recode_education(x):
###    if x in [1, 2, 20, 21, 24]:
###        return 3  # universitario
###    elif x in [3, 4, 5, 19, 22, 23]:
###        return 2  # técnico / profesional
###    elif x in [6,7,8,9,10,11,12,13,14,15,16,17,18]:
###        return 1  # secundaria o menos
###    else:
###        return np.nan

###df_model["educacion"] = df_model["educacion"].apply(recode_education)


# SECTOR
# 1 sector privado, 0 sector publico
df_model["sector"] = df_model["sector"].apply(
    lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
)

# PUESTO DE MANAGER
df_model["puesto"] = df_model["puesto"].apply(
    lambda x: 1 if x == 1 else (0 if x in [2,3] else np.nan)
)


# TECNOLOGÍA
tech_cols = [
    "n_wktechinet1",
    "n_wktechinet2",
    "n_wktechinet3",
    "n_wktechinet4",
    "n_wktechinet5",
    "n_wktechinet6"
]
# Normalizacion, si hace uso de al menos un dispositivo 1, si no 0
###df_model["tech"] = (    
###    df_model[tech_cols]    
###    .fillna(0)    
###    .sum(axis=1) > 0
###).astype(int)
# Eliminacion de las columnas usadas para crear esta ultima
###df_model = df_model.drop(columns=tech_cols)


# AUTONOMÍA
aut_cols = [
    "aut_tareas",
    "aut_ritmo",
    "aut_metodo",
    "aut_orden",
    "aut_horas"
]
df_model["autonomia"] = df_model[aut_cols].mean(axis=1, skipna=True)
# Eliminacion de las columnas usadas para crear esta ultima
df_model = df_model.drop(columns=aut_cols)


# TAMAÑO EMPRESA
# Asegurar el tipo de datos
df_model["tam_empresa"] = df_model["tam_empresa"].astype(float)

# SALARIO
# Eliminacion de outliers extremos
df_model = df_model[df_model["salario"] < df_model["salario"].quantile(0.99)]

# TIPO DE CONTRATO
# 1 si es fijo, 0 si no
###df_model["contrato"] = df_model["contrato"].apply(
###    lambda x: 1 if x in [1,2] else (0 if x == 3 else np.nan)
###)

# OCUPACION
### df_model = pd.get_dummies(df_model, columns=["ocupacion"], drop_first=True)

## ELIMINACIÓN DE NULOS
### df_model = df_model.dropna(subset=["teletrabajo", "satisfaccion"])
print("Antes de replace missing:", len(df_model))
print(df_model.isnull().sum().sort_values(ascending=False))
df_model = df_model.dropna()
print("Tras replace missing:", len(df_model))

print(df_model.columns.tolist())