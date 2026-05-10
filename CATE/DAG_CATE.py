from CATE.preparacion_datos_CATE import df_model

import statsmodels.api as sm

# Definicion del DAG
dag = """
digraph {
    sexo -> teletrabajo
    educacion -> teletrabajo
    ocupacion -> teletrabajo
    tam_empresa -> teletrabajo
    trabajo_fisico -> teletrabajo
    horario_habitual -> teletrabajo
    horario_fin_semana -> teletrabajo

    sexo -> satisfaccion
    educacion -> satisfaccion
    ocupacion -> satisfaccion
    tam_empresa -> satisfaccion
    trabajo_fisico -> satisfaccion
    horario_habitual -> satisfaccion
    horario_fin_semana -> satisfaccion

    teletrabajo -> autonomia
    teletrabajo -> digitalizacion
    teletrabajo -> horas_trabajadas
    teletrabajo -> horas_extra
    teletrabajo -> satisfaccion_ingresos
    teletrabajo -> seguridad_laboral
    teletrabajo -> paga_rendimiento

    autonomia -> satisfaccion
    digitalizacion -> satisfaccion
    horas_trabajadas -> satisfaccion
    horas_extra -> satisfaccion
    satisfaccion_ingresos -> satisfaccion
    seguridad_laboral -> satisfaccion
    paga_rendimiento -> satisfaccion

    teletrabajo -> satisfaccion
}
"""

df = df_model.copy()

# VARIABLES
T = "teletrabajo"
Y = "satisfaccion"

W = [
    "sexo", "educacion", "ocupacion",
    "tam_empresa", "trabajo_fisico",
    "horario_habitual", "horario_fin_semana"
]

X = [
    "edad", "antiguedad", "tiene_hijos"
]

M = [
    "autonomia", "digitalizacion",
    "horas_trabajadas", "horas_extra",
    "satisfaccion_ingresos", "seguridad_laboral",
    "paga_rendimiento"
]


# TEST DE CONFUSION
# ¿W realmente explica T?
X = df[W]
y = df[T]

X = sm.add_constant(X)
model = sm.Logit(y, X).fit()

print("----- TEST DE CONFUSION W -> T -----")
print(model.summary())

# ¿W explica Y?
X = df[W]
y = df[Y]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print("----- TEST DE CONFUSION W -> Y -----")
print(model.summary())

# TEST DE INDEPENDENCIA CONDICIONAL
X = df[W + [T]]
y = df[Y]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("----- TEST DE INDEPENDENCIA CONDICINAL -----")
print(model.summary())

# TEST DE MEDIACION
X1 = df[W + [T]]
X2 = df[W + M]
y = df[Y]

model1 = sm.OLS(y, sm.add_constant(X1)).fit()
model2 = sm.OLS(y, sm.add_constant(X2)).fit()

print("----- TEST DE MEDIACION -----")
print("Modelo con T:", model1.params[T])
print("Modelo con mediadores:", model2.summary())

