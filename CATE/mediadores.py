from CATE.preparacion_datos_CATE import df_model

mediadores = [
    "edad",
    "sexo",
    "tiene_hijos",
    "num_hijos",
    "sector",
    "ocupacion",
    "puesto",
    "tam_empresa",
    "contrato_perm",
    "horas_trabajadas",
    "horas_extra",
    "antiguedad",
    "educacion",
    "satisfaccion_ingresos",
    "digitalizacion",
    "autonomia",
    "trabajo_fisico",   # 1: muy, 2: bastante, 3: poco, 4: nada fisico
    "paga_rendimiento",   # Pay includes performance related pay 
    "horario_habitual",  # horario en el que se trabaja normalmente
    "horario_fin_semana",  # normalmente trabaja findes de semana
    "seguridad_laboral", 
]

import statsmodels.api as sm

for m in mediadores:
    print("MEDIADOR:", m)

    # modelo mediador ~ tratamiento
    X = sm.add_constant(df_model["teletrabajo"])
    model_m = sm.OLS(df_model[m], X).fit()

    print("Efecto teletrabajo ->", m)
    print(model_m.params["teletrabajo"])

    # modelo outcome ~ tratamiento + mediador
    X2 = sm.add_constant(df_model[["teletrabajo", m]])
    model_y = sm.OLS(df_model["satisfaccion"], X2).fit()

    print("Efecto directo teletrabajo:", model_y.params["teletrabajo"])
    print("Efecto mediador:", model_y.params[m])