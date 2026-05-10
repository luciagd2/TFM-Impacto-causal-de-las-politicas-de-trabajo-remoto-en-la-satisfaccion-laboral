from CATE.preparacion_datos_CATE import df_model

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

from sklearn.preprocessing import StandardScaler

# Variables para descubrimiento causal
vars = [
    "teletrabajo",          # 1
    "satisfaccion",         # 2

    "edad",                 # 3
    "sexo",                 # 4
    "educacion",            # 5
    "sector",               # 6
    "ocupacion",            # 7
    "tam_empresa",          # 8
    "puesto",               # 9
    "contrato_perm",        # 10
    "antiguedad",           # 11
    "tiene_hijos",          # 12

    "autonomia",            # 13
    "digitalizacion",       # 14
    "horas_trabajadas",     # 15
    "horas_extra",          # 16
    "seguridad_laboral",    # 17
    "motivacion_trabajo",   # 18
    "satisfaccion_ingresos",# 19
    "trabajo_fisico"        # 20
]

# Nos aseguramos de que no existen nulos
df_causal = df_model[vars].dropna()

# Escalado
scaler = StandardScaler()
data_scaled_causal = scaler.fit_transform(df_causal)

# PC Algorithm
cg_01 = pc(data_scaled_causal, alpha=0.01)
cg_05 = pc(data_scaled_causal, alpha=0.05)
cg_1 = pc(data_scaled_causal, alpha=0.1)

# Visualizar resultado
print("Variables causales, alpha = 0.01", cg_01.G)
print("Variables causales, alpha = 0.05", cg_05.G)
print("Variables causales, alpha = 0.1", cg_1.G)

# Generar grafos y guardar las imagenes
pyd = GraphUtils.to_pydot(cg_01.G)
pyd.write_png("dag_causal_001.png")
pyd = GraphUtils.to_pydot(cg_05.G)
pyd.write_png("dag_causal_005.png")
pyd = GraphUtils.to_pydot(cg_1.G)
pyd.write_png("dag_causal_01.png")