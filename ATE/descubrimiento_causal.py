from ATE.preparacion_datos import df_model

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils

from sklearn.preprocessing import StandardScaler

# Variables para descubrimiento causal
vars_causal = [
    "teletrabajo",
    "satisfaccion",
    "edad",
    "horas_trabajadas",
    "tam_empresa",
    "salario",
    "autonomia"
]

# Variables demograficas
vars_demografico = [
    "teletrabajo",
    "satisfaccion",
    "edad",
    "sexo"
]

# Variables laborables
vars_laborales = [
    "teletrabajo",
    "satisfaccion",
    "horas_trabajadas",
    "tam_empresa",
    "puesto",
    "salario",
    "autonomia"
]

# Variables ocupacionales
vars_ocupacionales = [
    "teletrabajo",
    "satisfaccion",
    "ocupacion",
    "sector",
    "salario"
]

# Nos aseguramos de que no existen nulos
df_causal = df_model[vars_causal].dropna()
df_demografico = df_model[vars_demografico].dropna()
df_laboral = df_model[vars_laborales].dropna()
df_ocupacional = df_model[vars_ocupacionales].dropna()

# Escalado
scaler = StandardScaler()
data_scaled_causal = scaler.fit_transform(df_causal)
data_scaled_demografico = scaler.fit_transform(df_demografico)
data_scaled_laboral = scaler.fit_transform(df_laboral)
data_scaled_ocupacional = scaler.fit_transform(df_ocupacional)

# PC Algorithm
cg_c01 = pc(data_scaled_causal, alpha=0.01)
cg_c05 = pc(data_scaled_causal, alpha=0.05)
cg_c1 = pc(data_scaled_causal, alpha=0.1)

cg_d01 = pc(data_scaled_demografico, alpha=0.01)
cg_d05 = pc(data_scaled_demografico, alpha=0.05)
cg_d1 = pc(data_scaled_demografico, alpha=0.1)

cg_l01 = pc(data_scaled_laboral, alpha=0.01)
cg_l05 = pc(data_scaled_laboral, alpha=0.05)
cg_l1 = pc(data_scaled_laboral, alpha=0.1)

cg_o01 = pc(data_scaled_ocupacional, alpha=0.01)
cg_o05 = pc(data_scaled_ocupacional, alpha=0.05)
cg_o1 = pc(data_scaled_ocupacional, alpha=0.1)

# Visualizar resultado
print("Variables causales, alpha = 0.01", cg_c01.G)
print("Variables causales, alpha = 0.05", cg_c05.G)
print("Variables causales, alpha = 0.1", cg_c1.G)

print("Variables demograficas, alpha = 0.01", cg_d01.G)
print("Variables demograficas, alpha = 0.05", cg_d05.G)
print("Variables demograficas, alpha = 0.1", cg_d1.G)

print("Variables laborales, alpha = 0.01", cg_l01.G)
print("Variables laborales, alpha = 0.05", cg_l05.G)
print("Variables laborales, alpha = 0.1", cg_l1.G)

print("Variables ocupacionales, alpha = 0.01", cg_o01.G)
print("Variables ocupacionales, alpha = 0.05", cg_o05.G)
print("Variables ocupacionales, alpha = 0.1", cg_o1.G)


# Generar grafos y guardar las imagenes
pyd = GraphUtils.to_pydot(cg_c01.G)
pyd.write_png("dag_causal_c001.png")
pyd = GraphUtils.to_pydot(cg_c05.G)
pyd.write_png("dag_causal_c005.png")
pyd = GraphUtils.to_pydot(cg_c1.G)
pyd.write_png("dag_causal_c01.png")

pyd = GraphUtils.to_pydot(cg_d01.G)
pyd.write_png("dag_causal_d001.png")
pyd = GraphUtils.to_pydot(cg_d05.G)
pyd.write_png("dag_causal_d005.png")
pyd = GraphUtils.to_pydot(cg_d1.G)
pyd.write_png("dag_causal_d01.png")

pyd = GraphUtils.to_pydot(cg_l01.G)
pyd.write_png("dag_causal_l001.png")
pyd = GraphUtils.to_pydot(cg_l05.G)
pyd.write_png("dag_causal_l005.png")
pyd = GraphUtils.to_pydot(cg_l1.G)
pyd.write_png("dag_causal_l1.png")

pyd = GraphUtils.to_pydot(cg_o01.G)
pyd.write_png("dag_causal_o001.png")
pyd = GraphUtils.to_pydot(cg_o05.G)
pyd.write_png("dag_causal_o005.png")
pyd = GraphUtils.to_pydot(cg_o1.G)
pyd.write_png("dag_causal_o01.png")