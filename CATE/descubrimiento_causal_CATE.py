###
# EL GRAFO ANTERIOR ERA DEMASIADO DENSO
# SE REPITE ELIMINANDO LAS VARIABLES QUE NO SE CONECTABAN NI CON TELETRABAJO NI CON SATISFACCION
# EN NINGUN CASO DE ALPHA
###

from preparacion_datos_CATE import df_model

from causallearn.search.ConstraintBased.PC import pc

from sklearn.preprocessing import StandardScaler

from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.Graph import Graph
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

# Variables para descubrimiento causal


vars = [
    "teletrabajo",          # 1
    "satisfaccion",         # 2

    "edad",                 # 3
    "sexo",                 # 4
    "educacion",            # 5
    "sector",               # 6
    "ocupacion",            # 7
    "puesto",               # 8
    "tam_empresa",          # 9
    "antiguedad",           # 10
    "contrato_perm",        # 11
    "tiene_hijos",          # 12

    "autonomia",            # 13
    "digitalizacion",       # 14
    "horas_trabajadas",     # 15
    "horas_extra",          # 16
    "seguridad_laboral",    # 17
    "motivacion_trabajo",   # 18
    "satisfaccion_ingresos",# 19
    "trabajo_fisico",       # 20
    "paga_rendimiento",     # 21
    "horario_habitual",     # 22
    "horario_fin_semana"    # 23
]

vars_2 = [
    "teletrabajo",          # 1
    "satisfaccion",         # 2

    "edad",                 # 3
    "sexo",                 # 4
    "educacion",            # 5
    "ocupacion",            # 6
    "tam_empresa",          # 7
    "antiguedad",           # 8
    "tiene_hijos",          # 9

    "autonomia",            # 10
    "digitalizacion",       # 11
    "horas_trabajadas",     # 12
    "horas_extra",          # 13
    "seguridad_laboral",    # 14
    "motivacion_trabajo",   # 15
    "satisfaccion_ingresos",# 16
    "trabajo_fisico"        # 17
]

# Nos aseguramos de que no existen nulos
df_causal = df_model[vars_2].dropna()

# Se reduce la carga computacional
df_sample = df_causal.sample(
    n=5000,
    random_state=42
)

# Escalado
scaler = StandardScaler()
###data_scaled_causal = scaler.fit_transform(df_causal)
data_scaled_causal = scaler.fit_transform(df_sample)

# Algoritmo PC
cg_01 = pc(data_scaled_causal, alpha=0.01)
cg_05 = pc(data_scaled_causal, alpha=0.05)
cg_1 = pc(data_scaled_causal, alpha=0.1)

# Visualizar resultado
print("Variables causales, alpha = 0.01", cg_01.G)
print("Variables causales, alpha = 0.05", cg_05.G)
print("Variables causales, alpha = 0.1", cg_1.G)

# Generar grafos y guardar las imagenes
pyd = GraphUtils.to_pydot(cg_01.G)
pyd.write_png("dag_causal_001_2.png")
pyd = GraphUtils.to_pydot(cg_05.G)
pyd.write_png("dag_causal_005_2.png")
pyd = GraphUtils.to_pydot(cg_1.G)
pyd.write_png("dag_causal_01_2.png")


# ----- Imagenes mas visuales del DAG simplificado -----
filtered_edges001 = nx.DiGraph()
filtered_edges005 = nx.DiGraph()
filtered_edges01 = nx.DiGraph()

for e in cg_01.G.get_graph_edges():
    n1 = e.get_node1().get_name()
    n2 = e.get_node2().get_name()

    if "X1" in [n1, n2] or "X2" in [n1, n2]:
        filtered_edges001.add_edge(n1, n2)

for e in cg_05.G.get_graph_edges():
    n1 = e.get_node1().get_name()
    n2 = e.get_node2().get_name()

    if "X1" in [n1, n2] or "X2" in [n1, n2]:
        filtered_edges005.add_edge(n1, n2)

for e in cg_1.G.get_graph_edges():
    n1 = e.get_node1().get_name()
    n2 = e.get_node2().get_name()

    if "X1" in [n1, n2] or "X2" in [n1, n2]:
        filtered_edges01.add_edge(n1, n2)

# Generar grafos y guardar las imagenes
pyd = to_pydot(filtered_edges001)
pyd.write_png("dag_causal_simp_001.png")
pyd = to_pydot(filtered_edges005)
pyd.write_png("dag_causal_simp_005.png")
pyd = to_pydot(filtered_edges01)
pyd.write_png("dag_causal_simp_01.png")