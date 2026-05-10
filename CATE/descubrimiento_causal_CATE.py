###
# EL GRAFO ANTERIOR ERA DEMASIADO DENSO
# SE REPITE ELIMINANDO LAS VARIABLES QUE NO SE CONECTABAN NI CON TELETRABAJO NI CON SATISFACCION
# EN NINGUN CASO DE ALPHA
###

from CATE.preparacion_datos_CATE import df_model

from causallearn.search.ConstraintBased.PC import pc

from sklearn.preprocessing import StandardScaler

from causallearn.utils.GraphUtils import GraphUtils
from graphviz import Digraph

# Variables para descubrimiento causal
vars_ant = [
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

# Nos aseguramos de que no existen nulos
df_causal = df_model[vars].dropna()

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
# Variables objetivo
focus_vars = ["teletrabajo", "satisfaccion"]

# Crear grafo
dot = Digraph()
# Recorrer aristas
for edge in cg_01.G.get_graph_edges():
    node1 = edge.get_node1().get_name()
    node2 = edge.get_node2().get_name()
    # Mantener solo conexiones relevantes
    if node1 in focus_vars or node2 in focus_vars:
        # Tipo de arista
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()
        # Direccion simple
        if str(endpoint1) == "TAIL" and str(endpoint2) == "ARROW":
            dot.edge(node1, node2)
        elif str(endpoint1) == "ARROW" and str(endpoint2) == "TAIL":
            dot.edge(node2, node1)
        else:
            # Arista no orientada
            dot.edge(node1, node2, dir="none")
# Guardar
dot.render("dag_parcial_001", format="png", cleanup=True)

# Crear grafo
dot = Digraph()
# Recorrer aristas
for edge in cg_05.G.get_graph_edges():
    node1 = edge.get_node1().get_name()
    node2 = edge.get_node2().get_name()
    # Mantener solo conexiones relevantes
    if node1 in focus_vars or node2 in focus_vars:
        # Tipo de arista
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()
        # Direccion simple
        if str(endpoint1) == "TAIL" and str(endpoint2) == "ARROW":
            dot.edge(node1, node2)
        elif str(endpoint1) == "ARROW" and str(endpoint2) == "TAIL":
            dot.edge(node2, node1)
        else:
            # Arista no orientada
            dot.edge(node1, node2, dir="none")
# Guardar
dot.render("dag_parcial_005", format="png", cleanup=True)

# Crear grafo
dot = Digraph()
# Recorrer aristas
for edge in cg_1.G.get_graph_edges():
    node1 = edge.get_node1().get_name()
    node2 = edge.get_node2().get_name()
    # Mantener solo conexiones relevantes
    if node1 in focus_vars or node2 in focus_vars:
        # Tipo de arista
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()
        # Direccion simple
        if str(endpoint1) == "TAIL" and str(endpoint2) == "ARROW":
            dot.edge(node1, node2)
        elif str(endpoint1) == "ARROW" and str(endpoint2) == "TAIL":
            dot.edge(node2, node1)
        else:
            # Arista no orientada
            dot.edge(node1, node2, dir="none")
# Guardar
dot.render("dag_parcial_01", format="png", cleanup=True)