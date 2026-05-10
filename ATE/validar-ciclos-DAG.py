import networkx as nx

from ATE.preparacion_datos import df_model

# Construcción del grafo
G = nx.DiGraph()

edges = [
    ("teletrabajo", "salario"),
    ("teletrabajo", "autonomia"),
    ("edad", "salario"),
    ("edad", "satisfaccion"),
    ("horas_trabajadas", "salario"),
    ("horas_trabajadas", "satisfaccion"),
    ("tam_empresa", "salario"),
    ("tam_empresa", "autonomia"),
    ("tam_empresa", "puesto"),
    ("salario", "autonomia"),
    ("autonomia", "horas_trabajadas"),
    ("autonomia", "satisfaccion"),
    ("sexo", "satisfaccion"),
    ("sexo", "teletrabajo"),
    ("puesto", "autonomia"),
    ("puesto", "horas_trabajadas"),
    ("ocupacion", "teletrabajo"),
    ("ocupacion", "salario"),
    ("sector", "teletrabajo"),
    ("sector", "salario"),
    ("sector", "ocupacion"),
]

G.add_edges_from(edges)

# Detectar ciclos
cycles = list(nx.simple_cycles(G))

print("CICLOS DETECTADOS:")
for c in cycles:
    print(c)