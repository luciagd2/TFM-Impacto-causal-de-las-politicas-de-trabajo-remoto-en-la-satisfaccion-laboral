"""
3. Validación de aciclicidad del grafo causal.

En este script se verifica que la estructura causal propuesta cumple la propiedad de aciclicidad 
requerida por los grafos acíclicos dirigidos (DAGs).

Para ello, se comprueba que no existan caminos cerrados que permitan regresar a una variable 
siguiendo la dirección de las aristas.
"""

import networkx as nx
from ATE.preparacion_datos import df_model

# Construcción del grafo
G = nx.DiGraph()

relaciones = [
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

G.add_edges_from(relaciones)

# Detectar ciclos
ciclos = list(nx.simple_cycles(G))

print("CICLOS DETECTADOS:")
for c in ciclos:
    print(c)