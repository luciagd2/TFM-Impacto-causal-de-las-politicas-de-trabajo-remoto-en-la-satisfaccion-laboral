"""
1.2. Validacion mediante Kruskal-Wallis
"""

from Clustering.clusters import df_test
from CATE.CATE import M_cols

from scipy.stats import kruskal

# KRUSKAL
groups = [
    df_test[df_test["cluster"] == c]["CATE"]
    for c in sorted(df_test["cluster"].unique())
]

stat, p = kruskal(*groups)

print("----- KRUSKAL -----")
print("Kruskal-Wallis stat:", stat)
print("p-value:", p)

# KRUSKAL SOBRE MECANISMOS
print("----- KRUSKAL SOBRE MECANISMOS -----")
for m in M_cols:
    groups = [
        df_test[df_test["cluster"] == c][m]
        for c in sorted(df_test["cluster"].unique())
    ]
    stat, p = kruskal(*groups)
    print(m, p)