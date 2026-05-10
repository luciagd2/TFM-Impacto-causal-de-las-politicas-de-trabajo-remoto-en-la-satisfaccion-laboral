from CATE.CATE import X_train
from CATE.CATE import cf_model

import pandas as pd
import matplotlib.pyplot as plt

# FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    "variable": X_train.columns,
    "importance": cf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)
print("----- FEATURE IMPORTANCE -----")
print(feature_importance)

feature_importance.plot(
    x="variable",
    y="importance",
    kind="bar"
)

plt.title("Importancia causal de las variables")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.show()
plt.close()