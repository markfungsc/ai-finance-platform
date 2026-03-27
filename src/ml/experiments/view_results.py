import pandas as pd

df = pd.read_csv("experiments/results.csv")

print(df.sort_values("directional_accuracy", ascending=False))
