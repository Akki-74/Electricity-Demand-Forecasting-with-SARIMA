import nbformat
from nbclient import NotebookClient

print("Reading Electricity_Demand_Forecasting.ipynb...")
with open("Electricity_Demand_Forecasting.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")

print("Executing all cells (this will populate all visual outputs, tables, and metrics)...")
client.execute()

with open("Electricity_Demand_Forecasting.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook successfully executed and saved with all cell outputs!")
