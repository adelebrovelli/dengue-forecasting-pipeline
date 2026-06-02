from pysus.online_data.SINAN import download
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

anos = range(2018, 2025)  # 2018 até 2025
uf = "PE"
dfs = []

for ano in anos:
    print(f"Baixando {uf} - {ano}...")
    df = download("DENG", ano, uf).to_dataframe()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df_jaboatao = df[df['ID_MN_RESI'].astype(str) == "260790"]

print(f"Casos em Jaboatão: {len(df_jaboatao)}")

df_jaboatao.to_parquet("data/raw/dengue_jaboatao_2018_2025.parquet")
print("Salvo")