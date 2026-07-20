from pysus.online_data.SINAN import download
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

anos = range(2018, 2020)
uf = "PE"
municipio_jaboatao = "260790"

for ano in anos:
    print(f"Baixando {uf} - {ano}...")
    
    df_estado = download("DENG", ano, uf).to_dataframe()
    
    df_jaboatao = df_estado[df_estado['ID_MN_RESI'].astype(str) == municipio_jaboatao].copy()
    del df_estado  # libera o DataFrame do estado inteiro assim que possível
    
    caminho = f"data/raw/dengue_jaboatao_{ano}.parquet"
    df_jaboatao.to_parquet(caminho)
    print(f"Salvo {ano}: {len(df_jaboatao)} casos em {caminho}")
    
    del df_jaboatao

print("Coleta finalizada")
