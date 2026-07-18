import pandas as pd
import glob
import os

def limpar_dados():
    arquivos = sorted(glob.glob("data/raw/dengue_jaboatao_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in arquivos], ignore_index=True)
    print(f"Total de arquivos anuais combinados: {len(arquivos)}")

    df_confirmados = df[df['CLASSI_FIN'].isin(['10', '11', '12'])].copy()
    print(f"Casos confirmados em Jaboatão: {len(df_confirmados)}")

    df_confirmados['DT_NOTIFIC'] = pd.to_datetime(df_confirmados['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
    df_confirmados = df_confirmados[df_confirmados['DT_NOTIFIC'].notna()]
    df_confirmados['ano'] = df_confirmados['DT_NOTIFIC'].dt.year

    return df_confirmados

if __name__ == "__main__":
    os.makedirs("data/silver", exist_ok=True)
    df_limpo = limpar_dados()
    df_limpo.to_parquet("data/silver/casos_dengue_limpos.parquet", index=False)
    print(f"Silver salvo: {len(df_limpo)} casos")