import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler

os.makedirs("data/processed", exist_ok=True)

# Junta todos os parquets anuais em um único DataFrame
arquivos = sorted(glob.glob("data/raw/dengue_jaboatao_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in arquivos], ignore_index=True)
print(f"Total de arquivos anuais combinados: {len(arquivos)}")

df_confirmados = df[df['CLASSI_FIN'].isin(['10', '11', '12'])].copy()
print(f"Casos confirmados em Jaboatão: {len(df_confirmados)}")

df_confirmados['DT_NOTIFIC'] = pd.to_datetime(df_confirmados['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
df_confirmados = df_confirmados[df_confirmados['DT_NOTIFIC'].notna()]
df_confirmados['ano'] = df_confirmados['DT_NOTIFIC'].dt.year

dados_ibge = {2010: 644_620, 2022: 644_037}

def interpolar_populacao(ano_inicio, ano_fim):
    anos = list(range(ano_inicio, ano_fim + 1))
    populacoes = []
    for ano in anos:
        if ano <= 2010:
            pop = dados_ibge[2010]
        elif ano >= 2022:
            pop = dados_ibge[2022]
        else:
            t = (ano - 2010) / (2022 - 2010)
            pop = dados_ibge[2010] + t * (dados_ibge[2022] - dados_ibge[2010])
        populacoes.append({'ano': ano, 'populacao': int(pop)})
    return pd.DataFrame(populacoes)

ano_min, ano_max = df_confirmados['ano'].min(), df_confirmados['ano'].max()
df_pop = interpolar_populacao(ano_min, ano_max)

df_confirmados = df_confirmados.sort_values('DT_NOTIFIC')
df_semanal = df_confirmados.groupby(pd.Grouper(key='DT_NOTIFIC', freq='W-SUN')).size().reset_index(name='casos')
df_semanal = df_semanal.rename(columns={'DT_NOTIFIC': 'data_semana'})
df_semanal = df_semanal.set_index('data_semana').asfreq('W-SUN', fill_value=0).reset_index()
df_semanal['ano'] = df_semanal['data_semana'].dt.year
df_semanal = df_semanal.merge(df_pop, on='ano', how='left')
df_semanal['incidencia'] = (df_semanal['casos'] / (df_semanal['populacao']/52)) * 100000

# Normalização: fit SÓ no período de treino (2018-2024), aplicado depois em toda a série.
# Evita vazamento de dado (data leakage) do período de teste (2025) no ajuste do scaler.
treino_mask = df_semanal['ano'] <= 2024
scaler = MinMaxScaler()
scaler.fit(df_semanal.loc[treino_mask, ['incidencia']])
df_semanal['incidencia_norm'] = scaler.transform(df_semanal[['incidencia']])

df_semanal.to_parquet("data/processed/dengue_jaboatao_semanal_processed.parquet", index=False)
print(f"Processado: {len(df_semanal)} semanas salvas")
