from pysus.online_data.SINAN import download
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

os.makedirs("data/raw", exist_ok=True)

df = pd.read_parquet("data/raw/dengue_jaboatao_2018_2025.parquet")
  
df_confirmados = df[df['CLASSI_FIN'].isin(['10', '11', '12'])].copy() # para colocar no tcc dps que separei apenas os casos confirmados de notificacao usando: # CLASSI_FIN: 10 = Dengue, 11 = Dengue com sinais de alarme, 12 = Dengue grave
print(f"Casos confirmados em Jaboatão: {len(df_confirmados)}")


df_confirmados['DT_NOTIFIC'] = pd.to_datetime(df_confirmados['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
df_confirmados = df_confirmados[df_confirmados['DT_NOTIFIC'].notna()]
df_confirmados['ano'] = df_confirmados['DT_NOTIFIC'].dt.year

df_confirmados = df_confirmados[df_confirmados['ano'] <= 2025]
print(df_confirmados['ano'].value_counts().sort_index())


df_confirmados = df_confirmados[df_confirmados['DT_NOTIFIC'].notna()]

# incidência por 100k habitantes
# usando interpolação linear entre 2010 e 2022 para estimar população de Jaboatão dos Guararapes
dados_ibge = {
    2010: 644_620,  
    2022: 644_037  
}

def interpolar_populacao(ano_inicio=2018, ano_fim=2025):
    
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
        
        populacoes.append({
            'ano': ano,
            'populacao': int(pop)
        })
    
    return pd.DataFrame(populacoes)

df_pop = interpolar_populacao()
print(df_pop)

df_confirmados = df_confirmados.sort_values('DT_NOTIFIC')

df_semanal = df_confirmados.groupby(
    pd.Grouper(key='DT_NOTIFIC', freq='W-SUN')
).size().reset_index(name='casos')

df_semanal = df_semanal.rename(columns={'DT_NOTIFIC': 'data_semana'})

df_semanal = df_semanal.set_index('data_semana').asfreq('W-SUN', fill_value=0).reset_index()

df_semanal['ano'] = df_semanal['data_semana'].dt.year

df_semanal = df_semanal.merge(df_pop, on='ano', how='left')

# taxa de incidência por 100.000 habitantes
df_semanal['incidencia'] = (df_semanal['casos'] / (df_semanal['populacao']/52)) * 100000

print(df_semanal.head(20))

# normalização Min-Max da incidência
scaler = MinMaxScaler()
df_semanal['incidencia_norm'] = scaler.fit_transform(df_semanal[['incidencia']])

os.makedirs("data/processed", exist_ok=True)
df_semanal.to_parquet("data/processed/dengue_jaboatao_semanal_2018_2025_processed.parquet", index=False)


print("debugando")
print(f"dados brutos: {len(df)}")
print(f"dados após filtro confirmados + datas válidas: {len(df_confirmados)}")

df_confirmados['DT_NOTIFIC'] = pd.to_datetime(df_confirmados['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
df_confirmados_valid = df_confirmados[df_confirmados['DT_NOTIFIC'].notna()]
print(f"dados sem datas inválidas: {len(df_confirmados_valid)}")

df_confirmados_valid = df_confirmados_valid.sort_values('DT_NOTIFIC')

df_semanal_debug = df_confirmados_valid.groupby(
    pd.Grouper(key='DT_NOTIFIC', freq='W-SUN')
).size().reset_index(name='casos')

print(f"semanas na série: {len(df_semanal_debug)}")
print(f"semanas com casos < 0: {(df_semanal_debug['casos'] < 0).sum()}")