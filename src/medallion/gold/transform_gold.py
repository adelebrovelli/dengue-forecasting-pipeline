import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

MUNICIPIO_ID = 260790
MUNICIPIO_NOME = "Jaboatão dos Guararapes"

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


def agregar_semanal(df_confirmados):
    df_confirmados = df_confirmados.sort_values('DT_NOTIFIC')
    df_semanal = df_confirmados.groupby(pd.Grouper(key='DT_NOTIFIC', freq='W-SUN')).size().reset_index(name='casos')
    df_semanal = df_semanal.rename(columns={'DT_NOTIFIC': 'data_semana'})
    df_semanal = df_semanal.set_index('data_semana').asfreq('W-SUN', fill_value=0).reset_index()
    df_semanal['ano'] = df_semanal['data_semana'].dt.year
    return df_semanal


if __name__ == "__main__":
    os.makedirs("data/gold", exist_ok=True)

    df_confirmados = pd.read_parquet("data/silver/casos_dengue_limpos.parquet")
    df_semanal = agregar_semanal(df_confirmados)

    ano_min, ano_max = df_semanal['ano'].min(), df_semanal['ano'].max()
    df_pop = interpolar_populacao(ano_min, ano_max)
    df_semanal = df_semanal.merge(df_pop, on='ano', how='left')
    df_semanal['incidencia'] = (df_semanal['casos'] / (df_semanal['populacao'] / 52)) * 100000

    treino_mask = df_semanal['ano'] <= 2024

    # Normalização: fit SÓ no treino (<=2024), evita vazamento do teste (2025+)
    scaler = MinMaxScaler()
    scaler.fit(df_semanal.loc[treino_mask, ['incidencia']])
    df_semanal['incidencia_norm'] = scaler.transform(df_semanal[['incidencia']])

    # Limiar de surto: percentil 75% calculado SÓ no período de treino
    limiar_surto = df_semanal.loc[treino_mask, 'incidencia'].quantile(0.75)
    df_semanal['is_surto'] = df_semanal['incidencia'] > limiar_surto
    print(f"Limiar de surto (p75, treino): {limiar_surto:.2f} casos/100k hab")

    # dim_tempo_epidemiologico
    dim_tempo = df_semanal[['data_semana', 'ano']].copy()
    dim_tempo['semana_epi_id'] = range(1, len(dim_tempo) + 1)
    dim_tempo['semana_epidemiologica'] = dim_tempo['data_semana'].dt.isocalendar().week
    dim_tempo['mes'] = dim_tempo['data_semana'].dt.month
    dim_tempo['trimestre'] = dim_tempo['data_semana'].dt.quarter
    dim_tempo = dim_tempo[['semana_epi_id', 'ano', 'semana_epidemiologica', 'mes', 'trimestre', 'data_semana']]

    # dim_municipio
    dim_municipio = pd.DataFrame([{
        'municipio_id': MUNICIPIO_ID,
        'codigo_ibge': MUNICIPIO_ID,
        'nome_municipio': MUNICIPIO_NOME,
        'uf': 'PE'
    }])

    # fato_casos_dengue (registros observados — previsão entra depois, na task de predict)
    fato = df_semanal.copy()
    fato['semana_epi_id'] = dim_tempo['semana_epi_id'].values
    fato['municipio_id'] = MUNICIPIO_ID
    fato['tipo_registro'] = 'observado'
    fato['modelo_versao'] = None
    fato['data_geracao'] = None
    fato = fato[['semana_epi_id', 'municipio_id', 'tipo_registro', 'casos',
                 'incidencia', 'incidencia_norm', 'is_surto', 'modelo_versao', 'data_geracao']]

    dim_tempo.to_parquet("data/gold/dim_tempo_epidemiologico.parquet", index=False)
    dim_municipio.to_parquet("data/gold/dim_municipio.parquet", index=False)
    fato.to_parquet("data/gold/fato_casos_dengue.parquet", index=False)

    print(f"Gold salvo: {len(fato)} semanas | dim_tempo={len(dim_tempo)} | dim_municipio={len(dim_municipio)}")