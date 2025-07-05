import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt

"""
Experiment IV: Analyze temporal patterns and structural changes in spectral index time series
from Brazil Data Cube (BDC) data, focusing on NDVI, NDMI, and NBR.

This experiment performs the following steps:

1. Loads previously saved time series of spectral indices from CSV files (masked and unmasked).
2. For selected indices (NDVI, NDMI, NBR), performs seasonal decomposition into trend,
   seasonality, and residual components using a fixed period (e.g., 23 for Sentinel-2 16-day composites).
3. Detects structural changes (changepoints) in the time series using the PELT algorithm
   with an RBF model from the ruptures library.
4. Plots and saves the decomposition and changepoint detection results to PDF files
   for both unmasked and cloud-masked datasets.

The objective is to understand long-term temporal behavior, seasonal patterns,
and abrupt changes in vegetation and moisture indices over time.
"""

def analyze_index_series(all_indices, indices=['NDVI', 'NDMI', 'NBR'], period=23, output_dir='analysis_results'):
    """
    Faz análise de séries temporais para índices espectrais:
    - decomposição (tendência, sazonalidade, ruído)
    - detecção de rupturas estruturais (changepoints)
    
    Parâmetros:
    - all_indices: pd.DataFrame com colunas ['datetime', 'value', 'index_name']
    - indices: lista de índices a analisar (default: ['NDVI', 'NDMI', 'NBR'])
    - period: número de observações por ciclo (ex: 23 para S2-16D)
    - output_dir: pasta onde salvar os plots
    """
    sns.set_style('darkgrid')
    os.makedirs(output_dir, exist_ok=True)

    palette = ['#024873FF', '#A2A637FF', '#D9AA1EFF', '#D98825FF', '#BF4F26FF']

    for index in indices:
        df = all_indices[all_indices['index_name'] == index].copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.dropna().sort_values('datetime')
        df.set_index('datetime', inplace=True)

        if len(df) < period * 2:
            print(f"[AVISO] Série muito curta para decomposição: {index}")
            continue

        # --- Decomposição ---
        try:
            result = seasonal_decompose(df['value'], model='additive', period=period)
        except Exception as e:
            print(f"[ERRO] Decomposição falhou para {index}: {e}")
            continue

        # --- Rupturas ---
        try:
            serie = df['value'].values
            dates = df.index

            algo = rpt.Pelt(model='rbf').fit(serie)
            for pen in [10, 5, 2, 1, 0.5]:
                breaks = algo.predict(pen=pen)
                if len(breaks) > 1:
                    break

            # Imprime as datas dos pontos de ruptura (exceto o último, que é o fim da série)
            rupture_dates = [dates[bk - 1].strftime('%Y-%m-%d') for bk in breaks[:-1]]
            print(f"Rupturas detectadas para {index}: {rupture_dates}")

        except Exception as e:
            print(f"[ERRO] Ruptura falhou para {index}: {e}")
            breaks = []

        # --- Figura com todos os componentes ---
        fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

        result.observed.plot(ax=axs[0], label='Original', color=palette[0], marker='o', markersize=4)
        axs[0].set_ylabel('Observed')
        axs[0].legend(facecolor='white')

        result.trend.plot(ax=axs[1], label='Trend', color=palette[1], marker='o', markersize=4)
        axs[1].set_ylabel('Trend')
        axs[1].legend(facecolor='white')

        result.seasonal.plot(ax=axs[2], label='Seasonal', color=palette[2], marker='o', markersize=4)
        axs[2].set_ylabel('Seasonal')
        axs[2].legend(facecolor='white')

        result.resid.plot(ax=axs[3], label='Residuals', color=palette[3], marker='o', markersize=4)
        axs[3].set_ylabel('Residuals')
        axs[3].legend(facecolor='white')

        axs[4].plot(dates, serie, label=index, color=palette[4], marker='o', markersize=4)
        for i, bk in enumerate(breaks[:-1]):
            axs[4].axvline(dates[bk - 1], color='black', linestyle='--', alpha=0.7,
                           label='Change Point' if i == 0 else None)
        axs[4].set_ylabel('Value')
        axs[4].set_title("Change Point Detection")
        axs[4].legend(facecolor='white')

        fig.suptitle(f"{index} – Decomposition & Change Points", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f"{index}_decomposition_changepoints.pdf"), dpi=300)
        plt.close()
        print(f"✅ Salvo: {index}_decomposition_changepoints.pdf")

if __name__ == '__main__':
    csv_path = 'results/all_indices.csv'
    csv_path_masked = 'results/all_indices_masked.csv'

    df = pd.read_csv(csv_path)
    df_masked = pd.read_csv(csv_path_masked)

    analyze_index_series(df, indices=['NDVI', 'NDMI', 'NBR'], period=23, output_dir='analysis_from_csv')
    analyze_index_series(df_masked, indices=['NDVI', 'NDMI', 'NBR'], period=23, output_dir='analysis_from_csv_masked')
