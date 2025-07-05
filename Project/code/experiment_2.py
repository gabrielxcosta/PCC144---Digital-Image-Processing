'''
Experiment II: Evaluate the of cloud masking on spectral time series and get raw and filtered series.

This script uses Brazil Data Cube (BDC) to analyze the impact of applying a cloud mask
on spectral index time series derived from Sentinel-2 data. It performs the following steps:

1. Defines a bounding box and queries images from the S2-16D-2 collection via STAC API.
2. Loads necessary bands (B02, B03, B04, B05, B08, B11) for index computation.
3. Computes time series of spectral indices both with and without cloud masking (CDI < -0.0).
4. Saves and plots the resulting time series for visual comparison.
5. Exports the data to CSV files for further analysis.

The goal is to assess whether applying a cloud mask improves the quality and reliability
of vegetation and moisture indices over time.
'''
import os
import time
import shapely
import rasterio
import numpy as np
import pystac_client
from rasterio.crs import CRS
from rasterio.warp import transform
from rasterio.windows import from_bounds
from BDC import BDCDataCube
import matplotlib.pyplot as plt

def split_dataframe_by_index(df):
    """
    Divide um DataFrame único (com colunas 'index_name', 'datetime', 'value')
    em um dicionário de DataFrames, um para cada índice.
    """
    return {name: group.drop(columns='index_name').reset_index(drop=True)
            for name, group in df.groupby('index_name')}

if __name__ == '__main__':
    X = 0.028  # Incremento para o bbox

    # === Experimento II: Avaliar o impacto da máscara de nuvens ===
    base_path = 'results'

    bbox = (
        -43.51316331904583 - X,
        -20.40132828606603 - X,
        -43.49356331904583 + X,
        -20.38332828606603 + X
    )

    coverage_name = 'S2-16D-2'
    start_time = time.time()

    cube = BDCDataCube()
    cube.set_query(
        collection=coverage_name,
        bbox=bbox,
        datetime='2019-01-01/2025-05-30'
    )

    n_items = cube.search_items()
    print(f'{n_items} items found')

    # === Ler bandas necessárias ===
    band_names = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11']
    all_bands_data = cube.read_all_bands(band_names)

    # === Computar índices espectrais (sem máscara) ===
    print("\n[1/2] Computando índices SEM máscara de nuvem...")
    df_no_mask = cube.compute_all_indices_parallel(mask_clouds=False)

    # === Computar índices espectrais (com máscara) ===
    print("\n[2/2] Computando índices COM máscara de nuvem (EVI < -0.0)...")
    df_with_mask = cube.compute_all_indices_parallel(mask_clouds=True)

    # === Plotar as séries temporais para os dois casos ===
    cube.plot_time_series_indices(
        save_path=os.path.join(base_path, 'time_series_indices_raw.pdf'),
        df_dict=split_dataframe_by_index(df_no_mask)
    )
    cube.plot_time_series_indices(
        save_path=os.path.join(base_path, 'time_series_indices_cloud_mask.pdf'),
        df_dict=split_dataframe_by_index(df_with_mask)
    )

    # === Salvar os dataframes brutos ===
    df_no_mask.to_csv(os.path.join(base_path, 'all_indices.csv'), index=False)
    df_with_mask.to_csv(os.path.join(base_path, 'all_indices_masked.csv'), index=False)

    print(f"\n✅ Experimento II concluído.")
    print(f"⏱ Tempo total de execução: {time.time() - start_time:.2f} segundos")

