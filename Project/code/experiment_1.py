'''
Experiment I: Compute raw time series of vegetation and water indices (e.g., NDVI, NDWI, MNDWI)
from BDC images over Ouro Preto (MG), using all available scenes regardless of cloud cover.

This script performs the following tasks:

1. Defines a bounding box over the Ouro Preto region (MG).
2. Queries Sentinel-2 data (S2-16D-2) from the Brazil Data Cube between 2019 and 2025.
3. Reads spectral bands (B02, B03, B04, B05, B08, B11) and generates RGB and band visualizations.
4. Computes a variety of spectral indices such as NDVI, NDWI, MNDWI, EVI, SAVI, CDI, etc.
5. Plots maps of spectral indices for a selected item.
6. Computes time series of indices for the entire time range without any cloud masking.
7. Plots and saves the resulting time series.

The objective is to analyze the raw temporal behavior of spectral indices over time, without applying
any cloud filtering or masking, to assess the unprocessed data quality.
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

if __name__ == '__main__':
    X = 0.028  # Incremento para o bbox

    # Criar pasta para resultados
    path = 'results'
    os.makedirs(path, exist_ok=True)

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

    # Definir bandas de interesse
    band_names = ['B02', 'B03', 'B04', 'B05', 'B08', 'B11']
    all_bands_data = cube.read_all_bands(band_names)

    # Selecionar um item para visualização (exemplo: o sexto de trás pra frente)
    selected_bands = all_bands_data[-1]  # dicionário com bandas

    blue = selected_bands['B02']
    green = selected_bands['B03']
    red = selected_bands['B04']
    red_edge = selected_bands['B05']
    nir = selected_bands['B08']
    swir = selected_bands['B11']

    rgb = np.dstack((
        cube.normalize(red),
        cube.normalize(green),
        cube.normalize(blue)
    ))

    # === Salvar imagens das bandas ===
    cube.plot_bands({
        'Blue (B02)': blue,
        'Green (B03)': green,
        'Red (B04)': red,
        'Red Edge (B05)': red_edge,
        'NIR (B08)': nir,
        'SWIR (B11)': swir,
        'RGB Composite': rgb
    }, item=cube.items[-1], save_path=os.path.join(path, 'bands.pdf'))

    # === Índices espectrais ===
    indices_dict = {
        'NDVI': cube.compute_index('NDVI', nir=nir, red=red),
        'NDRE': cube.compute_index('NDRE', nir=nir, red_edge=red_edge),
        'NDWI': cube.compute_index('NDWI', nir=nir, swir1=swir),
        'MNDWI': cube.compute_index('MNDWI', green=green, swir1=swir),
        'CDI': cube.compute_index('CDI', swir1=swir, blue=blue),
        'GNDVI': cube.compute_index('GNDVI', nir=nir, green=green),
        'SAVI': cube.compute_index('SAVI', nir=nir, red=red),
        'EVI': cube.compute_index('EVI', nir=nir, red=red, blue=blue),
        'MSAVI': cube.compute_index('MSAVI', nir=nir, red=red),
        'ARVI': cube.compute_index('ARVI', nir=nir, red=red, blue=blue),
        'NDMI': cube.compute_index('NDMI', nir=nir, swir1=swir),
        'BSI': cube.compute_index('BSI', swir1=swir, red=red, nir=nir, blue=blue)
    }

    cube.plot_indices(indices_dict, save_path=os.path.join(path, 'indices_plot.pdf'))

    # === Computar índices para todos os itens (paralelo) ===
    df_all = cube.compute_all_indices_parallel()
    print(df_all.head())

    # === Séries temporais ===
    cube.plot_time_series_indices(save_path=os.path.join(path, 'time_series_indices.pdf'))

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
