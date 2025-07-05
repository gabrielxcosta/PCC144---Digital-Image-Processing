#!/bin/bash

pip install pystac-client
# Instala a biblioteca `pystac-client`, usada para acessar catálogos STAC como o do Brazil Data Cube

pip install wtss==2.0.0a3
# Instala a versão alfa da biblioteca `wtss` (Web Time Series Service), útil para séries temporais remotas

pip install rasterio
# Instala a biblioteca `rasterio`, usada para leitura e manipulação de dados raster (imagens geográficas)

pip install statsmodels
# Instala `statsmodels`, biblioteca estatística que inclui decomposição de séries temporais (ex: `seasonal_decompose`)

pip install ruptures
# Instala `ruptures`, usada para detecção de pontos de ruptura (changepoints) em séries temporais

# chmod +x setup.sh
# ./setup.sh
# source /media/work/fernandoduarte/ASOP/venv/bin/activate