import os                            
import time                          
import math                          
import shapely                       
import rasterio                      
import numpy as np                   
import pystac_client                
import warnings as w                
import seaborn as sns                
import pandas as pd                 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wtss import *                   
from wtss import WTSS                
from rasterio.crs import CRS         
from datetime import datetime        
import multiprocessing as mp         
import matplotlib.gridspec as gridspec     
from rasterio.warp import transform        
from rasterio.windows import from_bounds   
from matplotlib.ticker import MaxNLocator, FuncFormatter  
w.filterwarnings('ignore')

def get_item_datetime(self, item):
    """
    Returns the formatted datetime string of a data cube item.

    Attempts to extract and format the `datetime` attribute from the given item
    using ISO 8601 format with microsecond precision and a 'Z' suffix to indicate UTC.

    Parameters:
        item: An object expected to have a `datetime` attribute (of type `datetime.datetime`).

    Returns:
        str: A formatted datetime string in the format 'YYYY-MM-DDTHH:MM:SS.ffffffZ' if available.
        None: If the item does not have a `datetime` attribute.

    Example:
        '2023-07-12T14:30:15.123456Z'
    """
    try:
        return item.datetime.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    except AttributeError:
        return None

def human_format(x, pos):
    """
    Format large numbers using human-readable suffixes (e.g., 'k' for thousands, 'M' for millions).

    Parameters:
    - x (float): The number to format.
    - pos: Required for compatibility with matplotlib's formatter (unused).

    Returns:
    - str: Formatted string with appropriate suffix.
    """
    if x >= 1e6:
        return f'{x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}k'
    else:
        return f'{x:.0f}'

def read_item_bands(args):
    """
    Helper to read multiple bands from a single item (used in parallel processing).

    Parameters:
    - args (tuple): (item, band_names, bbox, crs)

    Returns:
    - dict: {band_name: array}
    """
    item, band_names, bbox, crs = args
    result = {}
    try:
        source_crs = CRS.from_string('EPSG:4326') if crs is None else CRS.from_string(crs)
        w, s, e, n = bbox
        for band_name in band_names:
            uri = item.assets[band_name].href
            with rasterio.open(uri) as dataset:
                transformer = transform(source_crs, dataset.crs, [w, e], [s, n])
                window = from_bounds(transformer[0][0], transformer[1][0],
                                     transformer[0][1], transformer[1][1], dataset.transform)
                band_array = dataset.read(1, window=window, masked=True)
                result[band_name] = band_array
        return result
    except Exception as e:
        print(f"[ERRO] ao ler bandas do item {item.id}: {e}")
        return None

def process_index_item(self, item, band_dict, index_name, mask_clouds=False, cloud_threshold=0.0):
    """
    Compute a single index for one item, with optional cloud masking using CDI.

    Parameters:
    - item: STAC item metadata.
    - band_dict: Dictionary with band arrays for the current item.
    - index_name (str): Spectral index to compute.
    - mask_clouds (bool): Whether to mask cloudy pixels using CDI.
    - cloud_threshold (float): Threshold for CDI to define cloudy pixels.

    Returns:
    - tuple: (timestamp, mean_value, index_name) or None on failure.
    """
    band_map = {
        'NDVI': {'nir': 'B08', 'red': 'B04'},
        'NDRE': {'nir': 'B08', 'red_edge': 'B05'},
        'NDWI': {'nir': 'B08', 'swir1': 'B11'},
        'MNDWI': {'green': 'B03', 'swir1': 'B11'},
        'CDI': {'blue': 'B02', 'swir1': 'B11'},
        'GNDVI': {'nir': 'B08', 'green': 'B03'},
        'SAVI': {'nir': 'B08', 'red': 'B04'},
        'EVI': {'nir': 'B08', 'red': 'B04', 'blue': 'B02'},
        'MSAVI': {'nir': 'B08', 'red': 'B04'},
        'ARVI': {'nir': 'B08', 'red': 'B04', 'blue': 'B02'},
        'NDMI': {'nir': 'B08', 'swir1': 'B11'},
        'BSI': {'swir1': 'B11', 'red': 'B04', 'nir': 'B08', 'blue': 'B02'},
        'NBR': {'nir': 'B08', 'swir1': 'B11'}
    }

    try:
        bands = band_map[index_name]
        kwargs = {alias: band_dict.get(band) for alias, band in bands.items()}

        if any(v is None for v in kwargs.values()):
            raise ValueError(f"Missing required bands for {index_name}: {kwargs}")

        index_array = self.compute_index(index_name, **kwargs)

        # Aplica máscara de nuvem baseada no CDI, se necessário
        if mask_clouds and index_name != 'CDI':
            try:
                cdi_kwargs = {
                    'swir': band_dict.get('B11'),
                    'blue': band_dict.get('B02')
                }
                if any(v is None for v in cdi_kwargs.values()):
                    raise ValueError("Missing bands for CDI computation.")
                cdi_array = self.compute_index('CDI', **cdi_kwargs)
                cloud_mask = cdi_array < cloud_threshold
                index_array = np.where(cloud_mask, np.nan, index_array)
            except Exception as e:
                print(f"[ERRO] Falha ao aplicar máscara de nuvem (CDI) para {index_name}: {e}")

        mean_value = float(np.nanmean(index_array))
        timestamp = self.get_item_datetime(item)
        return (timestamp, mean_value, index_name)

    except Exception as e:
        print(f"[ERRO] {self.get_item_datetime(item)} / {index_name}: {e}")
        return None

class BDCDataCube:
    """
    A class for handling satellite image collections from the Brazil Data Cube (BDC) using the STAC API.

    Main features:
    - Connect to the BDC STAC service.
    - Search satellite image items using bounding box and date range.
    - Read and normalize spectral bands from satellite images.
    - Compute vegetation and water indices (NDVI, NDRE, NDWI, MNDWI, CDI).
    - Visualize bands and histograms.
    - Compute and save time series for indices.
    """
    # === Initialization ===
    def __init__(self, stac_url='https://data.inpe.br/bdc/stac/v1/'):
        """
        Initialize the BDCDataCube instance by connecting to the STAC API.

        Parameters:
        - stac_url (str): URL of the STAC service.
        """
        self.service_stac = pystac_client.Client.open(stac_url)
        self.items = []
        self.bbox = None
        self.collection = None
        self.datetime = None
        self.indices_dataframes = {}

    # === Querying STAC ===
    def set_query(self, collection, bbox, datetime):
        """
        Set the query parameters to search for image items.

        Parameters:
        - collection (str): Name of the satellite image collection (e.g., 'S2-16D-2').
        - bbox (tuple): Bounding box (west, south, east, north).
        - datetime (str): Date range (format: 'YYYY-MM-DD/YYYY-MM-DD').
        """
        self.collection = collection
        self.bbox = bbox
        self.datetime = datetime

    def search_items(self):
        """
        Search the STAC service for items using the current query parameters.

        Returns:
        - int: Number of items found.
        """
        print(self.bbox)
        item_search = self.service_stac.search(
            collections=[self.collection],
            bbox=self.bbox,
            datetime=self.datetime
        )
        self.items = list(item_search.items())
        return len(self.items)

    @staticmethod
    def get_item_datetime(item):
        """
        Retrieve the acquisition datetime from a STAC item.

        Parameters:
        - item: STAC item object.

        Returns:
        - str: Datetime string in the format 'YYYY-MM-DD HH:MM:SS', or 'Unknown DateTime'.
        """
        if 'datetime' in item.properties:
            datetime_iso = item.properties['datetime']
            date_part, time_part = datetime_iso.split('T')
            time_part = time_part.replace('Z', '')  # Remove trailing Z if present
            return f'{date_part} {time_part}'
        else:
            return 'Unknown DateTime'

    # === Data Retrieval ===
    def read_all_bands(self, band_names, crs=None):
        """
        Read specified bands from all items using parallel processing.

        Parameters:
        - band_names (list): List of band names (e.g., ['B04', 'B08']).
        - crs (str): Desired CRS (default: 'EPSG:4326').

        Returns:
        - list of dict: Each dict contains {band_name: array} for one item.
        """
        args = [(item, band_names, self.bbox, crs) for item in self.items]
        with mp.Pool(mp.cpu_count() - 1) as pool:
            results = pool.map(read_item_bands, args)
        return [r for r in results if r is not None]

    def compute_all_indices_parallel(self, mask_clouds=False, cloud_threshold=0.0):
        """
        Compute all supported spectral indices in parallel.

        Parameters:
        - mask_clouds (bool): Whether to mask cloudy pixels using CDI before computing other indices.
        - cloud_threshold (float): CDI threshold below which pixels are considered clouds (default: 0.0).

        Returns:
        - pd.DataFrame: Time series of all indices.
        """
        supported_indices = [
            'NDVI', 'NDRE', 'NDWI', 'MNDWI', 'CDI',
            'GNDVI', 'SAVI', 'EVI', 'MSAVI', 'ARVI',
            'NDMI', 'BSI', 'NBR'
        ]

        band_map = {
            'NDVI': {'nir': 'B08', 'red': 'B04'},
            'NDRE': {'nir': 'B08', 'red_edge': 'B05'},
            'NDWI': {'nir': 'B08', 'swir1': 'B11'},
            'MNDWI': {'green': 'B03', 'swir1': 'B11'},
            'CDI': {'blue': 'B02', 'swir1': 'B11'},
            'GNDVI': {'nir': 'B08', 'green': 'B03'},
            'SAVI': {'nir': 'B08', 'red': 'B04'},
            'EVI': {'nir': 'B08', 'red': 'B04', 'blue': 'B02'},
            'MSAVI': {'nir': 'B08', 'red': 'B04'},
            'ARVI': {'nir': 'B08', 'red': 'B04', 'blue': 'B02'},
            'NDMI': {'nir': 'B08', 'swir1': 'B11'},
            'BSI': {'swir1': 'B11', 'red': 'B04', 'nir': 'B08', 'blue': 'B02'},
            'NBR': {'nir': 'B08', 'swir1': 'B11'}  
        }

        # Coletar todas as bandas necessárias
        all_bands = set()
        for bands in band_map.values():
            all_bands.update(bands.values())
        all_band_arrays = self.read_all_bands(band_names=list(all_bands))

        item_band_dict = {
            item: band_dict for item, band_dict in zip(self.items, all_band_arrays)
        }

        results = []

        for item, band_dict in item_band_dict.items():
            cloud_mask = None

            if mask_clouds:
                try:
                    cdi_array = self.compute_index('CDI',
                                                swir1=band_dict.get('B11'),
                                                blue=band_dict.get('B02'))
                    cloud_mask = cdi_array < cloud_threshold
                except Exception as e:
                    print(f"[ERRO] Falha ao calcular CDI para máscara de nuvem: {e}")

            for index_name in supported_indices:
                try:
                    bands = band_map[index_name]
                    kwargs = {alias: band_dict.get(band) for alias, band in bands.items()}
                    index_array = self.compute_index(index_name, **kwargs)

                    if mask_clouds and cloud_mask is not None and index_name != 'CDI':
                        index_array = np.where(cloud_mask, np.nan, index_array)

                    mean_value = float(np.nanmean(index_array))
                    timestamp = self.get_item_datetime(item)
                    results.append((timestamp, mean_value, index_name))
                except Exception as e:
                    print(f"[ERRO] {self.get_item_datetime(item)} / {index_name}: {e}")
                    continue

        df = pd.DataFrame(results, columns=['datetime', 'value', 'index_name'])
        df = df.sort_values(['index_name', 'datetime']).reset_index(drop=True)

        self.indices_dataframes = {
            index: df[df['index_name'] == index][['datetime', 'value']].reset_index(drop=True)
            for index in supported_indices
        }

        return df

    def compute_index(self, index_name, **kwargs):
        """
        Compute the specified spectral index using the provided bands.

        Parameters:
        - index_name (str): The name of the index to compute (e.g., 'NDVI', 'NDRE').
        - **kwargs: Variable length arguments with band arrays (e.g., nir, red, green, etc.)

        Returns:
        - Computed index as an array.

        Raises:
        - ValueError: If required bands are missing or index is unknown.
        """
        indices = {
            'NDVI': (['nir', 'red'], lambda b: (b['nir'] - b['red']) / (b['nir'] + b['red'])),
            'NDRE': (['nir', 'red_edge'], lambda b: (b['nir'] - b['red_edge']) / (b['nir'] + b['red_edge'])),
            'NDWI': (['nir', 'swir1'], lambda b: (b['nir'] - b['swir1']) / (b['nir'] + b['swir1'])),
            'MNDWI': (['green', 'swir1'], lambda b: (b['green'] - b['swir1']) / (b['green'] + b['swir1'])),
            'CDI': (['swir1', 'blue'], lambda b: (b['swir1'] - b['blue']) / (b['swir1'] + b['blue'])),
            'GNDVI': (['nir', 'green'], lambda b: (b['nir'] - b['green']) / (b['nir'] + b['green'])),
            'SAVI': (['nir', 'red'], lambda b: ((b['nir'] - b['red']) / (b['nir'] + b['red'] + 0.5)) * 1.5),
            'EVI': (['nir', 'red', 'blue'], lambda b: 2.5 * (b['nir'] - b['red']) / (b['nir'] + 6 * b['red'] - 7.5 * b['blue'] + 1)),
            'MSAVI': (['nir', 'red'], lambda b: (2 * b['nir'] + 1 - np.sqrt((2 * b['nir'] + 1)**2 - 8 * (b['nir'] - b['red']))) / 2),
            'ARVI': (['nir', 'red', 'blue'], lambda b: (b['nir'] - b['red']) / (b['nir'] + b['red'] - 2 * b['blue'])),
            'NDMI': (['nir', 'swir1'], lambda b: (b['nir'] - b['swir1']) / (b['nir'] + b['swir1'])),
            'BSI': (['swir1', 'red', 'nir', 'blue'], lambda b: (b['blue'] + b['red'] - b['nir'] - b['swir1']) / (b['blue'] + b['red'] + b['nir'] + b['swir1'])),
            'NBR': (['nir', 'swir1'], lambda b: (b['nir'] - b['swir1']) / (b['nir'] + b['swir1']))  # CORRIGIDO para 'swir1'
        }

        if index_name not in indices:
            raise ValueError(f"Index '{index_name}' is not recognized.")

        required_bands, formula = indices[index_name]
        missing = [band for band in required_bands if kwargs.get(band) is None]

        if missing:
            raise ValueError(f"{index_name} requires the following bands: {', '.join(required_bands)}. Missing: {', '.join(missing)}")

        return formula(kwargs)

    # === Data Processing ===
    @staticmethod
    def normalize(array):
        """
        Normalize a numpy array to range [0, 1].

        Parameters:
        - array (np.ndarray): Input array

        Returns:
        - np.ndarray: Normalized array
        """
        array = array.astype('float32')
        array = (array - array.min()) / (array.max() - array.min())
        return np.clip(array, 0, 1)
    
    def mask_clouds_by_cdi(index_array, cdi_array, threshold=0.0):
        """
        Aplica máscara de nuvens baseada em valores baixos de CDI.
        Pixels com CDI < limiar são mascarados como NaN.

        Parameters:
        - index_array (np.ndarray): Índice espectral a ser mascarado (ex: NDVI, NDWI).
        - cdi_array (np.ndarray): Matriz do CDI correspondente.
        - threshold (float): Limite inferior de CDI para identificar nuvens. Default: 0.0

        Returns:
        - np.ndarray: Índice com valores de nuvem mascarados como NaN.
        """
        mask = cdi_array < threshold
        return np.where(mask, np.nan, index_array)

    # === Visualization ===
    def plot_bands(self, bands_dict, item=None, save_path='bands.pdf'):
        """
        Display multiple bands in subplots and save the visualization to PDF.
        The last image is centered in the row if the number of bands is not a multiple of columns.
        """
        sns.set_style('white')
        sns.set_context('notebook', font_scale=1.5)

        band_names = list(bands_dict.keys())
        band_data_list = list(bands_dict.values())
        n_bands = len(band_names)

        n_cols = 3
        n_rows = math.ceil(n_bands / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True)
        axs = axs.ravel()

        remainder = n_bands % n_cols

        # Plot all except the last if it needs to be centered manually
        plot_upto = n_bands if remainder == 0 else n_bands - 1
        for i in range(plot_upto):
            ax = axs[i]
            cmap = None if 'RGB' in band_names[i] else 'gray'
            ax.imshow(band_data_list[i], cmap=cmap)
            ax.set_title(band_names[i], fontsize=16)
            ax.axis('off')

        # Hide unused axes
        for i in range(plot_upto, n_rows * n_cols):
            axs[i].remove()

        # Center the last image if needed
        if remainder != 0:
            empty_slots = n_cols - remainder
            last_row_start = (n_rows - 1) * n_cols + 1

            # Add left empty slots
            for i in range(empty_slots // 2):
                fig.add_subplot(n_rows, n_cols, last_row_start + i).axis('off')

            # Add centered last image
            center_idx = last_row_start + empty_slots // 2
            ax = fig.add_subplot(n_rows, n_cols, center_idx)
            cmap = None if 'RGB' in band_names[-1] else 'gray'
            ax.imshow(band_data_list[-1], cmap=cmap)
            ax.set_title(band_names[-1], fontsize=16)
            ax.axis('off')

            # Add right empty slots
            for i in range(empty_slots - empty_slots // 2):
                fig.add_subplot(n_rows, n_cols, center_idx + 1 + i).axis('off')

        # Add datetime title if available
        if item is not None:
            datetime_str = self.get_item_datetime(item)
            clean_datetime = datetime_str.split('.')[0]
            fig.suptitle(f'Bands Visualization - Date & Time: {clean_datetime}', fontsize=20)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_histograms(self, bands_dict, save_path='band_histograms.pdf'):
        """
        Plot overlaid histograms of the provided bands and save the result to a PDF.

        Parameters:
        - bands_dict (dict): Dictionary {band_name: band_array}.
        - save_path (str): Path to save the output PDF file.
        """
        plt.figure(figsize=(12, 8))
        sns.set_style('darkgrid')
        sns.set_context('notebook', font_scale=1.8)  # Larger fonts

        colors = {
            'Blue (B02)': 'blue',
            'Green (B03)': 'green',
            'Red (B04)': 'red',
            'NIR (B08)': 'orange',
            'SWIR (B11)': "#431c70",
            'Red Edge (B05)': "#5e0c0c"
        }

        for band_name, band_data in bands_dict.items():
            sns.histplot(
                band_data.ravel(),
                bins=100,
                color=colors.get(band_name, 'gray'),
                label=band_name,
                kde=False,
                alpha=0.5
            )

        plt.title('Band Histograms')
        plt.xlabel('Reflectance')
        plt.ylabel('Frequency')
        plt.legend(facecolor='white')
        plt.xlim(0, 8000)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(human_format))

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_indices(self, indices_dict, save_path='indices_plot.pdf'):
        """
        Plot a grid of spectral index images with vertical colorbars.

        Parameters
        ----------
        indices_dict : dict[str, np.ndarray]
            Dictionary with spectral index names as keys and 2D arrays as values.
        save_path : str
            Output path to save the PDF plot. Default is 'indices_plot.pdf'.
        """
        n_rows, n_cols = 3, 4
        fig = plt.figure(figsize=(24, 18))
        total = n_rows * n_cols

        # Layout config
        left, right, top, bottom = 0.05, 0.95, 0.95, 0.05
        h_gap, cbar_gap, v_gap = 0.04, 0.005, 0.03
        total_width = right - left
        width_col = (total_width - (n_cols - 1) * h_gap) / n_cols
        img_w = width_col * 0.965
        cbar_w = width_col * 0.03
        img_w -= cbar_gap
        cbar_offset = img_w + cbar_gap

        total_height = top - bottom
        height_row = (total_height - (n_rows - 1) * v_gap) / n_rows

        for i, (title, data) in enumerate(indices_dict.items()):
            if i >= total:
                break  # Limit to 12 plots

            row, col = divmod(i, n_cols)
            x_img = left + col * (width_col + h_gap)
            y_img = top - (row + 1) * height_row - row * v_gap

            ax_img = fig.add_axes([x_img, y_img, img_w, height_row])
            cmap = self._get_cmap_for_index(title)
            vmin, vmax = self._get_vmin_vmax_for_index(title)
            im = ax_img.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax_img.set_title(title, fontsize=14)
            ax_img.axis('off')

            ax_cbar = fig.add_axes([x_img + cbar_offset, y_img, cbar_w, height_row])
            cbar = fig.colorbar(im, cax=ax_cbar)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(title, rotation=270, labelpad=18)
            ax_cbar.yaxis.set_ticks_position('right')
            ax_cbar.yaxis.set_label_position('right')

        # Fill empty slots if less than 12 indices
        for j in range(len(indices_dict), total):
            row, col = divmod(j, n_cols)
            x_img = left + col * (width_col + h_gap)
            y_img = top - (row + 1) * height_row - row * v_gap
            fig.add_axes([x_img, y_img, img_w, height_row]).axis('off')
            fig.add_axes([x_img + cbar_offset, y_img, cbar_w, height_row]).axis('off')

        fig.suptitle('Calculated Spectral Indices', fontsize=24, y=0.98)
        plt.savefig(save_path)
        plt.close()

    def _get_cmap_for_index(self, index_title):
        """
        Get the colormap for each index.

        Parameters:
        - index_title (str): The title of the index (e.g. 'NDVI', 'NDRE')

        Returns:
        - cmap (str): The colormap to use for that index
        """
        cmap_dict = {
            'NDVI': 'RdYlGn',
            'NDRE': 'PuOr',
            'NDWI': 'BrBG',
            'MNDWI': 'Blues',
            'CDI': 'viridis',
            'GNDVI': 'YlGn',
            'SAVI': 'Spectral',
            'EVI': 'coolwarm',
            'MSAVI': 'cividis',
            'ARVI': 'copper',
            'NDMI': 'pink',
            'BSI': 'inferno'
        }
        return cmap_dict.get(index_title, 'viridis')  # Default to 'viridis' if index title is not found

    def _get_vmin_vmax_for_index(self, index_title):
        """
        Get the vmin and vmax for each index.

        Parameters:
        - index_title (str): The title of the index (e.g. 'NDVI', 'NDRE')

        Returns:
        - vmin, vmax (float, float): Minimum and maximum values for the colormap scaling
        """
        vmin_vmax_dict = {
            'NDVI': (-1, 1),
            'NDRE': (-1, 1),
            'NDWI': (-1, 1),
            'MNDWI': (-1, 1),
            'CDI': (-1, 1),
            'GNDVI': (-1, 1),
            'SAVI': (-1, 1),
            'EVI': (-2, 2),
            'MSAVI': (-1, 1),
            'ARVI': (-1, 1),
            'NDMI': (-1, 1),
            'BSI': (-1, 1)
        }
        return vmin_vmax_dict.get(index_title, (0, 1))  # Default range if index title is not found
    
    def plot_time_series_indices(self, save_path=None, df_dict=None):
        """
        Plot time series of spectral indices grouped into categories:
        - Vegetation indices
        - Moisture indices
        - Bare soil and cloud indices
        - Burn index (NBR)

        Parameters:
        - save_path (str, optional): If provided, saves the plot to the given file path.
        - df_dict (dict, optional): Dictionary of DataFrames where each key is an index name
        and each value is a DataFrame with columns ['datetime', 'value'].
        """
        sns.set_style('darkgrid')
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 12), sharex=True)

        vegetation_indices = ['NDVI', 'EVI', 'SAVI', 'MSAVI', 'ARVI', 'GNDVI', 'NDRE']
        moisture_indices = ['NDWI', 'MNDWI', 'NDMI']
        soil_cloud_indices = ['BSI', 'CDI']
        nbr_indices = ['NBR']

        category_map = {
            'Vegetation Indices': (vegetation_indices, axes[0]),
            'Moisture Indices': (moisture_indices, axes[1]),
            'Bare Soil and Cloud Indices': (soil_cloud_indices, axes[2]),
            'Burn Index (NBR)': (nbr_indices, axes[3])
        }

        # Nova paleta fornecida (sem alpha)
        full_palette = [
            "#950404", "#E04B28", "#C38961", "#9F5630",
            "#388F30", "#0F542F", "#007D82", "#004042"
        ]

        nbr_color = full_palette[0]  # cor fixa para NBR

        data_source = df_dict if df_dict is not None else self.indices_dataframes

        def get_interpolated_colors(palette, n):
            """Interpola cores entre as fornecidas para produzir `n` cores."""
            base_rgb = [mcolors.to_rgb(c) for c in palette]
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_palette", base_rgb)
            return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)] if n > 1 else [mcolors.to_hex(base_rgb[0])]

        for category_title, (index_list, ax) in category_map.items():
            existing_indices = [i for i in index_list if i in data_source]
            n_curves = len(existing_indices)

            if category_title == 'Burn Index (NBR)':
                colors = [nbr_color]
            else:
                colors = get_interpolated_colors(full_palette, n_curves)

            for color, index_name in zip(colors, existing_indices):
                df = data_source[index_name].copy()
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.dropna(subset=['datetime', 'value']).sort_values('datetime')  # <- Corrige saltos
                ax.plot(
                    df['datetime'],
                    df['value'],
                    marker='.',
                    markersize=3,
                    linewidth=1.3,
                    color=color,
                    label=index_name
                )

            ax.set_title(category_title)
            ax.set_ylabel('Index Value')
            ax.grid(True)
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
                frameon=True,
                facecolor='white'
            )

        axes[-1].set_xlabel('Date')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()