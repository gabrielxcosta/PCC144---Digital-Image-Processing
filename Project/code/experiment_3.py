"""
Experiment III: Evaluate the effect of cloud masking on spectral time series.

This script compares spectral index time series with and without the application of a cloud mask.
It uses CDI (Cloud Discrimination Index) masking to remove cloud-contaminated pixels from the analysis.

Key steps:
- Load two time series datasets: one with cloud masking applied, and one without.
- For each common spectral index (e.g., NDVI, NDWI, NBR):
    - Merge both time series on datetime.
    - Compute correlation, MAE (Mean Absolute Error), and maximum absolute difference.
    - Plot both time series side by side.
    - Plot absolute difference over time.
- Optionally, save:
    - A combined plot with all indices (`all_indices_comparison.pdf`)
    - A CSV file with all computed statistics (`comparison_statistics.csv`)
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def compare_indices_with_and_without_mask(path_with_mask, path_without_mask, save_stats=False, save_plots=False, output_folder='index_comparison'):
    """
    Compare time series of spectral indices between versions with and without cloud mask.

    Parameters:
    - path_with_mask (str): Path to CSV file containing data with cloud mask.
    - path_without_mask (str): Path to CSV file containing data without cloud mask.
    - save_stats (bool): If True, saves a CSV file with correlation, MAE, and max difference.
    - save_plots (bool): If True, saves a single plot with all index comparisons.
    - output_folder (str): Folder name to save output files, if applicable.
    """
    sns.set_style('darkgrid')  # ✅ Apply darkgrid style globally

    if save_stats or save_plots:
        os.makedirs(output_folder, exist_ok=True)

    # Load data
    df_with = pd.read_csv(path_with_mask, parse_dates=['datetime'])
    df_without = pd.read_csv(path_without_mask, parse_dates=['datetime'])

    # List of indices to compare
    indices = sorted(set(df_with['index_name']).intersection(df_without['index_name']))
    print("Índices encontrados para comparação:", indices)
    results = []

    print("NBR presente em df_with:", 'NBR' in df_with['index_name'].unique())
    print("NBR presente em df_without:", 'NBR' in df_without['index_name'].unique())


    # Adjusted height for compact layout (e.g., 2.5 inches per index)
    n_indices = len(indices)
    fig_height = max(2.5 * n_indices, 4)
    fig, axs = plt.subplots(nrows=n_indices, ncols=2, figsize=(14, fig_height), squeeze=False)

    for i, index in enumerate(indices):
        df_w = df_with[df_with['index_name'] == index].rename(columns={'value': 'value_with'})
        df_wo = df_without[df_without['index_name'] == index].rename(columns={'value': 'value_without'})
        df_merged = pd.merge(df_w[['datetime', 'value_with']], df_wo[['datetime', 'value_without']], on='datetime', how='inner')

        if df_merged.empty:
            print(f"[!] No common dates for index {index}. Skipping.")
            continue

        df_merged.dropna(inplace=True)
        df_merged['diff'] = (df_merged['value_without'] - df_merged['value_with']).abs()
        mae = mean_absolute_error(df_merged['value_with'], df_merged['value_without'])
        correl = df_merged[['value_with', 'value_without']].corr().iloc[0, 1]
        max_diff = df_merged['diff'].abs().max()

        results.append({
            'index': index,
            'correlation': correl,
            'MAE': mae,
            'max_absolute_difference': max_diff
        })

        # === Time Series Plot ===
        ax_ts = axs[i, 0]
        ax_ts.plot(df_merged['datetime'], df_merged['value_with'], label='With Mask', color='#FFC72CFF', alpha=0.6, marker='o', markersize=5)
        ax_ts.plot(df_merged['datetime'], df_merged['value_without'], label='Without Mask', color='red', alpha=0.4, marker='o', markersize=5)
        ax_ts.set_title(f'{index} – Time Series')
        ax_ts.set_xlabel('Date')
        ax_ts.set_ylabel('Index Value')
        ax_ts.legend(loc='best', fontsize='small', facecolor='white')
        ax_ts.grid(True)

        # === Difference Plot ===
        ax_diff = axs[i, 1]
        sns.lineplot(x='datetime', y='diff', data=df_merged, ax=ax_diff, marker='o', color='olive', alpha=0.6)
        ax_diff.axhline(0, color='black', linestyle='--', linewidth=1)
        ax_diff.set_title(f'{index} – Difference (No Mask - Mask)')
        ax_diff.set_xlabel('Date')
        ax_diff.set_ylabel('Difference')
        ax_diff.grid(True)

        print(f"✅ {index} | Correlation: {correl:.3f} | MAE: {mae:.4f} | Max Abs. Diff: {max_diff:.4f}")

    plt.tight_layout(h_pad=2)
    if save_plots:
        fig_path = os.path.join(output_folder, 'all_indices_comparison.pdf')
        plt.savefig(fig_path, dpi=300)
        print(f"\n📊 Combined plot saved to: {fig_path}")
    plt.show()
    plt.close()

    if save_stats:
        df_results = pd.DataFrame(results)
        stats_path = os.path.join(output_folder, 'comparison_statistics.csv')
        df_results.to_csv(stats_path, index=False)
        print(f"\n📁 Statistics saved to: {stats_path}")

if __name__ == '__main__':
    compare_indices_with_and_without_mask(
        path_with_mask='results/all_indices_masked.csv',
        path_without_mask='results/all_indices.csv',
        save_stats=True,
        save_plots=True
    )