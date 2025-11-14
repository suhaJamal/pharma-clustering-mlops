# Visualization Images

This folder contains the key visualizations from the analysis.

## Generating Images

To generate the actual visualization images, run:

```bash
python scripts/generate_images.py
```

This will create the following images:
1. `01_data_availability_heatmap.png` - Data completeness heatmap (2011-2020)
2. `02_countries_per_year.png` - Country coverage over time
3. `03_correlation_heatmap.png` - Feature correlation matrix
4. `04_elbow_silhouette.png` - Cluster optimization analysis
5. `05_cluster_distributions.png` - Feature distributions by cluster

## Requirements

Make sure you have installed all dependencies before running the script:

```bash
pip install -r ../requirements.txt
```

## Alternative

You can also view all visualizations by running the Jupyter notebooks in the `notebooks/` folder.
