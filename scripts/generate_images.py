"""
Generate Key Visualizations for README
========================================
This script generates PNG images of key visualizations from the analysis
and saves them to the images/ folder for inclusion in the README.

Run this script from the project root directory:
    python scripts/generate_images.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("="*70)
print("GENERATING KEY VISUALIZATIONS FOR README")
print("="*70)

# ============================================================================
# IMAGE 1: Data Availability Heatmap (2011-2020)
# Source: 01_data_cleaning.ipynb
# ============================================================================
print("\n[1/5] Generating Data Availability Heatmap...")

# Load the data
df = pd.read_csv('data/processing/eda_data.csv')

# Filter for 2011-2020
start_year = 2011
end_year = 2020
df_filtered = df[(df['YEAR'] >= start_year) & (df['YEAR'] <= end_year)]

# Create year range and get countries
years_range = range(start_year, end_year + 1)
countries = sorted(df_filtered['COUNTRY'].unique())
num_years = len(years_range)
num_countries = len(countries)

# Create availability matrix (1 = data exists, 0 = missing)
availability_matrix = np.zeros((num_countries, num_years))
for i, country in enumerate(countries):
    country_years = df_filtered[df_filtered['COUNTRY'] == country]['YEAR'].values
    for j, year in enumerate(years_range):
        if year in country_years:
            availability_matrix[i, j] = 1

# Calculate statistics
total_possible = num_countries * num_years
total_available = availability_matrix.sum()
coverage_pct = (total_available / total_possible) * 100

# Create heatmap
fig, ax = plt.subplots(figsize=(20, 14))

sns.heatmap(availability_matrix,
            xticklabels=list(years_range),
            yticklabels=countries,
            cmap=['#ff6b6b', '#51cf66'],  # Red to Green
            cbar_kws={'label': 'Data Status', 'ticks': [0.25, 0.75]},
            linewidths=0.5,
            linecolor='white',
            ax=ax)

# Customize colorbar
colorbar = ax.collections[0].colorbar
colorbar.set_ticklabels(['Missing', 'Available'])

# Format axes
ax.set_xlabel('Year', fontsize=13, fontweight='bold')
ax.set_ylabel('Country', fontsize=13, fontweight='bold')

# Title
title = f'Data Availability Heatmap: {start_year}-{end_year}\n'
title += f'(Green = Available, Red = Missing | Coverage: {coverage_pct:.1f}% | {num_countries} Countries)'
ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('images/01_data_availability_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/01_data_availability_heatmap.png")

# ============================================================================
# IMAGE 2: Countries per Year Line Plot
# Source: 01_data_cleaning.ipynb
# ============================================================================
print("\n[2/5] Generating Countries per Year Line Plot...")

records_per_year = df.groupby('YEAR').size().sort_index()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(records_per_year.index, records_per_year.values, marker='o', linewidth=2, markersize=4, color='steelblue')
ax.fill_between(records_per_year.index, records_per_year.values, alpha=0.3, color='steelblue')
ax.axhline(y=records_per_year.mean(), color='red', linestyle='--', linewidth=2, label=f'Average: {records_per_year.mean():.1f} countries')
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Countries', fontsize=12, fontweight='bold')
ax.set_title('Data Coverage: Number of Countries with Data per Year', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('images/02_countries_per_year.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/02_countries_per_year.png")

# ============================================================================
# IMAGE 3: Correlation Heatmap
# Source: 02_feature_engineering.ipynb
# ============================================================================
print("\n[3/5] Generating Correlation Heatmap...")

df_clean = pd.read_csv('data/processing/cleaned_data_2011_2020.csv')
features = ['PC_HEALTHXP', 'PC_GDP', 'USD_CAP', 'TOTAL_SPEND']
correlation = df_clean[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=2,
            vmin=-1, vmax=1)

plt.title('Correlation Between Features\n(1 = perfect correlation, 0 = no correlation)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/03_correlation_heatmap.png")

# ============================================================================
# IMAGE 4: Elbow Method & Silhouette Scores
# Source: 03_clustering_analysis.ipynb
# ============================================================================
print("\n[4/5] Generating Elbow Method & Silhouette Analysis...")

# Load engineered features
df_features = pd.read_csv('data/processing/engineered_features.csv')
X = df_features.drop('COUNTRY', axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate metrics for different k values
k_range = range(2, 8)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Elbow plot
ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
ax1.axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Chosen k=3')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
ax1.set_title('Elbow Method: Finding Optimal k', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Silhouette plot
ax2.plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='darkorange')
ax2.axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Chosen k=3')
ax2.axhline(y=silhouette_scores[1], color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'k=3 Score: {silhouette_scores[1]:.3f}')
ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('Silhouette Analysis: Cluster Quality', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('images/04_elbow_silhouette.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/04_elbow_silhouette.png")

# ============================================================================
# IMAGE 5: Cluster Distribution Comparison
# Source: 03_clustering_analysis.ipynb
# ============================================================================
print("\n[5/5] Generating Cluster Distribution Comparison...")

# Load clustering results
df_results = pd.read_csv('data/processing/clustering_results.csv')

# Select key features for visualization
features_to_plot = ['PC_HEALTHXP_avg', 'PC_GDP_avg', 'USD_CAP_avg',
                    'PC_HEALTHXP_growth', 'PC_GDP_growth', 'USD_CAP_growth',
                    'PC_HEALTHXP_volatility', 'PC_GDP_volatility', 'USD_CAP_volatility']

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

# Color palette for clusters
cluster_colors = {0: '#ff6b6b', 1: '#ffd93d', 2: '#6bcf7f'}

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx]

    # Create box plot
    data_by_cluster = [df_results[df_results['Cluster_k3'] == cluster][feature].values
                       for cluster in sorted(df_results['Cluster_k3'].unique())]

    bp = ax.boxplot(data_by_cluster,
                    labels=['Cluster 0\n(Crisis)', 'Cluster 1\n(Stable)', 'Cluster 2\n(High-Value)'],
                    patch_artist=True,
                    widths=0.6)

    # Color the boxes
    for patch, cluster_id in zip(bp['boxes'], sorted(df_results['Cluster_k3'].unique())):
        patch.set_facecolor(cluster_colors[cluster_id])
        patch.set_alpha(0.7)

    ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=9)

plt.suptitle('Feature Distributions Across Market Segments',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/05_cluster_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/05_cluster_distributions.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("IMAGE GENERATION COMPLETE!")
print("="*70)
print("\nGenerated images:")
print("  1. images/01_data_availability_heatmap.png")
print("  2. images/02_countries_per_year.png")
print("  3. images/03_correlation_heatmap.png")
print("  4. images/04_elbow_silhouette.png")
print("  5. images/05_cluster_distributions.png")
print("\nAll images are ready to be included in the README.")
print("="*70)
