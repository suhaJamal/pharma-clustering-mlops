"""
Create Placeholder Images for README
======================================
This creates simple placeholder images that reference the actual visualizations
in the notebooks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_placeholder(filename, title, source_notebook, description):
    """Create a placeholder image with instructions"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 8.5, title, fontsize=20, fontweight='bold',
            ha='center', va='center')

    # Source
    ax.text(5, 7.5, f'Source: {source_notebook}', fontsize=14,
            ha='center', va='center', style='italic', color='#666666')

    # Description
    ax.text(5, 5.5, description, fontsize=12,
            ha='center', va='center', wrap=True)

    # Instructions
    instructions = 'To view this visualization:\n1. Open the source notebook\n2. Run all cells\n3. Scroll to the relevant section'
    ax.text(5, 3, instructions, fontsize=11,
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Border
    rect = mpatches.Rectangle((0.5, 0.5), 9, 9, linewidth=2,
                               edgecolor='steelblue', facecolor='none')
    ax.add_patch(rect)

    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {filename}")

# Create placeholders
print("Creating placeholder images...")
print("="*70)

create_placeholder(
    'images/01_data_availability_heatmap.png',
    'Data Availability Heatmap (2011-2020)',
    '01_data_cleaning.ipynb',
    'Shows data completeness across 36 countries over 10 years.\nJustifies time window selection (88.6% average completeness).\nGreen = data available, Red = missing data.'
)

create_placeholder(
    'images/02_countries_per_year.png',
    'Countries per Year Coverage',
    '01_data_cleaning.ipynb',
    'Line plot showing number of countries with data per year.\nDemonstrates why 2011-2020 is the optimal time window.\nShows peak coverage of 40+ countries in 2010s.'
)

create_placeholder(
    'images/03_correlation_heatmap.png',
    'Feature Correlation Matrix',
    '02_feature_engineering.ipynb',
    'Correlation heatmap of the 4 original features.\nJustifies dropping TOTAL_SPEND (0.703 correlation with USD_CAP).\nGuides feature selection for clustering analysis.'
)

create_placeholder(
    'images/04_elbow_silhouette.png',
    'Elbow Method & Silhouette Analysis',
    '03_clustering_analysis.ipynb',
    'Dual plot justifying k=3 as optimal number of clusters.\nLeft: Elbow curve showing inertia decrease.\nRight: Silhouette scores showing cluster quality.'
)

create_placeholder(
    'images/05_cluster_distributions.png',
    'Cluster Distribution Comparison',
    '03_clustering_analysis.ipynb',
    'Box plots comparing 9 engineered features across 3 clusters.\nHighlights differences in growth rates, spending levels, and volatility.\nProvides statistical evidence for cluster characteristics.'
)

print("="*70)
print("✓ All placeholder images created successfully!")
print("\nNote: These are placeholders. To generate actual visualizations,")
print("run the respective notebooks with the required dependencies installed.")
