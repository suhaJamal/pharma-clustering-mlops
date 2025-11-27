"""
Clustering Analyzer Class
==========================
Author: Suha Islaih (with AI assistance from Claude)
Date: November 2025

Purpose:
--------
A comprehensive class for performing pharmaceutical spending clustering analysis
on OECD country data. Encapsulates data preprocessing, clustering, evaluation,
and visualization in a reusable object-oriented structure.

Usage Example:
--------------
    from clustering_analyzer import ClusteringAnalyzer
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(data_path='../data/processing/engineered_features.csv')
    
    # Perform clustering
    analyzer.fit(k=3)
    
    # Get results
    results = analyzer.get_results()
    
    # Visualize
    analyzer.plot_interactive_clusters()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')


class ClusteringAnalyzer:
    """
    A comprehensive clustering analysis tool for pharmaceutical spending data.
    
    This class handles the entire clustering workflow including:
    - Data loading and preprocessing
    - Feature scaling
    - K-means clustering
    - Cluster evaluation (elbow method, silhouette analysis)
    - PCA dimensionality reduction
    - Interactive visualizations
    
    Attributes:
    -----------
    df : pd.DataFrame
        Original dataset
    X : np.ndarray
        Feature matrix
    X_scaled : np.ndarray
        Standardized feature matrix
    scaler : StandardScaler
        Fitted scaler object
    kmeans : KMeans
        Fitted K-means model
    clusters : np.ndarray
        Cluster assignments
    pca : PCA
        Fitted PCA object
    X_pca : np.ndarray
        PCA-transformed features
    results_df : pd.DataFrame
        DataFrame with cluster assignments and country data
    """
    
    def __init__(self, data_path=None, df=None, random_state=42):
        """
        Initialize the ClusteringAnalyzer.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the engineered features CSV file
        df : pd.DataFrame, optional
            Pre-loaded DataFrame (use instead of data_path)
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state
        self.df = None
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.kmeans = None
        self.clusters = None
        self.pca = None
        self.X_pca = None
        self.results_df = None
        self.cluster_names = {}
        
        # Load data
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            self.load_data(data_path)
        else:
            raise ValueError("Either 'data_path' or 'df' must be provided")
        
        # Country name mapping
        self.country_names = {
            'AUS': 'Australia', 'AUT': 'Austria', 'BEL': 'Belgium', 'CAN': 'Canada',
            'CHE': 'Switzerland', 'CRI': 'Costa Rica', 'CYP': 'Cyprus', 'CZE': 'Czech Republic',
            'DEU': 'Germany', 'DNK': 'Denmark', 'ESP': 'Spain', 'EST': 'Estonia',
            'FIN': 'Finland', 'FRA': 'France', 'GBR': 'United Kingdom', 'GRC': 'Greece',
            'HRV': 'Croatia', 'HUN': 'Hungary', 'IRL': 'Ireland', 'ISL': 'Iceland',
            'ISR': 'Israel', 'ITA': 'Italy', 'JPN': 'Japan', 'KOR': 'South Korea',
            'LTU': 'Lithuania', 'LUX': 'Luxembourg', 'LVA': 'Latvia', 'MEX': 'Mexico',
            'NLD': 'Netherlands', 'NOR': 'Norway', 'NZL': 'New Zealand', 'POL': 'Poland',
            'PRT': 'Portugal', 'ROU': 'Romania', 'SVK': 'Slovakia', 'SVN': 'Slovenia',
            'SWE': 'Sweden', 'TUR': 'Turkey', 'USA': 'United States'
        }
        
    def load_data(self, file_path):
        """
        Load the engineered features dataset.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        """
        self.df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully: {self.df.shape[0]} countries, {self.df.shape[1]} features")
        
    def prepare_features(self, exclude_cols=['COUNTRY']):
        """
        Prepare and scale features for clustering.
        
        Parameters:
        -----------
        exclude_cols : list, default=['COUNTRY']
            Columns to exclude from clustering features
            
        Returns:
        --------
        self : ClusteringAnalyzer
            Returns self for method chaining
        """
        # Extract features (all columns except specified)
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.X = self.df[feature_cols].values
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✓ Features prepared and scaled: {len(feature_cols)} features")
        return self
        
    def find_optimal_k(self, k_range=range(2, 11), method='both'):
        """
        Find optimal number of clusters using elbow method and/or silhouette analysis.
        
        Parameters:
        -----------
        k_range : range or list, default=range(2, 11)
            Range of k values to test
        method : str, default='both'
            Method to use: 'elbow', 'silhouette', or 'both'
            
        Returns:
        --------
        dict : Dictionary with inertias and silhouette scores
        """
        if self.X_scaled is None:
            self.prepare_features()
            
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            clusters = kmeans.fit_predict(self.X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, clusters))
        
        # Plotting
        if method in ['elbow', 'both']:
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
            plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(k_range)
            
        if method in ['silhouette', 'both']:
            if method == 'both':
                plt.subplot(1, 2, 2)
            else:
                plt.figure(figsize=(7, 5))
                
            plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(k_range)
            
        plt.tight_layout()
        plt.show()
        
        # Print recommendations - top 2 k values
        k_list = list(k_range)
        scores_array = np.array(silhouette_scores)
        
        # Get indices of top 2 scores
        top_2_indices = np.argsort(scores_array)[-2:][::-1]  # Descending order
        
        print(f"\n📊 Optimal k Analysis:")
        print(f"   Best k by Silhouette Score:")
        print(f"      1st: k={k_list[top_2_indices[0]]} (score: {scores_array[top_2_indices[0]]:.3f})")
        print(f"      2nd: k={k_list[top_2_indices[1]]} (score: {scores_array[top_2_indices[1]]:.3f})")
        
        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
        
    def fit(self, k=3, cluster_names=None):
        """
        Fit K-means clustering model.
        
        Parameters:
        -----------
        k : int, default=3
            Number of clusters
        cluster_names : dict, optional
            Dictionary mapping cluster numbers to names
            Example: {0: 'Crisis Markets', 1: 'Stable Markets', 2: 'High-Value Markets'}
            
        Returns:
        --------
        self : ClusteringAnalyzer
            Returns self for method chaining
        """
        if self.X_scaled is None:
            self.prepare_features()
            
        # Fit K-means
        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.X_scaled)
        
        # Set cluster names
        if cluster_names:
            self.cluster_names = cluster_names
        else:
            self.cluster_names = {i: f"Cluster {i}" for i in range(k)}
        
        # Calculate silhouette score
        silhouette = silhouette_score(self.X_scaled, self.clusters)
        
        print(f"✓ K-means clustering completed (k={k})")
        print(f"   Silhouette Score: {silhouette:.3f}")
        print(f"   Inertia: {self.kmeans.inertia_:.2f}")
        
        return self
        
    def apply_pca(self, n_components=2):
        """
        Apply PCA for dimensionality reduction and visualization.
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of principal components
            
        Returns:
        --------
        self : ClusteringAnalyzer
            Returns self for method chaining
        """
        if self.X_scaled is None:
            raise ValueError("Features must be prepared first. Call prepare_features() or fit()")
            
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        variance_explained = self.pca.explained_variance_ratio_ * 100
        print(f"✓ PCA applied ({n_components} components)")
        print(f"   Explained variance: {' + '.join([f'{v:.1f}%' for v in variance_explained])} = {sum(variance_explained):.1f}%")
        
        return self
        
    def get_results(self):
        """
        Get clustering results as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame : DataFrame with countries, features, and cluster assignments
        """
        if self.clusters is None:
            raise ValueError("Model must be fitted first. Call fit()")
            
        # Create results dataframe
        self.results_df = self.df.copy()
        self.results_df[f'Cluster_k{len(np.unique(self.clusters))}'] = self.clusters
        self.results_df['Cluster_Name'] = [self.cluster_names[c] for c in self.clusters]
        
        return self.results_df
        
    def get_cluster_summary(self):
        """
        Get summary statistics for each cluster.
        
        Returns:
        --------
        pd.DataFrame : Summary statistics by cluster
        """
        if self.results_df is None:
            self.get_results()
            
        cluster_col = [col for col in self.results_df.columns if col.startswith('Cluster_k')][0]
        
        # Get numeric columns only
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.startswith('Cluster')]
        
        summary = self.results_df.groupby(cluster_col)[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        
        return summary
        
    def plot_clusters_2d(self, figsize=(12, 8)):
        """
        Plot clusters in 2D using PCA (static matplotlib plot).
        
        Parameters:
        -----------
        figsize : tuple, default=(12, 8)
            Figure size (width, height)
        """
        if self.X_pca is None:
            self.apply_pca(n_components=2)
            
        if self.results_df is None:
            self.get_results()
            
        # Define colors
        colors = {
            0: '#F24495',  # Pink
            1: '#432E8C',  # Purple
            2: '#66C4D9',  # Blue
            3: '#F7931E',  # Orange
            4: '#00A651',  # Green
        }
        
        plt.figure(figsize=figsize)
        
        for cluster in sorted(np.unique(self.clusters)):
            mask = self.clusters == cluster
            plt.scatter(
                self.X_pca[mask, 0],
                self.X_pca[mask, 1],
                c=colors.get(cluster, 'gray'),
                label=self.cluster_names[cluster],
                s=150,
                alpha=0.7,
                edgecolors='white',
                linewidth=2
            )
            
            # Add country labels
            for i, (x, y) in enumerate(zip(self.X_pca[mask, 0], self.X_pca[mask, 1])):
                country = self.df.iloc[np.where(mask)[0][i]]['COUNTRY']
                plt.annotate(
                    country,
                    (x, y),
                    fontsize=8,
                    ha='center',
                    va='center'
                )
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
        plt.title('Country Clusters: Pharmaceutical Spending Segmentation', fontsize=14, fontweight='bold')
        plt.legend(title='Cluster', loc='best', frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_interactive_clusters(self, width=900, height=600):
        """
        Create an interactive Plotly visualization of clusters.
        
        Parameters:
        -----------
        width : int, default=900
            Plot width in pixels
        height : int, default=600
            Plot height in pixels
        """
        if self.X_pca is None:
            self.apply_pca(n_components=2)
            
        if self.results_df is None:
            self.get_results()
            
        # Create plot dataframe
        countries_list = self.results_df['COUNTRY'].values
        cluster_col = [col for col in self.results_df.columns if col.startswith('Cluster_k')][0]
        
        plot_df = pd.DataFrame({
            'PC1': self.X_pca[:, 0],
            'PC2': self.X_pca[:, 1],
            'Country_Code': countries_list,
            'Cluster': self.results_df[cluster_col].values,
            'Cluster_Name': [self.cluster_names[c] for c in self.results_df[cluster_col].values]
        })
        
        # Add country names
        plot_df['Country_Name'] = plot_df['Country_Code'].map(self.country_names)
        
        # Add metrics for hover (if available)
        if 'USD_CAP_avg' in self.results_df.columns:
            plot_df['Spending_USD'] = plot_df['Country_Code'].map(
                self.results_df.set_index('COUNTRY')['USD_CAP_avg']
            ).round(0)
            plot_df['Pharma_Pct'] = plot_df['Country_Code'].map(
                self.results_df.set_index('COUNTRY')['PC_HEALTHXP_avg']
            ).round(1)
            plot_df['Growth_Pct'] = plot_df['Country_Code'].map(
                self.results_df.set_index('COUNTRY')['USD_CAP_growth']
            ).round(2)
        
        # Custom colors
        colors = {
            0: '#F24495',  # Pink
            1: '#432E8C',  # Purple
            2: '#66C4D9',  # Blue
            3: '#F7931E',  # Orange
            4: '#00A651',  # Green
        }
        
        # Create interactive plot
        fig = go.Figure()
        
        for cluster in sorted(plot_df['Cluster'].unique()):
            cluster_data = plot_df[plot_df['Cluster'] == cluster]
            
            # Build hover template
            if 'Spending_USD' in plot_df.columns:
                hover_template = ('<b>%{text}</b><br>' +
                                'Code: %{customdata[0]}<br>' +
                                'Spending: $%{customdata[1]:,.0f}/capita<br>' +
                                'Pharma Share: %{customdata[2]:.1f}%<br>' +
                                'Growth: %{customdata[3]:.2f}%/year<br>' +
                                '<extra></extra>')
                customdata = cluster_data[['Country_Code', 'Spending_USD', 'Pharma_Pct', 'Growth_Pct']]
            else:
                hover_template = '<b>%{text}</b><br>Code: %{customdata[0]}<br><extra></extra>'
                customdata = cluster_data[['Country_Code']]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                mode='markers',
                name=f'{self.cluster_names[cluster]}',
                marker=dict(
                    size=15,
                    color=colors.get(cluster, 'gray'),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=cluster_data['Country_Name'],
                customdata=customdata,
                hovertemplate=hover_template
            ))
        
        # Update layout
        k = len(np.unique(self.clusters))
        fig.update_layout(
            title={
                'text': f'Country Clusters: Pharmaceutical Spending Segmentation (k={k})',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial, sans-serif', 'color': '#333'}
            },
            xaxis_title=f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}% variance)',
            yaxis_title=f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}% variance)',
            width=width,
            height=height,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, family='Arial, sans-serif'),
            legend=dict(
                title=dict(text='Market Segment', font=dict(size=14, family='Arial, sans-serif')),
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#ccc',
                borderwidth=1
            ),
            xaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#ccc'
            ),
            yaxis=dict(
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#ccc'
            )
        )
        
        fig.show()
        print(f"✓ Interactive visualization created")
        
    def plot_cluster_distributions(self, feature_cols=None, figsize=(15, 10)):
        """
        Plot feature distributions by cluster using box plots.
        
        Parameters:
        -----------
        feature_cols : list, optional
            List of feature columns to plot. If None, uses all numeric columns.
        figsize : tuple, default=(15, 10)
            Figure size (width, height)
        """
        if self.results_df is None:
            self.get_results()
            
        cluster_col = [col for col in self.results_df.columns if col.startswith('Cluster_k')][0]
        
        if feature_cols is None:
            feature_cols = [col for col in self.df.columns if col != 'COUNTRY']
        
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(feature_cols):
            ax = axes[idx]
            self.results_df.boxplot(
                column=feature,
                by=cluster_col,
                ax=ax,
                patch_artist=True
            )
            ax.set_title(feature)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Value')
            plt.sca(ax)
            plt.xticks(rotation=0)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions by Cluster', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        
    def export_results(self, output_path):
        """
        Export clustering results to CSV.
        
        Parameters:
        -----------
        output_path : str
            Path to save the results CSV file
        """
        if self.results_df is None:
            self.get_results()
            
        self.results_df.to_csv(output_path, index=False)
        print(f"✓ Results exported to: {output_path}")
        
    def get_cluster_countries(self, cluster_id=None):
        """
        Get list of countries in a specific cluster or all clusters.
        
        Parameters:
        -----------
        cluster_id : int, optional
            Cluster ID to filter by. If None, returns all clusters.
            
        Returns:
        --------
        dict or list : Dictionary of {cluster_id: [countries]} or list of countries
        """
        if self.results_df is None:
            self.get_results()
            
        cluster_col = [col for col in self.results_df.columns if col.startswith('Cluster_k')][0]
        
        if cluster_id is not None:
            return self.results_df[self.results_df[cluster_col] == cluster_id]['COUNTRY'].tolist()
        else:
            return {
                cluster: self.results_df[self.results_df[cluster_col] == cluster]['COUNTRY'].tolist()
                for cluster in sorted(self.results_df[cluster_col].unique())
            }

    def get_comparative_metrics(self, display=True):
            """
            Get comprehensive comparative metrics across clusters.
            
            This method calculates key metrics for each cluster including:
            - Average and standard deviation of spending metrics
            - Growth rate analysis
            - 5-year projected spending
            
            Parameters:
            -----------
            display : bool, default=True
                If True, prints formatted output. If False, only returns data.
                
            Returns:
            --------
            dict : Dictionary containing:
                - 'comparison_metrics': DataFrame with mean/std for key metrics
                - 'growth_analysis': Dictionary with growth projections by cluster
            """
            if self.results_df is None:
                self.get_results()
                
            cluster_col = [col for col in self.results_df.columns if col.startswith('Cluster_k')][0]
            
            # Calculate comparative metrics
            comparison_metrics = self.results_df.groupby(cluster_col).agg({
                'USD_CAP_avg': ['mean', 'std'],
                'USD_CAP_growth': ['mean', 'std'],
                'PC_HEALTHXP_avg': ['mean', 'std'],
                'PC_GDP_avg': ['mean', 'std']
            }).round(2)
            
            comparison_metrics.columns = ['_'.join(col) for col in comparison_metrics.columns]
            
            # Calculate growth projections
            growth_analysis = {}
            for cluster_id in sorted(self.results_df[cluster_col].unique()):
                cluster_data = self.results_df[self.results_df[cluster_col] == cluster_id]
                avg_growth = cluster_data['USD_CAP_growth'].mean()
                total_spending = cluster_data['USD_CAP_avg'].mean()
                
                growth_analysis[cluster_id] = {
                    'cluster_name': self.cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                    'avg_per_capita_spending': total_spending,
                    'avg_annual_growth_rate': avg_growth,
                    'projected_1yr': total_spending * (1 + avg_growth/100)**1,
                    'projected_3yr': total_spending * (1 + avg_growth/100)**3,
                    'projected_5yr': total_spending * (1 + avg_growth/100)**5,
                    'n_countries': len(cluster_data)
                }
            
            # Display if requested
            if display:
                print("\n" + "="*80)
                print("KEY COMPARATIVE METRICS ACROSS CLUSTERS")
                print("="*80)
                print("\nMean and Standard Deviation by Cluster:")
                print("-"*80)
                print(comparison_metrics.to_string())
                
                print("\n\n" + "="*80)
                print("GROWTH ANALYSIS & PROJECTIONS")
                print("="*80)
                
                for cluster_id in sorted(growth_analysis.keys()):
                    info = growth_analysis[cluster_id]
                    print(f"\n{info['cluster_name']} (Cluster {cluster_id}):")
                    print(f"  Number of countries: {info['n_countries']}")
                    print(f"  Average per capita spending: ${info['avg_per_capita_spending']:.2f}")
                    print(f"  Average annual growth rate: {info['avg_annual_growth_rate']:.2f}%")
                    print(f"  Projected spending:")
                    print(f"    • 1-year:  ${info['projected_1yr']:.2f}")
                    print(f"    • 3-year:  ${info['projected_3yr']:.2f}")
                    print(f"    • 5-year:  ${info['projected_5yr']:.2f}")
                    
                    # Calculate percentage change
                    pct_change_5yr = ((info['projected_5yr'] - info['avg_per_capita_spending']) / 
                                    info['avg_per_capita_spending'] * 100)
                    print(f"    • 5-year % change: {pct_change_5yr:+.1f}%")
                
                print("\n" + "="*80)
            
            return {
                'comparison_metrics': comparison_metrics,
                'growth_analysis': growth_analysis
            }