# Pharmaceutical Drug Spending Clustering Analysis

## 🎯 Project Overview

This data science project analyzes pharmaceutical drug spending patterns across OECD countries to identify distinct market segments and inform strategic market entry decisions. Using clustering analysis on a decade of healthcare expenditure data (2011-2020), we segment countries into three distinct groups based on their pharmaceutical spending characteristics, growth trajectories, and market stability.

**Team DS-4 | Cohort 7**
- Ahil Khuwaja
- [Fabiana Camargo Franco Barril](https://www.linkedin.com/in/fabiana-barril-mba-233976125/)
- [Mohammad Faisal](https://www.linkedin.com/in/mo-r-faisal/)
- [Saranya Manoharan](https://www.linkedin.com/in/smano1/)
- [Suha Islaih](https://www.linkedin.com/in/suha-islaih/)

**Repository:** [https://github.com/saranya-mano/DS04-Team-Project](https://github.com/saranya-mano/DS04-Team-Project)

---

## 📊 Business Question

**Primary Question:**
> What are the clusters among countries in terms of pharmaceutical spending (as a percentage of health spending, percentage of GDP, and per capita spending), and how do these patterns relate to total spending across different years? How can these insights inform market entry strategies?

**Value Proposition:**
This analysis provides pharmaceutical companies, healthcare policymakers, and investment firms with:
- Data-driven market segmentation for strategic planning
- Risk assessment frameworks for market entry decisions
- Growth opportunity identification in pharmaceutical markets
- Evidence-based resource allocation strategies

---

## 🎯 Key Findings

### Three Distinct Market Segments Identified

Our clustering analysis revealed three distinct pharmaceutical market segments:

#### 🔴 Cluster 0: Crisis/Declining Markets (5 countries)
**Countries:** Costa Rica, Croatia, Hungary, Ireland, Slovakia

**Characteristics:**
- **Economic decline:** Severe negative GDP growth (-4% to -6% annually)
- **High pharma share:** 13-30% of health budget despite economic contraction
- **Moderate spending:** ~$450 per capita
- **High volatility:** Unstable spending patterns and economic conditions

**Strategic Recommendation:** ❌ **NOT recommended for new market entry**
- High market risk due to economic instability
- Declining purchasing power and unpredictable payment patterns
- If already present: Focus on essential medicines, flexible pricing, minimal inventory

#### 🟡 Cluster 1: Stable Moderate Markets (23 countries)
**Countries:** Australia, Austria, Belgium, Canada, Switzerland, Cyprus, Czech Republic, Denmark, Spain, Estonia, Finland, France, Iceland, Israel, Italy, Luxembourg, Mexico, Netherlands, Norway, Poland, Portugal, Slovenia, Sweden

**Characteristics:**
- **Steady growth:** 2.45% average annual growth
- **Balanced budgets:** Moderate pharma share (12-22%) in stable economies
- **Mid-range spending:** ~$495 per capita
- **Low volatility:** Predictable, reliable spending patterns

**Strategic Recommendation:** ✅ **IDEAL for market expansion**
- Stable, predictable growth trajectory
- Long-term contracts with price escalation possible
- Target for branded generics and biosimilars
- **5-year growth projection:** +12.9% increase in per capita spending

#### 🟢 Cluster 2: High-Value Pharma Markets (8 countries)
**Countries:** Germany, Greece, Japan, South Korea, Lithuania, Latvia, Romania, USA

**Characteristics:**
- **Accelerating growth:** 4.0% average annual growth
- **High spending:** ~$659 per capita (highest segment)
- **Strategic investment:** 18-32% pharma share in expanding budgets
- **Premium potential:** Moderate volatility reflects innovation adoption

**Strategic Recommendation:** ⭐ **PRIORITY for innovative products**
- Strongest growth trajectory in pharmaceutical spending
- Premium pricing achievable
- Ideal for specialty drugs and innovative therapies
- **5-year growth projection:** +21.7% increase in per capita spending

---

## 🛠️ Methods & Technologies

### Data Science Techniques
- **K-means Clustering:** Segmentation of countries into market groups
- **Principal Component Analysis (PCA):** Dimensionality reduction for visualization
- **Elbow Method & Silhouette Analysis:** Optimal cluster determination (k=3)
- **Feature Engineering:** Growth rates, volatility metrics, temporal aggregations
- **Correlation Analysis:** Feature selection and redundancy elimination

### Technology Stack
```
Python 3.8+
├── Data Manipulation: pandas 1.5.3, numpy 1.26.4
├── Machine Learning: scikit-learn 1.5.0
├── Visualization: matplotlib 3.6.0, seaborn 0.13.2, plotly 6.20
└── Statistical Analysis: scipy 1.12.0
```

### Analytical Workflow
1. **Data Cleaning:** Missing value analysis, duplicate detection, outlier identification
2. **Time Window Selection:** 2011-2020 chosen for optimal country coverage (36 countries with complete data)
3. **Feature Engineering:** Created 9 engineered features from 3 base metrics
4. **Feature Scaling:** StandardScaler normalization for clustering
5. **Clustering Analysis:** K-means with k=3 (Silhouette Score: 0.289)
6. **Validation:** Business interpretation and strategic profiling

---

## 📁 Repository Structure

```
DS04-Team-Project/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/
│   │   └── data.csv                   # Original OECD pharmaceutical spending data (1970-2022)
│   └── processing/
│       ├── eda_data.csv              # Preprocessed data for initial exploration
│       ├── cleaned_data_2011_2020.csv # Filtered dataset (36 countries, 2011-2020)
│       ├── engineered_features.csv    # Clustering-ready features (9 features per country)
│       └── clustering_results.csv     # Final cluster assignments with labels
│
├── notebooks/
│   ├── 00_quick_eda.ipynb            # Initial exploratory data analysis (full dataset)
│   ├── 01_data_cleaning.ipynb        # Comprehensive data cleaning & time window selection
│   ├── 02_feature_engineering.ipynb  # Feature creation & transformation
│   ├── 03_clustering_analysis.ipynb  # K-means clustering & market segmentation
│   └── clustering_analyzer.py        # Custom ClusteringAnalyzer class
│
├── images/                            # Visualization images for README
│   └── README.md                      # Instructions for generating images
│
├── scripts/                           # Utility scripts
│   ├── generate_images.py            # Script to generate visualization PNGs
│   ├── create_placeholder_images.py  # Alternative placeholder generator
│   └── GENERATE_IMAGES_INSTRUCTIONS.md # Detailed image generation guide
│
└── experiments/                       # Experimental notebooks (scaling tests, alternative approaches)
    ├── clustering.ipynb
    └── scaling.ipynb
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/saranya-mano/DS04-Team-Project.git
cd DS04-Team-Project
```

2. **Create a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Run notebooks in sequence (recommended for understanding the workflow)**

Execute the notebooks in order:
1. `00_quick_eda.ipynb` - Initial exploration (~1 minute)
2. `01_data_cleaning.ipynb` - Data preparation (~2 minutes)
3. `02_feature_engineering.ipynb` - Feature creation (~1 minute)
4. `03_clustering_analysis.ipynb` - Clustering & insights (~3 minutes)

**Option 2: Quick start with final analysis**

Jump directly to `03_clustering_analysis.ipynb` - it uses the pre-processed data from `data/processing/`

**Option 3: Use the ClusteringAnalyzer class**

```python
from notebooks.clustering_analyzer import ClusteringAnalyzer

# Initialize analyzer
analyzer = ClusteringAnalyzer(data_path='data/processing/engineered_features.csv')

# Perform clustering
analyzer.fit(k=3, cluster_names={
    0: 'Crisis/Declining Markets',
    1: 'Stable Moderate Markets',
    2: 'High-Value Pharma Markets'
})

# Get results
results = analyzer.get_results()

# Visualize
analyzer.plot_interactive_clusters()
```

---

## 📈 Key Visualizations

The analysis includes several critical visualizations that support our findings:

### 1. Data Availability Heatmap (2011-2020)
**Source:** [01_data_cleaning.ipynb](notebooks/01_data_cleaning.ipynb)

![Data Availability Heatmap](images/01_data_availability_heatmap.png)

This heatmap shows data completeness across 36 countries over the 10-year analysis period. Green cells indicate available data, while red cells show missing data. The visualization demonstrates 88.6% average completeness and justifies our selection of 2011-2020 as the optimal time window for analysis.

### 2. Countries per Year Coverage
**Source:** [01_data_cleaning.ipynb](notebooks/01_data_cleaning.ipynb)

![Countries per Year](images/02_countries_per_year.png)

This line plot illustrates the number of countries with available data for each year from 1970 to 2022. The chart clearly shows peak coverage of 40+ countries during the 2010-2020 period, supporting our time window selection.

### 3. Feature Correlation Matrix
**Source:** [02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb)

![Correlation Heatmap](images/03_correlation_heatmap.png)

The correlation heatmap reveals relationships between the four original features. Most notably, it shows a 0.703 correlation between TOTAL_SPEND and USD_CAP, which justified our decision to drop TOTAL_SPEND from the analysis to avoid redundancy.

### 4. Elbow Method & Silhouette Analysis
**Source:** [03_clustering_analysis.ipynb](notebooks/03_clustering_analysis.ipynb)

![Elbow and Silhouette Analysis](images/04_elbow_silhouette.png)

These dual plots justify k=3 as the optimal number of clusters. The elbow plot (left) shows the point of diminishing returns in inertia reduction, while the silhouette analysis (right) confirms that k=3 provides good cluster quality with a score of 0.289.

### 5. Cluster Distribution Comparison
**Source:** [03_clustering_analysis.ipynb](notebooks/03_clustering_analysis.ipynb)

![Cluster Distributions](images/05_cluster_distributions.png)

Box plots comparing the nine engineered features across the three market segments. These visualizations provide statistical evidence for the distinct characteristics of each cluster, showing clear differences in growth rates, spending levels, and market volatility.

---

### Generating Visualizations

To generate high-resolution versions of these images:

1. **Ensure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the image generation script:**
   ```bash
   python scripts/generate_images.py
   ```

3. **Or view in notebooks:**
   Open the source notebooks in Jupyter and run all cells to see interactive versions of these visualizations.

For detailed instructions, see [scripts/GENERATE_IMAGES_INSTRUCTIONS.md](scripts/GENERATE_IMAGES_INSTRUCTIONS.md).

---

## 📊 Dataset Description

**Source:** OECD Pharmaceutical Drug Spending Dataset
**Link:** [https://datahub.io/core/pharmaceutical-drug-spending](https://datahub.io/core/pharmaceutical-drug-spending)

**Original Dataset:**
- **Records:** 1,341 country-year observations
- **Countries:** 44 OECD and partner countries
- **Time Period:** 1970-2022 (53 years)
- **Features:** 6 variables

**Analysis Dataset (cleaned):**
- **Records:** 360 country-year observations (36 countries × 10 years)
- **Countries:** 36 countries with complete data
- **Time Period:** 2011-2020 (10 years)
- **Features:** 9 engineered features

### Variables

**Original Features:**
- `PC_HEALTHXP` - Pharmaceutical spending as % of total health expenditure
- `PC_GDP` - Health expenditure as % of GDP
- `USD_CAP` - Health spending per capita (USD, PPP-adjusted)
- `TOTAL_SPEND` - Total health spending (million USD)
- `COUNTRY` - ISO 3-letter country code
- `YEAR` - Year of observation

**Engineered Features (used for clustering):**
- `PC_HEALTHXP_growth` - Annual compound growth rate (2011-2020)
- `PC_GDP_growth` - Annual compound growth rate (2011-2020)
- `USD_CAP_growth` - Annual compound growth rate (2011-2020)
- `PC_HEALTHXP_avg` - Mean spending level (2011-2020)
- `PC_GDP_avg` - Mean GDP percentage (2011-2020)
- `USD_CAP_avg` - Mean per capita spending (2011-2020)
- `PC_HEALTHXP_volatility` - Standard deviation (market stability)
- `PC_GDP_volatility` - Standard deviation (market stability)
- `USD_CAP_volatility` - Standard deviation (market stability)

### Data Quality & Limitations

**Strengths:**
- ✅ Authoritative source (OECD)
- ✅ Standardized methodology across countries
- ✅ PPP-adjusted for fair cross-country comparison
- ✅ No missing values in final dataset (2011-2020)
- ✅ Complete time series for all 36 countries

**Limitations:**
- ⚠️ Incomplete historical data (pre-2010 has gaps)
- ⚠️ Recent years (2021-2022) mostly unavailable due to reporting delays
- ⚠️ 8 countries excluded due to incomplete data (Bulgaria, UK, Malta, Brazil, Colombia, Chile, New Zealand, Turkey)
- ⚠️ Does not capture sub-national variations
- ⚠️ Currency fluctuations may affect time comparisons despite PPP adjustment

---

## 🔄 Changes from Original Project Plan

Throughout the project, we made several strategic adjustments to enhance the analysis quality and align with data realities:

### 1. Time Window Refinement
**Original Plan:** Analyze 2012-2021
**Final Decision:** Analyze 2011-2020

**Rationale:**
- 2021-2022 data largely unavailable (reporting delays)
- 2011-2020 provides 88.6% data completeness vs. lower coverage in earlier periods
- Avoids COVID-19 anomalies (2020 is last pre-pandemic complete year)
- Maximizes country coverage (36 countries with complete data)

### 2. Feature Selection Strategy
**Original Plan:** Use all 4 original features (PC_HEALTHXP, PC_GDP, USD_CAP, TOTAL_SPEND)
**Final Decision:** Dropped TOTAL_SPEND, used 3 features

**Rationale:**
- Correlation analysis showed TOTAL_SPEND and USD_CAP were redundant (r=0.703)
- TOTAL_SPEND biased by country size (USA dominates due to population)
- USD_CAP is population-adjusted and directly addresses research question
- Focus on per capita metrics for fair comparison

### 3. Advanced Feature Engineering
**Original Plan:** Use raw spending metrics
**Final Decision:** Created 9 engineered features (growth rates, averages, volatility)

**Rationale:**
- Raw yearly data too granular for country-level segmentation
- Growth rates capture market trajectory (critical for market entry decisions)
- Volatility metrics assess market risk and stability
- Decade averages reduce noise while preserving signal

### 4. Cluster Interpretation Framework
**Original Plan:** Focus primarily on spending levels
**Final Decision:** Multi-dimensional segmentation (growth + spending + volatility)

**Rationale:**
- Spending level alone insufficient for strategic decisions
- Growth trajectory predicts future opportunity
- Volatility assesses market risk
- Combined perspective provides actionable business insights

### 5. COVID-19 Analysis Scope
**Original Plan:** Analyze COVID-19 impact (2020-2021)
**Final Decision:** Limited COVID analysis (only 2020 data available)

**Rationale:**
- 2021 data unavailable for most countries
- 2020 shows early pandemic impact but incomplete picture
- Focused on pre-pandemic stable patterns for generalizable insights
- Future work could revisit once 2021-2023 data becomes available

### 6. Country Exclusions
**Original Plan:** Attempt to include all countries with imputation
**Final Decision:** Strict inclusion criteria (100% completeness for 2011-2020)

**Rationale:**
- Imputation introduces artificial patterns in clustering
- 36 countries with complete data sufficient for robust analysis
- Maintains data integrity and reproducibility
- UK, Bulgaria considered but excluded to avoid bias

---

## 🎯 Target Audience & Stakeholders

This analysis delivers value to multiple stakeholder groups:

### Primary Stakeholders
1. **Pharmaceutical Companies**
   - Market entry and expansion strategy planning
   - Resource allocation optimization
   - Risk assessment for international operations
   - Portfolio positioning by market segment

2. **Healthcare Investment Firms**
   - Market attractiveness evaluation
   - Risk-adjusted return forecasting
   - Portfolio diversification guidance
   - Due diligence for healthcare acquisitions

3. **Healthcare Policy Makers**
   - Cross-country spending benchmarking
   - Budget allocation insights
   - Identification of peer countries for policy learning
   - Understanding of pharmaceutical market dynamics

### Secondary Stakeholders
4. **Health Economics Researchers**
   - Data-driven segmentation methodology
   - Reproducible analysis framework
   - Publicly available dataset and code

5. **Healthcare Consultancies**
   - Evidence base for client recommendations
   - Market assessment frameworks
   - Strategic positioning tools

---

## 🎥 Team Member Reflections

Each team member has recorded a 3-5 minute video reflecting on the project experience, challenges, and learnings:

- **Ahil Khuwaja:** [Video Link](https://drive.google.com/file/d/1OXS1T6Y8MI9Zf-v2qu1ttTfS3AjY3D__/view)
- **Fabiana Camargo Franco Barril:** [Video Link](https://drive.google.com/file/d/1rExWHvyPyOhDeocrkubGVOz5YggxRd8R/view?usp=sharing)
- **Mohammad Faisal:** [Video Link](https://drive.google.com/drive/folders/1JZq0E0L0oanboX3-LL36e_jPD90b7fGs?usp=sharing)
- **Saranya Manoharan:** [Video Link](https://drive.google.com/file/d/1L5Pzrn7YIiE7ZrHz0NvPn7SOxwFlFse9/view?usp=sharing)
- **Suha Islaih:** [Video Link](https://drive.google.com/file/d/1uulPX_Fk02HpPt1l_fz74qq5Yta6iFZW/view?usp=sharing)

*Videos will be updated with links before final submission.*

---

## 🤝 Team Collaboration

### Collaborative Approach

This project was truly a team effort, with all five members actively contributing throughout the entire analysis pipeline. We embraced a collaborative workflow where team members worked together on different aspects of the project, regularly sharing knowledge, reviewing each other's work, and iteratively refining our analysis based on collective feedback.

### Git & GitHub Workflow
Our team followed data science best practices for version control and collaboration:
- ✅ Each member created pull requests for their contributions
- ✅ Peer code reviews performed before merging
- ✅ Feature branches used for development work
- ✅ Regular commits with clear, descriptive messages
- ✅ No direct commits to main branch

### Communication & Coordination
- **Weekly team meetings:** Discussed progress, addressed blockers, and aligned on next steps
- **Asynchronous communication:** Quick questions and updates via team chat
- **Shared project board:** Tracked tasks and maintained visibility across the team
- **Collaborative problem-solving:** Pair programming sessions for complex challenges
- **Knowledge sharing:** Regular code walkthroughs and technique demonstrations

### Division of Work

Rather than strict role assignments, our team adopted a flexible, collaborative approach:

- **All members participated in** data exploration, analysis planning, and strategic discussions
- **Shared responsibilities** across data cleaning, feature engineering, clustering implementation, and visualization
- **Cross-functional collaboration** ensured every team member understood the full pipeline
- **Collective code review** improved code quality and knowledge transfer
- **Joint documentation** with contributions from all members to ensure comprehensive coverage

This collaborative model allowed us to leverage each member's strengths while building collective expertise across all aspects of the project. Every team member engaged with every major component, from data preparation through final analysis and interpretation.

---

## 🚧 Risks & Uncertainties

### Data-Related Risks
1. **Incomplete Recent Data**
   - **Risk:** 2021-2022 data mostly unavailable
   - **Mitigation:** Focused on 2011-2020 with complete coverage
   - **Impact:** Analysis reflects pre-pandemic patterns

2. **Country Exclusions**
   - **Risk:** 8 countries excluded due to missing data
   - **Mitigation:** 36 countries still provide robust sample
   - **Impact:** Some markets not represented (e.g., UK, Brazil)

3. **Currency Fluctuations**
   - **Risk:** Exchange rate volatility despite PPP adjustment
   - **Mitigation:** Used PPP-adjusted USD for standardization
   - **Impact:** Minimal - OECD methodology accounts for this

### Methodological Uncertainties
1. **Optimal Cluster Number**
   - **Uncertainty:** k=3 vs k=4 trade-off
   - **Approach:** Used multiple validation metrics (elbow, silhouette)
   - **Resolution:** k=3 balances statistical validity and interpretability

2. **Feature Selection**
   - **Uncertainty:** Which features best represent market characteristics
   - **Approach:** Correlation analysis and domain expertise
   - **Resolution:** 9 engineered features capture growth, level, and stability

3. **Temporal Aggregation**
   - **Uncertainty:** How to summarize 10 years of data per country
   - **Approach:** Growth rates + averages + volatility
   - **Resolution:** Captures both static and dynamic characteristics

### Strategic Uncertainties
1. **Future Market Dynamics**
   - **Uncertainty:** Post-COVID pharmaceutical market shifts
   - **Impact:** Projections based on 2011-2020 may need adjustment
   - **Recommendation:** Revisit analysis when 2021-2023 data available

2. **Policy Changes**
   - **Uncertainty:** Healthcare policy reforms could alter spending patterns
   - **Impact:** Cluster membership could shift over time
   - **Recommendation:** Periodic re-clustering (every 2-3 years)

---

## 🔮 Future Work & Extensions

### Immediate Extensions
1. **COVID-19 Impact Analysis**
   - Incorporate 2021-2023 data when available
   - Analyze pandemic-driven market shifts
   - Assess whether clusters remain stable post-COVID

2. **Hierarchical Clustering**
   - Compare with k-means results
   - Explore dendrogram for sub-cluster insights
   - Assess stability of country groupings

3. **Additional Features**
   - Demographic variables (aging population, disease burden)
   - Regulatory environment indicators
   - Healthcare infrastructure metrics

### Advanced Analytics
4. **Predictive Modeling**
   - Forecast cluster membership for new countries
   - Predict cluster transitions over time
   - Develop early warning system for market downgrades

5. **Time Series Analysis**
   - Dynamic clustering (how clusters evolve year-by-year)
   - Trend analysis within clusters
   - Seasonality and cyclical pattern detection

6. **Sub-Segmentation**
   - Within-cluster analysis (e.g., split Cluster 1 into sub-groups)
   - Therapeutic area-specific spending patterns
   - Generic vs. branded pharmaceutical spending

### Business Applications
7. **Interactive Dashboard**
   - Real-time cluster visualization
   - Country comparison tool
   - Market opportunity scoring

8. **Strategic Toolkit**
   - Market entry decision framework
   - Risk scoring calculator
   - Resource allocation optimizer

---

## 📚 References & Data Sources

### Primary Data Source
- **OECD Pharmaceutical Drug Spending Dataset**
  - URL: [https://datahub.io/core/pharmaceutical-drug-spending](https://datahub.io/core/pharmaceutical-drug-spending)
  - Accessed: October-November 2024
  - License: Open Data Commons Public Domain Dedication and License (PDDL)

### Technical Documentation
- **Scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **K-means Clustering:** [https://scikit-learn.org/stable/modules/clustering.html#k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

### Methodological References
- **OECD Health Statistics Methodology:** [https://www.oecd.org/health/health-data.htm](https://www.oecd.org/health/health-data.htm)
- **Cluster Validation Metrics:** Rousseeuw, P.J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

---

## 📝 License & Citation

### License
This project is released under the **MIT License** for code and analysis, with data sourced from OECD under **PDDL** (Public Domain Dedication and License).

### How to Cite This Work
If you use this analysis or methodology in your research or business applications, please cite:

```
Team DS-4 (Khuwaja, A., Barril, F.C.F., Faisal, M., Manoharan, S., & Islaih, S.). (2024).
Pharmaceutical Drug Spending Clustering Analysis: Market Segmentation for Strategic Entry.
University of Toronto Data Science Program, Cohort 7.
https://github.com/saranya-mano/DS04-Team-Project
```

---

## 📞 Contact & Support

For questions, collaboration opportunities, or feedback:

- **Repository Issues:** [https://github.com/saranya-mano/DS04-Team-Project/issues](https://github.com/saranya-mano/DS04-Team-Project/issues)
- **Team Lead:** Saranya Manoharan (repository owner)
- **Project Maintainers:** Team DS-4 (all members)

---

## ✨ Acknowledgments

We would like to thank:
- **University of Toronto Data Science Program** for the structured learning framework and project guidance
- **OECD** for providing high-quality, publicly accessible healthcare data
- **Open-source community** for the excellent Python libraries that made this analysis possible
- **Our instructional team** for feedback on the project proposal and methodology

---

**Last Updated:** November 13, 2024
**Project Status:** ✅ Complete and reproducible
**Recommended Citation Format:** See License & Citation section above
