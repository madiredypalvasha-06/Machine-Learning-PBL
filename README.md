# Customer Segmentation & Market Basket Analysis

This project performs customer segmentation and market basket analysis on retail transaction data using machine learning techniques.

## Dataset

- **Source**: Online Retail Dataset (Excel file)
- **Records**: ~398,000 transactions
- **Customers**: 4,338 unique customers

## Features Analyzed

### RFM Metrics
- **Recency**: Days since last purchase
- **Frequency**: Number of orders placed
- **Monetary**: Total spending
- **Average Basket Size**: Average order value
- **Unique Products**: Number of distinct products purchased
- **CLV (Customer Lifetime Value)**: Total revenue from customer

## Customer Segments (K-Means Clustering)

| Cluster | Name | Customers | Avg CLV | Avg Orders | Description |
|---------|------|-----------|---------|------------|-------------|
| 0 | Moderate Recent | 295 | $2,412 | 2.3 | Moderate engagement, some recent activity |
| 1 | Frequent Buyers | 1,638 | $982 | 3.3 | Regular customers with recent purchases |
| 2 | **High Value Bulk Buyers** | **881** | **$6,978** | **12.2** | **Best customers - highest CLV, most orders, most products** |
| 3 | At-Risk | 1,524 | $291 | 1.1 | One-time/rare buyers, need re-engagement |

## Key Findings

### Bulk Buyers (Cluster 2)
- **Highest CLV**: $6,978 average
- **Most orders**: 12.2 per customer
- **Most products**: 152 unique products
- **Most recent**: 19 days average recency
- **Strategy**: Wholesale accounts, bulk discounts, priority support

### Association Rules (Market Basket Analysis)
Top rule: Regency Teacup sets (Rose → Green) with **89% confidence**

## Running the Project

```bash
# Activate virtual environment
source venv/bin/activate

# Run with default data path
python3 ml.py

# Or specify custom data path
DATA_PATH='/path/to/Online Retail.xlsx' python3 ml.py
```

## Output Files

| File | Description |
|------|-------------|
| `customer_clusters.csv` | All customers with cluster assignments |
| `cluster_descriptions.csv` | Summary of each cluster |
| `cluster_profiles_kmeans.csv` | Detailed cluster metrics |
| `clv_analysis.csv` | CLV analysis by cluster |
| `association_rules.csv` | Market basket association rules |
| `frequent_itemsets.csv` | Frequent product combinations |
| `metrics.csv` | Clustering performance metrics |

## Visualizations

Generated in `reports/figures/`:
- `tsne_visualization.png` - t-SNE 2D projection
- `clv_distribution.png` - CLV distribution by cluster
- `cluster_radar.png` - Cluster profile radar chart
- `cluster_products.png` - Top products per cluster

## Technologies Used

- Python 3
- scikit-learn (K-Means, DBSCAN, Hierarchical Clustering)
- pandas, numpy
- matplotlib
- mlxtend (Apriori algorithm)
