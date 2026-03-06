# Customer Segmentation & Market Basket Analysis

This project performs customer segmentation and market basket analysis on retail transaction data using machine learning techniques.

## Dataset

- **Source**: Online Retail Dataset (Excel file)
- **Records**: ~398,000 transactions
- **Customers**: 4,338 unique customers
- **Date Range**: 13 months

## Features Analyzed

### RFM Metrics
- **Recency**: Days since last purchase
- **Frequency**: Number of orders placed
- **Monetary**: Total spending
- **CLV (Customer Lifetime Value)**: Total revenue from customer
- **Average Basket Size**: Average order value
- **Unique Products**: Number of distinct products purchased

---

## Customer Segments (K-Means Clustering)

| Segment | Customers | Avg CLV | Revenue % | Description |
|---------|-----------|---------|-----------|-------------|
| **Premium Bulk Buyers** | 881 | $6,978 | **69%** | Highest value, most orders (12.2), most products |
| Regular Loyalists | 1,638 | $982 | 18% | Good engagement, recent purchases |
| Occasional Buyers | 295 | $2,412 | 8% | Moderate engagement |
| One-Time Shoppers | 1,524 | $291 | 5% | At-risk, need re-engagement |

### Key Insight
**Premium Bulk Buyers** contribute **69% of total revenue** despite being only 20% of customers!

---

## RFM Scoring (1-5 Scale)

| Segment | Customers | Avg CLV | Strategy |
|---------|-----------|---------|----------|
| **Champions** | 439 | $9,204 | VIP treatment, exclusive offers |
| Loyal Customers | 194 | $1,549 | Loyalty rewards |
| Potential Loyalist | 245 | $1,541 | Nurture campaigns |
| Recent Customers | 136 | $2,010 | Build relationship |
| Needs Attention | 188 | $903 | Re-engagement |
| At Risk | 185 | $487 | Win-back |
| Hibernating | 208 | $554 | Special offers |
| Lost | 366 | $545 | Aggressive win-back |

---

## Market Basket Analysis (Association Rules)

**Top Rule**: Regency Teacup Sets
- If customer buys **Pink + Rose** Regency Teacup → 89% will buy **Green** too
- Lift: 24.0 (very strong association)

**Top Frequent Items**:
1. Regency Teacup Sets (various colors)
2. Gardeners Kneeling Pad
3. Lunch Boxes (Dolly Girl / Spaceboy)
4. Alarm Clocks (Bakelike)

---

## Time Analysis

- **Peak Revenue Month**: November 2011 ($1.16M)
- **Total Period**: 13 months
- Strong seasonal patterns detected (holiday shopping)

---

## Country Analysis

| Country | Revenue | Customers |
|---------|---------|-----------|
| United Kingdom | $7.3M | 3,920 |
| Netherlands | $285K | 9 |
| EIRE | $266K | 3 |
| Germany | $229K | 94 |
| France | $209K | 87 |

---

## Clustering Quality Metrics

| Algorithm | Clusters | Silhouette | Quality |
|-----------|----------|------------|---------|
| K-Means | 4 | 0.31 | **Best** |
| DBSCAN | 3 | 0.21 | Good for outliers |
| Hierarchical | 4 | 0.26 | Moderate |

---

## Running the Project

```bash
# Activate virtual environment
source venv/bin/activate

# Run with data
python3 ml.py
# Or specify path:
DATA_PATH='/path/to/Online Retail.xlsx' python3 ml.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `customer_clusters.csv` | All customers with cluster + RFM scores |
| `cluster_descriptions.csv` | Segment summary |
| `clv_analysis.csv` | Detailed CLV metrics |
| `rfm_analysis.csv` | RFM segment analysis |
| `revenue_contribution.csv` | Revenue by segment |
| `time_analysis.csv` | Monthly trends |
| `country_analysis.csv` | Top countries |
| `association_rules.csv` | Market basket rules |
| `frequent_itemsets.csv` | Product combinations |
| `metrics.csv` | Clustering quality |

## Visualizations (in `reports/figures/`)

- `tsne_visualization.png` - Customer clusters
- `cluster_radar.png` - Segment profiles
- `clv_distribution.png` - CLV by segment
- `segment_distribution.png` - Revenue & customers pie charts
- `time_trends.png` - Monthly trends
- `association_rules.png` - Market basket rules
- `frequent_itemsets.png` - Top products

---

## Fraud Detection Analysis

| Metric | Value |
|--------|-------|
| Total Cancelled Orders | 8,905 |
| Unique Cancelled Customers | 1,589 |
| Customers with 5+ Cancellations | 150 |
| **High Risk Customers** | 176 |
| Total Cancel Amount | $611,342 |

### High Risk Indicators
- Customers with **3+ cancellations** AND **30%+ cancel ratio**
- Products with high cancellation rates (potential quality issues)
- Top cancelled: Discounts, CRUK Commission, certain decorative items

### Recommendations
- Investigate **176 high-risk customers** for potential fraud
- Review products with >50% cancellation rate
- Implement verification for bulk orders followed by cancellations

---

## Business Recommendations

1. **Focus on Premium Bulk Buyers** (69% of revenue) - wholesale accounts, priority support
2. **Convert Champions** - They have highest CLV ($9,204), offer VIP programs
3. **Win-back One-Time Shoppers** - 35% of customers, low-cost campaigns
4. **Cross-sell** - Use tea cup set associations for bundle deals
5. **Peak Season** - November is critical (prepare inventory)
6. **Fraud Prevention** - Monitor high-risk customers with high cancellation rates
