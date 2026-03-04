# Clustering Metrics Explanation

## What Do These Metrics Mean?

### Silhouette Score (-1 to 1)
- **Range**: -1 to 1
- **Higher is better**
- Measures how similar customers are to their own cluster vs other clusters
- > 0.5: Good separation
- > 0.7: Strong structure
- **Our result**: K-Means = 0.31 (Moderate)

### Calinski-Harabasz Index (Higher is better)
- **Range**: 0 to ∞
- Ratio of between-cluster dispersion to within-cluster dispersion
- **Our result**: K-Means = 2166 (Good)

### Davies-Bouldin Index (Lower is better)
- **Range**: 0 to ∞
- Average similarity between clusters (lower = better separation)
- **Our result**: K-Means = 1.15 (Moderate)

### Dunn Index (Higher is better)
- **Range**: 0 to ∞
- Ratio of smallest inter-cluster distance to largest intra-cluster distance
- **Our result**: K-Means = 1.50 (Good)

## Cluster Quality Summary

| Algorithm | Clusters | Silhouette | Interpretation |
|-----------|----------|------------|----------------|
| K-Means | 4 | 0.31 | Moderate separation - Best choice |
| DBSCAN | 3 | 0.21 | Weaker - detects outliers |
| Hierarchical | 4 | 0.26 | Moderate separation |

## Recommendations

1. **Use K-Means** for primary segmentation (best silhouette score)
2. **Premium Bulk Buyers** (Cluster 2) should receive highest priority
3. Focus retention efforts on **Regular Loyalists** (Cluster 1)
4. Re-engage **One-Time Shoppers** (Cluster 3) with win-back campaigns
