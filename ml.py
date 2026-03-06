import os
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.size'] = 10
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    APRIORI_AVAILABLE = True
except ImportError:
    APRIORI_AVAILABLE = False
    print("Installing mlxtend...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlxtend", "-q"])
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    APRIORI_AVAILABLE = True

DEFAULT_DATA_PATH = r"C:\Users\dhrit\Downloads\online+retail\Online Retail.xlsx"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH)
REPORTS_DIR = os.path.join(os.getcwd(), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
MODEL_DIR = os.path.join(REPORTS_DIR, "models")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")
xl = pd.ExcelFile(DATA_PATH)
print("Sheets:", xl.sheet_names)
df = pd.read_excel(DATA_PATH, sheet_name=xl.sheet_names[0])
print("Raw shape:", df.shape)

df = df.copy()
if "CustomerID" in df.columns:
    df = df.dropna(subset=["CustomerID"])

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print("\n" + "="*60)
print("FRAUD DETECTION ANALYSIS")
print("="*60)

df_original = df.copy()

cancelled_mask = df["InvoiceNo"].astype(str).str.startswith("C")
df_cancelled = df[cancelled_mask].copy()
df_valid = df[~cancelled_mask].copy()

print(f"Total Records: {len(df)}")
print(f"Cancelled Orders: {len(df_cancelled)} ({len(df_cancelled)/len(df)*100:.1f}%)")
print(f"Valid Orders: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")

df_cancelled["InvoiceNo"] = df_cancelled["InvoiceNo"].astype(str).str.replace("C", "", regex=False)
df_cancelled["Quantity"] = df_cancelled["Quantity"].abs()

cancelled_by_customer = df_cancelled.groupby("CustomerID").agg({
    "InvoiceNo": "nunique",
    "TotalPrice": ["sum", "mean"],
    "Quantity": "sum"
}).reset_index()
cancelled_by_customer.columns = ["CustomerID", "Cancel_Count", "Cancel_Amount", "Avg_Cancel_Value", "Cancel_Quantity"]
cancelled_by_customer = cancelled_by_customer.sort_values("Cancel_Amount", ascending=False)
cancelled_by_customer.to_csv(os.path.join(REPORTS_DIR, "fraud_customers.csv"), index=False)

high_cancel_customers = cancelled_by_customer[cancelled_by_customer["Cancel_Count"] >= 5]
print(f"\nCustomers with 5+ cancellations: {len(high_cancel_customers)}")
print("Top 5 Cancellation Customers:")
print(high_cancel_customers.head())

product_cancel_rates = df_cancelled.groupby("Description").agg({
    "Quantity": "sum",
    "InvoiceNo": "nunique"
}).reset_index()
product_cancel_rates.columns = ["Product", "Cancel_Qty", "Cancel_Orders"]

valid_product_qty = df_valid.groupby("Description")["Quantity"].sum().reset_index()
valid_product_qty.columns = ["Product", "Valid_Qty"]

product_analysis = product_cancel_rates.merge(valid_product_qty, on="Product", how="outer").fillna(0)
product_analysis["Cancel_Rate"] = product_analysis["Cancel_Qty"] / (product_analysis["Cancel_Qty"] + product_analysis["Valid_Qty"] + 1) * 100
product_analysis = product_analysis.sort_values("Cancel_Rate", ascending=False)
product_analysis.to_csv(os.path.join(REPORTS_DIR, "product_cancellation_analysis.csv"), index=False)
print("\nTop 5 Most Cancelled Products (by rate):")
print(product_analysis.head())

plt.figure(figsize=(10, 5))
top_cancel_products = product_analysis[product_analysis["Cancel_Orders"] >= 10].head(10)
plt.barh(range(len(top_cancel_products)), top_cancel_products["Cancel_Rate"].values)
plt.yticks(range(len(top_cancel_products)), top_cancel_products["Product"].values[:10])
plt.xlabel('Cancellation Rate (%)')
plt.title('Products with Highest Cancellation Rates')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fraud_products.png"), dpi=150)
plt.close()

print("\nFraud detection reports saved!")

df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

print("Clean shape:", df.shape)

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
max_date = df["InvoiceDate"].max()

agg = df.groupby("CustomerID").agg(
    last_purchase=("InvoiceDate", "max"),
    first_purchase=("InvoiceDate", "min"),
    frequency=("InvoiceNo", "nunique"),
    monetary=("TotalPrice", "sum"),
    avg_basket=("TotalPrice", "mean"),
    unique_products=("StockCode", "nunique"),
)

agg["recency"] = (max_date - agg["last_purchase"]).dt.days
agg["lifespan_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days + 1
agg["avg_order_value"] = agg["monetary"] / agg["frequency"]
agg["purchase_frequency_per_day"] = agg["frequency"] / agg["lifespan_days"]
agg["clv"] = agg["monetary"]

features = agg[["recency", "frequency", "monetary", "avg_basket", "unique_products", "clv", "lifespan_days"]].copy()

print("\n=== Customer Lifetime Value (CLV) Analysis ===")
print(f"Average CLV: ${agg['clv'].mean():.2f}")
print(f"Median CLV: ${agg['clv'].median():.2f}")
print(f"Max CLV: ${agg['clv'].max():.2f}")
print(f"Average Customer Lifespan: {agg['lifespan_days'].mean():.1f} days")

print("\n=== Enhanced Fraud Detection with Customer Metrics ===")
fraud_indicators = cancelled_by_customer.merge(
    agg[["monetary", "frequency", "clv"]].reset_index(),
    on="CustomerID", how="left"
)
fraud_indicators = fraud_indicators.fillna(0)
fraud_indicators["Cancel_Ratio"] = fraud_indicators["Cancel_Count"] / (fraud_indicators["frequency"] + fraud_indicators["Cancel_Count"] + 1)
fraud_indicators["Cancel_to_Spend_Ratio"] = fraud_indicators["Cancel_Amount"] / (fraud_indicators["monetary"] + 1)

high_risk = fraud_indicators[
    (fraud_indicators["Cancel_Count"] >= 3) & 
    (fraud_indicators["Cancel_Ratio"] > 0.3)
].sort_values("Cancel_Amount", ascending=False)
high_risk.to_csv(os.path.join(REPORTS_DIR, "high_risk_customers.csv"), index=False)
print(f"High Risk Customers (3+ cancellations, 30%+ cancel ratio): {len(high_risk)}")

fraud_summary = pd.DataFrame({
    "Metric": ["Total Cancelled Orders", "Unique Cancelled Customers", "Customers with 5+ Cancellations", "High Risk Customers", "Total Cancel Amount"],
    "Value": [
        len(df_cancelled),
        df_cancelled["CustomerID"].nunique(),
        len(cancelled_by_customer[cancelled_by_customer["Cancel_Count"] >= 5]),
        len(high_risk),
        f"${abs(df_cancelled['TotalPrice'].sum()):,.2f}"
    ]
})
fraud_summary.to_csv(os.path.join(REPORTS_DIR, "fraud_summary.csv"), index=False)
print(fraud_summary)

features_log = np.log1p(features)
scaler = StandardScaler()
X = scaler.fit_transform(features_log)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X)
joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))

ks = list(range(2, 11))
km_inertia = []
km_sil = []
km_ch_scores = []
km_db_scores = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    km_inertia.append(km.inertia_)
    try:
        km_sil.append(silhouette_score(X, labels))
        km_ch_scores.append(calinski_harabasz_score(X, labels))
        km_db_scores.append(davies_bouldin_score(X, labels))
    except Exception:
        km_sil.append(np.nan)
        km_ch_scores.append(np.nan)
        km_db_scores.append(np.nan)

best_k = 4

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
km_labels = kmeans.fit_predict(X)
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))

min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples)
nbrs.fit(X)
k_distances = np.sort(nbrs.kneighbors(X)[0][:, -1])
eps = float(np.percentile(k_distances, 90))

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
db_labels = dbscan.fit_predict(X)
joblib.dump(dbscan, os.path.join(MODEL_DIR, "dbscan.pkl"))

hc_sil_scores = []
hc_ch_scores = []
hc_db_scores = []

for n_clust in range(2, 8):
    hc = AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
    hc_labels = hc.fit_predict(X)
    hc_sil_scores.append(silhouette_score(X, hc_labels))
    hc_ch_scores.append(calinski_harabasz_score(X, hc_labels))
    hc_db_scores.append(davies_bouldin_score(X, hc_labels))

best_hc_k = 4
hc_final = AgglomerativeClustering(n_clusters=best_hc_k, linkage='ward')
hc_labels = hc_final.fit_predict(X)
joblib.dump(hc_final, os.path.join(MODEL_DIR, "hierarchical.pkl"))


def calculate_dunn_index(X, labels):
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return np.nan
    intra_dists = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            distances = pdist(cluster_points)
            intra_dists.append(np.min(distances))
    if not intra_dists or max(intra_dists) == 0:
        return np.nan
    inter_dists = []
    for i, label1 in enumerate(unique_labels):
        for label2 in list(unique_labels)[i+1:]:
            cluster1 = X[labels == label1]
            cluster2 = X[labels == label2]
            if len(cluster1) > 0 and len(cluster2) > 0:
                dist = np.min(cdist(cluster1, cluster2))
                inter_dists.append(dist)
    if not inter_dists:
        return np.nan
    return np.min(inter_dists) / np.max(intra_dists)


def calculate_inter_cluster_distance(X, labels):
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return np.nan
    centroids = []
    for label in unique_labels:
        centroids.append(X[labels == label].mean(axis=0))
    centroids = np.array(centroids)
    distances = pdist(centroids)
    return np.mean(distances)


def calculate_intra_cluster_distance(X, labels):
    unique_labels = set(labels) - {-1}
    total_dist = 0
    count = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            dist = np.mean(pdist(cluster_points))
            total_dist += dist
            count += 1
    return total_dist / count if count > 0 else np.nan


km_dunn = calculate_dunn_index(X, km_labels)
km_inter = calculate_inter_cluster_distance(X, km_labels)
km_intra = calculate_intra_cluster_distance(X, km_labels)

db_valid_mask = db_labels != -1
if sum(db_valid_mask) > 0:
    db_dunn = calculate_dunn_index(X[db_valid_mask], db_labels[db_valid_mask])
else:
    db_dunn = np.nan

hc_dunn = calculate_dunn_index(X, hc_labels)
hc_inter = calculate_inter_cluster_distance(X, hc_labels)
hc_intra = calculate_intra_cluster_distance(X, hc_labels)


def safe_silhouette(data, labels):
    unique = set(labels)
    if len(unique) <= 1:
        return np.nan
    if unique == {-1}:
        return np.nan
    if len(unique - {-1}) <= 1:
        return np.nan
    return silhouette_score(data, labels)


km_sil_best = safe_silhouette(X, km_labels)
clusters_db = len(set(db_labels) - {-1})

metrics = pd.DataFrame([
    {
        "algorithm": "KMeans",
        "n_clusters": int(best_k),
        "silhouette": float(km_sil_best),
        "calinski_harabasz": float(calinski_harabasz_score(X, km_labels)),
        "davies_bouldin": float(davies_bouldin_score(X, km_labels)),
        "dunn_index": float(km_dunn),
        "inter_cluster_dist": float(km_inter),
        "intra_cluster_dist": float(km_intra),
        "params": json.dumps({"n_clusters": int(best_k)})
    },
    {
        "algorithm": "DBSCAN",
        "n_clusters": int(clusters_db),
        "silhouette": float(safe_silhouette(X, db_labels)) if clusters_db > 1 else np.nan,
        "calinski_harabasz": float(calinski_harabasz_score(X[db_valid_mask], db_labels[db_valid_mask])) if clusters_db > 1 else np.nan,
        "davies_bouldin": float(davies_bouldin_score(X[db_valid_mask], db_labels[db_valid_mask])) if clusters_db > 1 else np.nan,
        "dunn_index": float(db_dunn) if not np.isnan(db_dunn) else np.nan,
        "inter_cluster_dist": np.nan,
        "intra_cluster_dist": np.nan,
        "params": json.dumps({"eps": eps, "min_samples": min_samples})
    },
    {
        "algorithm": "Hierarchical",
        "n_clusters": int(best_hc_k),
        "silhouette": float(hc_sil_scores[best_hc_k - 2]),
        "calinski_harabasz": float(hc_ch_scores[best_hc_k - 2]),
        "davies_bouldin": float(hc_db_scores[best_hc_k - 2]),
        "dunn_index": float(hc_dunn),
        "inter_cluster_dist": float(hc_inter),
        "intra_cluster_dist": float(hc_intra),
        "params": json.dumps({"n_clusters": int(best_hc_k), "linkage": "ward"})
    }
])

metrics_path = os.path.join(REPORTS_DIR, "metrics.csv")
metrics.to_csv(metrics_path, index=False)

output = features.copy()
output["clv"] = agg["clv"]
output["lifespan_days"] = agg["lifespan_days"]
output["kmeans_cluster"] = km_labels
output["dbscan_cluster"] = db_labels
output["hierarchical_cluster"] = hc_labels

cluster_names = {
    0: "Occasional Buyers",
    1: "Regular Loyalists",
    2: "Premium Bulk Buyers",
    3: "One-Time Shoppers"
}
output["cluster_name"] = output["kmeans_cluster"].map(cluster_names)

output.to_csv(os.path.join(REPORTS_DIR, "customer_clusters.csv"))

print("\n" + "="*60)
print("CLUSTER ANALYSIS SUMMARY")
print("="*60)

for cluster_id in range(best_k):
    cluster_data = output[output["kmeans_cluster"] == cluster_id]
    name = cluster_names[cluster_id]
    
    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster_id}: {name}")
    print(f"{'='*50}")
    print(f"  Number of Customers: {len(cluster_data)}")
    print(f"  Avg CLV: ${cluster_data['clv'].mean():,.2f}")
    print(f"  Avg Spending: ${cluster_data['monetary'].mean():,.2f}")
    print(f"  Avg Purchase Frequency: {cluster_data['frequency'].mean():.1f} orders")
    print(f"  Avg Recency: {cluster_data['recency'].mean():.0f} days")
    print(f"  Avg Lifespan: {cluster_data['lifespan_days'].mean():.0f} days")
    print(f"  Avg Products Purchased: {cluster_data['unique_products'].mean():.0f}")
    
    if cluster_id == 2:
        print(f"\n  ➤ PREMIUM BULK BUYERS: Highest CLV (${cluster_data['clv'].mean():,.0f}), most orders ({cluster_data['frequency'].mean():.1f}), most products ({cluster_data['unique_products'].mean():.0f})")
        print(f"  ➤ Strategy: Wholesale accounts, bulk discounts, priority support")
    elif cluster_id == 1:
        print(f"\n  ➤ REGULAR LOYALISTS: Good engagement (${cluster_data['clv'].mean():,.0f} CLV), {cluster_data['frequency'].mean():.1f} orders, recent activity ({cluster_data['recency'].mean():.0f} days)")
        print(f"  ➤ Strategy: Loyalty programs, personalized offers, referral bonuses")
    elif cluster_id == 0:
        print(f"\n  ➤ OCCASIONAL BUYERS: Moderate CLV (${cluster_data['clv'].mean():,.0f}), some engagement ({cluster_data['frequency'].mean():.1f} orders)")
        print(f"  ➤ Strategy: Nurture with product recommendations, upselling campaigns")
    else:
        print(f"\n  ➤ ONE-TIME SHOPPERS: Low CLV (${cluster_data['clv'].mean():,.0f}), only {cluster_data['frequency'].mean():.1f} order(s), haven't returned ({cluster_data['recency'].mean():.0f} days ago)")
        print(f"  ➤ Strategy: Win-back campaigns, special comeback offers")

cluster_descriptions = pd.DataFrame({
    'Cluster_ID': range(best_k),
    'Segment_Name': [cluster_names[i] for i in range(best_k)],
    'Number_of_Customers': [len(output[output['kmeans_cluster']==i]) for i in range(best_k)],
    'Avg_CLV_USD': [round(output[output['kmeans_cluster']==i]['clv'].mean(), 2) for i in range(best_k)],
    'Avg_Total_Spending_USD': [round(output[output['kmeans_cluster']==i]['monetary'].mean(), 2) for i in range(best_k)],
    'Avg_Orders': [round(output[output['kmeans_cluster']==i]['frequency'].mean(), 1) for i in range(best_k)],
    'Avg_Recency_Days': [round(output[output['kmeans_cluster']==i]['recency'].mean(), 0) for i in range(best_k)],
    'Avg_Lifespan_Days': [round(output[output['kmeans_cluster']==i]['lifespan_days'].mean(), 0) for i in range(best_k)],
    'Avg_Products_Purchased': [round(output[output['kmeans_cluster']==i]['unique_products'].mean(), 0) for i in range(best_k)],
    'Business_Strategy': [
        'Nurture with product recommendations, upselling campaigns',
        'Loyalty programs, personalized offers, referral bonuses',
        'Wholesale accounts, bulk discounts, priority support',
        'Win-back campaigns, special comeback offers'
    ]
})
cluster_descriptions.to_csv(os.path.join(REPORTS_DIR, "cluster_descriptions.csv"), index=False)
print(f"\n\nCluster descriptions saved to: reports/cluster_descriptions.csv")

clv_profile = output.groupby("kmeans_cluster")[["clv", "lifespan_days", "recency", "frequency", "monetary"]].agg(['mean', 'median', 'std'])
clv_profile.columns = ['_'.join(col).strip() for col in clv_profile.columns.values]
clv_profile = clv_profile.rename(columns={
    'clv_mean': 'CLV_Mean_USD', 'clv_median': 'CLV_Median_USD', 'clv_std': 'CLV_StdDev',
    'lifespan_days_mean': 'Avg_Lifespan_Days', 'lifespan_days_median': 'Median_Lifespan_Days', 'lifespan_days_std': 'Lifespan_StdDev',
    'recency_mean': 'Avg_Recency_Days', 'recency_median': 'Median_Recency_Days', 'recency_std': 'Recency_StdDev',
    'frequency_mean': 'Avg_Orders', 'frequency_median': 'Median_Orders', 'frequency_std': 'Orders_StdDev',
    'monetary_mean': 'Avg_Spending_USD', 'monetary_median': 'Median_Spending_USD', 'monetary_std': 'Spending_StdDev'
})
clv_profile['Segment_Name'] = [cluster_names[i] for i in clv_profile.index]
clv_profile = clv_profile[['Segment_Name', 'CLV_Mean_USD', 'CLV_Median_USD', 'CLV_StdDev', 
                            'Avg_Orders', 'Median_Orders', 'Avg_Spending_USD', 'Median_Spending_USD',
                            'Avg_Recency_Days', 'Median_Recency_Days', 'Avg_Lifespan_Days', 'Median_Lifespan_Days']]
clv_profile.to_csv(os.path.join(REPORTS_DIR, "clv_analysis.csv"))

kmeans_profile = output.groupby("kmeans_cluster")[["recency", "frequency", "monetary", "avg_basket", "unique_products", "clv"]].agg(['mean', 'std', 'median', 'min', 'max'])
kmeans_profile.to_csv(os.path.join(REPORTS_DIR, "cluster_profiles_kmeans.csv"))

dbscan_profile = output.groupby("dbscan_cluster")[["recency", "frequency", "monetary"]].mean().sort_index()
dbscan_profile.to_csv(os.path.join(REPORTS_DIR, "cluster_profiles_dbscan.csv"))

hierarchical_profile = output.groupby("hierarchical_cluster")[["recency", "frequency", "monetary"]].mean().sort_index()
hierarchical_profile.to_csv(os.path.join(REPORTS_DIR, "cluster_profiles_hierarchical.csv"))

plt.figure(figsize=(6, 4))
plt.plot(ks, km_inertia, marker="o")
plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kmeans_elbow.png"), dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(ks, km_sil, marker="o", color="blue")
plt.title("K-Means Silhouette Score")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kmeans_silhouette.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(ks, km_sil, marker="o", color="blue")
axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("k")
axes[0].grid(True, alpha=0.3)
axes[1].plot(ks, km_ch_scores, marker="o", color="green")
axes[1].set_title("Calinski-Harabasz Index")
axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.3)
axes[2].plot(ks, km_db_scores, marker="o", color="red")
axes[2].set_title("Davies-Bouldin Index")
axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "kmeans_all_metrics.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, s=15, cmap="viridis", alpha=0.7)
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", marker="X", s=200, edgecolors="black")
plt.colorbar(scatter, label="Cluster")
plt.title(f"K-Means Clusters (k={best_k}) - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_kmeans.png"), dpi=150)
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=km_labels, s=10, cmap="viridis", alpha=0.6)
plt.colorbar(scatter, label="Cluster", shrink=0.5)
ax.set_title(f"K-Means Clusters (k={best_k}) - 3D PCA")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_kmeans_3d.png"), dpi=150)
plt.close()

print("\n=== t-SNE Visualization (Non-linear) ===")
print("Computing t-SNE... (this may take a moment)")
sample_size = min(3000, len(X))
np.random.seed(42)
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_idx]
labels_sample = km_labels[sample_idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_sample)
print("t-SNE completed!")

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_sample, s=15, cmap="viridis", alpha=0.7)
plt.title("K-Means Clusters - t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tsne_kmeans.png"), dpi=150)
plt.close()

print("t-SNE visualization saved!")

plt.figure(figsize=(6, 4))
plt.plot(k_distances)
plt.axhline(y=eps, color='r', linestyle='--', label=f'eps={eps:.2f}')
plt.title("DBSCAN k-distance Graph")
plt.xlabel("Points (sorted)")
plt.ylabel("k-distance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dbscan_kdistance.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
mask_noise = db_labels == -1
plt.scatter(X_pca[~mask_noise, 0], X_pca[~mask_noise, 1], c=db_labels[~mask_noise], s=15, cmap="tab10", alpha=0.7)
plt.scatter(X_pca[mask_noise, 0], X_pca[mask_noise, 1], c="gray", s=10, alpha=0.3, label="Noise")
plt.title("DBSCAN Clusters - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_dbscan.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X[:1000], method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dendrogram.png"), dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(range(2, 8), hc_sil_scores, marker="o", color="green")
plt.title("Hierarchical Clustering Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hierarchical_silhouette.png"), dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels, s=15, cmap="Set1", alpha=0.7)
plt.title(f"Hierarchical Clusters (k={best_hc_k}) - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_hierarchical.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
algorithms = metrics["algorithm"].tolist()
colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

for i, (metric, title) in enumerate([("silhouette", "Silhouette (Higher is Better)"), ("calinski_harabasz", "Calinski-Harabasz (Higher is Better)"), ("davies_bouldin", "Davies-Bouldin (Lower is Better)")]):
    vals = metrics[metric].tolist()
    axes[i].bar(algorithms, vals, color=colors)
    axes[i].set_title(title)
    axes[i].set_ylabel("Score")
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "metrics_comparison.png"), dpi=150)
plt.close()

silhouette_vals_kmeans = silhouette_samples(X, km_labels)
silhouette_avg_kmeans = silhouette_score(X, km_labels)

plt.figure(figsize=(10, 7))
y_lower = 10
cmap = plt.cm.viridis
for i in range(best_k):
    cluster_silhouette_vals = silhouette_vals_kmeans[km_labels == i]
    cluster_silhouette_vals.sort()
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=cmap(i/best_k), alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
plt.axvline(x=silhouette_avg_kmeans, color="red", linestyle="--", label=f"Avg: {silhouette_avg_kmeans:.3f}")
plt.title("K-Means Silhouette Analysis per Cluster")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "silhouette_analysis.png"), dpi=150)
plt.close()

cluster_dist = output["kmeans_cluster"].value_counts().sort_index()
plt.figure(figsize=(8, 5))
cluster_colors = plt.cm.viridis(np.linspace(0, 1, best_k))
bars = plt.bar(cluster_dist.index, cluster_dist.values, color=cluster_colors)
plt.title("Customer Distribution Across K-Means Clusters")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.xticks(range(best_k))
for i, v in enumerate(cluster_dist.values):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_distribution.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for cluster_id in range(best_k):
    cluster_data = output[output["kmeans_cluster"] == cluster_id]
    axes[0].boxplot([cluster_data["recency"]], positions=[cluster_id], widths=0.6)
    axes[1].boxplot([cluster_data["frequency"]], positions=[cluster_id], widths=0.6)
    axes[2].boxplot([cluster_data["monetary"]], positions=[cluster_id], widths=0.6)

axes[0].set_title("Recency by Cluster")
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("Days")
axes[1].set_title("Frequency by Cluster")
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("Invoices")
axes[2].set_title("Monetary by Cluster")
axes[2].set_xlabel("Cluster")
axes[2].set_ylabel("Total Spending ($)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rfm_boxplots.png"), dpi=150)
plt.close()

cluster_means = output.groupby("kmeans_cluster")[["recency", "frequency", "monetary", "avg_basket", "unique_products"]].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

plt.figure(figsize=(10, 5))
plt.imshow(cluster_means_norm.T, cmap="RdYlGn", aspect="auto")
plt.xticks(range(best_k), [f"Cluster {i}" for i in range(best_k)])
plt.yticks(range(5), ["Recency", "Frequency", "Monetary", "Avg Basket", "Products"])
plt.colorbar(label="Normalized Value")
plt.title("Cluster Characteristics Heatmap (Normalized)")
for i in range(5):
    for j in range(best_k):
        plt.text(j, i, f"{cluster_means.iloc[j, i]:.0f}", ha="center", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_heatmap.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
for cluster_id in range(best_k):
    cluster_clv = output[output["kmeans_cluster"] == cluster_id]["clv"]
    plt.hist(cluster_clv, bins=30, alpha=0.5, label=f'Cluster {cluster_id}')
plt.xlabel('Customer Lifetime Value ($)')
plt.ylabel('Frequency')
plt.title('CLV Distribution by Cluster')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "clv_distribution.png"), dpi=150)
plt.close()

cluster_means_norm = output.groupby("kmeans_cluster")[["recency", "frequency", "monetary", "avg_basket", "unique_products", "clv"]].mean()
cluster_means_norm = (cluster_means_norm - cluster_means_norm.min()) / (cluster_means_norm.max() - cluster_means_norm.min() + 1e-10)

categories = ["Recency", "Frequency", "Monetary", "Avg Basket", "Products", "CLV"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
colors = plt.cm.viridis(np.linspace(0, 1, best_k))

for cluster_id in range(best_k):
    values = cluster_means_norm.loc[cluster_id].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}', color=colors[cluster_id])
    ax.fill(angles, values, alpha=0.1, color=colors[cluster_id])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Cluster Profiles (Radar Chart)', y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_radar.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for cluster_id in range(best_k):
    ax = axes[cluster_id // 2, cluster_id % 2]
    cluster_customers = output[output["kmeans_cluster"] == cluster_id].index
    cluster_products = df[df["CustomerID"].isin(cluster_customers)].groupby("Description")["Quantity"].sum()
    top_5 = cluster_products.sort_values(ascending=False).head(5)
    ax.barh(range(len(top_5)), top_5.values, color=colors[cluster_id])
    ax.set_yticks(range(len(top_5)))
    ax.set_yticklabels([str(x)[:30] for x in top_5.index], fontsize=8)
    ax.set_xlabel('Quantity')
    ax.set_title(f'Cluster {cluster_id} - Top 5 Products')
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_products.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
cluster_sizes = output["kmeans_cluster"].value_counts().sort_index()
colors_pie = plt.cm.viridis(np.linspace(0, 1, best_k))
labels = [f'Cluster {i}\n({cluster_sizes[i]} customers)' for i in range(best_k)]
wedges, texts, autotexts = axes[0, 0].pie(cluster_sizes.values, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Customer Distribution by Cluster')

cluster_clv_means = output.groupby("kmeans_cluster")["clv"].mean()
axes[0, 1].bar(range(best_k), cluster_clv_means.values, color=colors_pie)
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Average CLV ($)')
axes[0, 1].set_title('Average CLV by Cluster')
axes[0, 1].set_xticks(range(best_k))

cluster_monetary = output.groupby("kmeans_cluster")["monetary"].mean()
axes[1, 0].bar(range(best_k), cluster_monetary.values, color=colors_pie)
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Average Spending ($)')
axes[1, 0].set_title('Average Spending by Cluster')
axes[1, 0].set_xticks(range(best_k))

cluster_recency = output.groupby("kmeans_cluster")["recency"].mean()
axes[1, 1].bar(range(best_k), cluster_recency.values, color=colors_pie)
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Avg Days Since Last Purchase')
axes[1, 1].set_title('Average Recency by Cluster')
axes[1, 1].set_xticks(range(best_k))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_summary.png"), dpi=150)
plt.close()

print("\n=== Cluster Visualizations Saved ===")
print("- clv_distribution.png: CLV distribution per cluster")
print("- cluster_radar.png: Radar chart for cluster profiles")
print("- cluster_products.png: Top products per cluster")
print("- cluster_summary.png: Cluster summary statistics")

print("\n=== Association Rule Mining (Apriori Algorithm) ===")

transactions = df.groupby("InvoiceNo")["Description"].apply(lambda x: list(set(x.dropna()))).tolist()
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values('confidence', ascending=False).head(20)

print(f"\nFound {len(frequent_itemsets)} frequent itemsets")
print(f"Found {len(rules)} association rules")

rules.to_csv(os.path.join(REPORTS_DIR, "association_rules.csv"), index=False)
frequent_itemsets.to_csv(os.path.join(REPORTS_DIR, "frequent_itemsets.csv"), index=False)

print("\nTop 10 Association Rules by Confidence:")
for idx, row in rules.head(10).iterrows():
    print(f"  {set(row['antecedents'])} → {set(row['consequents'])}")
    print(f"    Confidence: {row['confidence']:.2%}, Support: {row['support']:.2%}, Lift: {row['lift']:.2f}")

top_1_itemsets = frequent_itemsets[frequent_itemsets['length'] == 1].nlargest(10, 'support')
plt.figure(figsize=(10, 5))
plt.barh(range(len(top_1_itemsets)), top_1_itemsets['support'].values)
plt.yticks(range(len(top_1_itemsets)), [str(x) for x in top_1_itemsets['itemsets'].values])
plt.xlabel('Support')
plt.title('Top 10 Frequent Itemsets (Single Items)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "frequent_itemsets.png"), dpi=150)
plt.close()

print("\n" + "="*60)
print("ADDITIONAL ANALYSES")
print("="*60)

print("\n--- RFM Scoring (1-5 Scale) ---")
output['R_Score'] = pd.qcut(output['recency'].rank(method='first'), q=5, labels=[5, 4, 3, 2, 1])
output['F_Score'] = pd.qcut(output['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
output['M_Score'] = pd.qcut(output['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
output['RFM_Score'] = output['R_Score'].astype(int) + output['F_Score'].astype(int) + output['M_Score'].astype(int)

rfm_segments = {
    (5, 5): "Champions",
    (5, 4): "Loyal Customers",
    (4, 4): "Potential Loyalist",
    (5, 3): "Recent Customers",
    (3, 3): "Needs Attention",
    (3, 2): "At Risk",
    (2, 2): "Hibernating",
    (1, 1): "Lost"
}
output['RFM_Segment'] = output.apply(lambda x: rfm_segments.get((int(x['R_Score']), int(x['F_Score'])), "Others"), axis=1)

rfm_summary = output.groupby('RFM_Segment').agg({
    'clv': ['count', 'mean', 'sum'],
    'frequency': 'mean',
    'recency': 'mean'
}).round(2)
rfm_summary.columns = ['Count', 'Avg_CLV', 'Total_CLV', 'Avg_Frequency', 'Avg_Recency']
rfm_summary = rfm_summary.sort_values('Total_CLV', ascending=False)
rfm_summary.to_csv(os.path.join(REPORTS_DIR, "rfm_analysis.csv"))
print(f"RFM Segments:\n{rfm_summary}")

print("\n--- Revenue Contribution by Segment ---")
segment_revenue = output.groupby('kmeans_cluster').agg({
    'monetary': 'sum',
    'clv': 'sum'
}).rename(columns={'monetary': 'Total_Revenue', 'Total_CLV': 'Total_CLV'})
segment_revenue['Revenue_Percentage'] = (segment_revenue['Total_Revenue'] / segment_revenue['Total_Revenue'].sum() * 100).round(2)
segment_revenue['Segment_Name'] = [cluster_names[i] for i in segment_revenue.index]
segment_revenue = segment_revenue[['Segment_Name', 'Total_Revenue', 'Revenue_Percentage']]
segment_revenue.to_csv(os.path.join(REPORTS_DIR, "revenue_contribution.csv"))
print(segment_revenue)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.Set2(np.linspace(0, 1, best_k))
axes[0].pie(segment_revenue['Total_Revenue'], labels=segment_revenue['Segment_Name'], autopct='%1.1f%%', colors=colors)
axes[0].set_title('Revenue Contribution by Segment')
customer_counts = output.groupby('kmeans_cluster').size()
axes[1].pie(customer_counts, labels=[cluster_names[i] for i in customer_counts.index], autopct='%1.1f%%', colors=colors)
axes[1].set_title('Customer Distribution by Segment')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "segment_distribution.png"), dpi=150)
plt.close()

print("\n--- Time-Based Analysis ---")
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
monthly_revenue = df.groupby('InvoiceMonth')['TotalPrice'].sum()
monthly_orders = df.groupby('InvoiceMonth')['InvoiceNo'].nunique()
monthly_customers = df.groupby('InvoiceMonth')['CustomerID'].nunique()

time_analysis = pd.DataFrame({
    'Month': monthly_revenue.index.astype(str),
    'Revenue': monthly_revenue.values,
    'Orders': monthly_orders.values,
    'Customers': monthly_customers.values
})
time_analysis.to_csv(os.path.join(REPORTS_DIR, "time_analysis.csv"), index=False)
print(f"Peak Revenue Month: {time_analysis.loc[time_analysis['Revenue'].idxmax(), 'Month']} (${time_analysis['Revenue'].max():,.0f})")
print(f"Total Months: {len(time_analysis)}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(time_analysis['Month'], time_analysis['Revenue'], marker='o', color='green')
plt.xticks(rotation=45)
plt.title('Monthly Revenue Trend')
plt.ylabel('Revenue ($)')
plt.grid(True, alpha=0.3)
plt.subplot(1, 2, 2)
plt.bar(time_analysis['Month'], time_analysis['Customers'], color='blue', alpha=0.7)
plt.xticks(rotation=45)
plt.title('Monthly Active Customers')
plt.ylabel('Customers')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "time_trends.png"), dpi=150)
plt.close()

print("\n--- Country Analysis ---")
country_revenue = df.groupby('Country').agg({
    'TotalPrice': 'sum',
    'CustomerID': 'nunique',
    'InvoiceNo': 'nunique'
}).rename(columns={'TotalPrice': 'Revenue', 'CustomerID': 'Customers', 'InvoiceNo': 'Orders'})
country_revenue = country_revenue.sort_values('Revenue', ascending=False).head(10)
country_revenue.to_csv(os.path.join(REPORTS_DIR, "country_analysis.csv"))
print("Top 10 Countries by Revenue:")
print(country_revenue)

top_rules = rules.head(10)
plt.figure(figsize=(10, 6))
plt.scatter(top_rules['support'], top_rules['confidence'], c=top_rules['lift'], cmap='viridis', s=100, alpha=0.7)
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules (Support vs Confidence)')
for idx, row in top_rules.head(5).iterrows():
    plt.annotate(f"Lift: {row['lift']:.1f}", (row['support'], row['confidence']))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "association_rules.png"), dpi=150)
plt.close()

print("\n--- Market Basket Analysis per Cluster ---")

top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
top_products.to_csv(os.path.join(REPORTS_DIR, "top_products.csv"))

cluster_product_freq = {}
for cluster_id in range(best_k):
    cluster_customers = output[output["kmeans_cluster"] == cluster_id].index
    cluster_products = df[df["CustomerID"].isin(cluster_customers)].groupby("Description")["Quantity"].sum()
    top_cluster_products = cluster_products.sort_values(ascending=False).head(5)
    cluster_product_freq[cluster_id] = top_cluster_products
    print(f"\nCluster {cluster_id} Top Products:")
    for prod, qty in top_cluster_products.items():
        print(f"  - {prod}: {qty}")

cluster_product_df = pd.DataFrame(cluster_product_freq)
cluster_product_df.to_csv(os.path.join(REPORTS_DIR, "cluster_top_products.csv"))

summary_stats = pd.DataFrame({
    "Metric": ["Total Customers", "Total Transactions", "Date Range (days)", "Avg Order Value", "Total Revenue"],
    "Value": [
        len(output),
        df["InvoiceNo"].nunique(),
        (max_date - df["InvoiceDate"].min()).days,
        f"${df['TotalPrice'].mean():.2f}",
        f"${df['TotalPrice'].sum():,.2f}"
    ]
})
summary_stats.to_csv(os.path.join(REPORTS_DIR, "summary_statistics.csv"), index=False)

print("\n=== FINAL SUMMARY ===")
print(f"Total Customers: {len(output)}")
print(f"K-Means Clusters: {best_k}")
print(f"DBSCAN Clusters: {clusters_db}")
print(f"Hierarchical Clusters: {best_hc_k}")
print("\nMetrics:")
print(metrics.to_string())

print("\nDone. All outputs saved in:", REPORTS_DIR)
print("Figures saved in:", FIG_DIR)
print("Models saved in:", MODEL_DIR)
