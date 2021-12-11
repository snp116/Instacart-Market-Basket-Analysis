"""
AUTHOR: Tong Wang, CENTER FOR COMPUTATIONAL AND INTEGRATIVE BIOLOGY
DATE: 12/2021
DESCRIPTION: This script is the third part used for Instacart-Market-Basket-analysis. Here we focus on KMeans analysis
                           of the users behavor.
USAGE(In Macbook/Amarel): (please run MBA_EDA_Features.py and MBA_Optimal_K.py at first to get the Optimal_K)
             1. Execute the script:
                 python MBA_KMeans.py Integer
               Note (Commands help):
                 python MBA_KMeans.py -h
             2. Deactivate the environment:
                 conda deactivate
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import argparse
import seaborn as sns

"""0. path and dictionary setting"""

parser = argparse.ArgumentParser(description='python MBA_KMeans.py integer')
parser.add_argument('integer', type=int, nargs='+',
                   help='the Optimal K for KMeans (should be equal to/larger than 4)')
args = parser.parse_args()

output_data_dir = os.path.join('.', 'output_data/')
standard_user_features_file = os.path.join(output_data_dir, 'decomposited_features_for_KMeans.csv')
standard_user_features_df = pd.read_csv(standard_user_features_file, index_col=["user_id"])
fig_dir = os.path.join('.', 'fig/')
output_data_dir = os.path.join('.', 'output_data/')

"""1. dataset preprocessing"""

data_dir = os.path.join('.', 'data')
order_file = os.path.join(data_dir, 'orders.csv')
order_prior_file = os.path.join(data_dir, "order_products__prior.csv")
product_file = os.path.join(data_dir, "products.csv")
order_df = pd.read_csv(order_file)
order_prior_df = pd.read_csv(order_prior_file)
product_df = pd.read_csv(product_file, index_col="product_id", dtype={'product_name': pd.StringDtype()})
prior_order_product = order_prior_df[order_prior_df["reordered"]==1] # reordered == 1 -> True; 0-> False
user_order_prior = pd.merge(prior_order_product, order_df, on="order_id", how="left")
user_order_prior = pd.merge(user_order_prior, product_df, on="product_id", how="left")

"""2. use sklearn KMeans method to create the user cluster model"""

## 2.1 cluster centroids fig
kmeans = KMeans(n_clusters=args.integer[0]).fit(standard_user_features_df)
centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(12,8))
x_coordinates = centroids[:, 0]
y_coordinates = centroids[:, 1]
for x, y in zip(x_coordinates, y_coordinates):
    rgb = (random.random(), random.random(), random.random())
    plt.scatter(x, y, c=[rgb])
plt.title("Centroids_K-Means_Clusters")
fig.savefig(fig_dir + 'Centroids_K-Means_Clusters.tif', format='tif')
# plt.show()

## 2.2 all user cluster fig
kmeans_res = standard_user_features_df
kmeans_res["cluster_id"] = kmeans.labels_
kmeans_res.head()
fig = plt.figure(figsize=(14,10))
# Unique category labels for clusters_id
color_labels = kmeans_res['cluster_id'].unique()
# prepare the color_palette
rgb_values = sns.color_palette("hls", 40)
# Map the label of clusters_id to RGB
color_map = dict(zip(color_labels, rgb_values))
# plot
plt.scatter(kmeans_res.iloc[:, 0], kmeans_res.iloc[:, 1], c=kmeans_res['cluster_id'].map(color_map))
plt.title("All_users_for_K-Means_Clusters")
fig.savefig(fig_dir + 'All_users_for_K-Means_Clusters.tif', format='tif')
# plt.show()

## 2.3 analyze the cluster results
cluster_order_info = pd.merge(kmeans_res, user_order_prior, how='left', on='user_id')
cluster_order_info.head()
cluster_product = cluster_order_info[['user_id','cluster_id','product_name']]
cluster_product

# the frequency of products in each cluster
cluster_count = cluster_product.groupby(['cluster_id','product_name']).agg('count')
# cluster_count
# top10 products in each cluster
top_products = cluster_count['user_id'].groupby(level=0, group_keys=False).nlargest(10).reset_index()
# top_products
all_clusters_top_products =top_products.pivot(index='cluster_id', columns='product_name', values='user_id').fillna(0)
all_clusters_top_products_percent = all_clusters_top_products.div(all_clusters_top_products.sum(axis=0), axis=1)
all_clusters_top_products.to_csv(output_data_dir+"top10_products_in_each_cluster.csv")
all_clusters_top_products_percent.to_csv(output_data_dir+"the_percent_of_each_cluster_top10_products_across_all_clusters.csv")

def plot_cluster_top_prod(cluster_l, cluster_right):
    fig, ax = plt.subplots(1,figsize=(12,8),sharey=True)
    group = []
    for i in range(cluster_l, cluster_right+1):
        group.append(i)
    ax.plot(all_clusters_top_products_percent.loc[group].transpose())
    ax.legend(all_clusters_top_products_percent.transpose().columns[cluster_l: cluster_right+1],
               title="Cluster_ID",loc='upper left',prop={'size': 12})
    ax.set_title('Percent_of_Products_from_Cluster{}_to_Cluster{}'.format(cluster_l, cluster_right),size=20)    
    plt.sca(ax)
    plt.xticks(rotation=90, size=12)
    plt.subplots_adjust(wspace=0, hspace=0.7)
    fig.savefig(fig_dir + 'Percent_of_Products_from_Cluster{}_to_Cluster{}.tif'.format(cluster_l, cluster_right), format='tif', bbox_inches='tight')

fig_num = args.integer[0] // 4
plot_cluster_top_prod(0, fig_num-1)
plot_cluster_top_prod(fig_num, fig_num*2-1)
plot_cluster_top_prod(fig_num*2, fig_num*3-1)
plot_cluster_top_prod(fig_num*3, args.integer[0]-1)
