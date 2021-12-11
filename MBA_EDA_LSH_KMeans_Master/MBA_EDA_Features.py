"""
AUTHOR: Tong Wang, CENTER FOR COMPUTATIONAL AND INTEGRATIVE BIOLOGY
DATE: 12/2021
DESCRIPTION: This script is the first part used for Instacart-Market-Basket-analysis. Here we focus on the EDA,
                           tfidf features generation, and LSH analysis.
USAGE(In Macbook/Amarel): (please install anaconda 4.10.3 at first, the python version is 3.9.7)
             0. change path to the work directory
             1. Creating an environment with commands (note: replace myenv with the environment name): 
                 conda create --name myenv
             2. Activate the environment with commands:
                 conda activate myenv
             3. Install packages in conda environment using requirements.txt:
                 conda install --file requirements.txt
             4. Execute the script:
                 python MBA_EDA_Features.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler
from IPython.display import display
import math
import random
import hashlib
import MBA_LSH


"""0. path and dictionary setting"""

data_dir = os.path.join('.', 'data')
order_file = os.path.join(data_dir, 'orders.csv')
order_prior_file = os.path.join(data_dir, "order_products__prior.csv")
product_file = os.path.join(data_dir, "products.csv")

if not os.path.exists(os.path.join('.', 'fig')):
    os.makedirs(os.path.join('.', 'fig'))
if not os.path.exists(os.path.join('.', 'output_data')):
    os.makedirs(os.path.join('.', 'output_data'))
fig_dir = os.path.join('.', 'fig/')
output_data_dir = os.path.join('.', 'output_data/')


"""1. load data and analysis the data characteristics - construct the dataset"""

## 1.1 general analysis of the data


order_df = pd.read_csv(order_file, dtype={"order_id":np.float32, "user_id":np.float32,"order_number":np.float32,"order_dow":np.float32,"order_hour_of_day":np.float32,"days_since_prior_order":np.float32})
# display(order_df.head())
order_df.loc[order_df["order_number"] == int(0), "order_number"] = np.nan
print('\033[1m'+"*"*100+'\033[0m')
print('\033[1m'+"order_data(missing_part):\n"+'\033[0m', order_df.isna().sum(), sep='')
print('\033[1m'+"*"*100+'\033[0m')

fig, axarr = plt.subplots(1, 2, figsize=(12, 8))
msno.matrix(order_df.sort_values("user_id"), ax=axarr[0], sparkline=False, color=([0, 128/255, 1]))
axarr[0].set_xlabel("order_data: missing value matrix sorted by user_id", fontsize=12)
msno.bar(order_df, ax=axarr[1], color=([0, 128/255, 1]))
axarr[1].set_xlabel("order_data: the proportion and number of non-missing values", fontsize=12)
fig.savefig(fig_dir + 'dataset_shape_and_missing_value.tif', format='tif', bbox_inches='tight')
# plt.show()

## 1.2 compare the number of orders and the number of customers

# the number of customers placed corresponding number of orders
order_per_cust_df = order_df.groupby("user_id")["order_number"].max().reset_index()
order_per_cust_df["order_number"] = order_per_cust_df["order_number"].astype(int)
# order_per_cust_df["order_number"].value_counts()

fig = plt.figure(figsize=(18,10))
sns.set_theme(style="darkgrid")
ax = sns.countplot(x = "order_number", data=order_per_cust_df)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
plt.title("Number_of_Orders_by_Each_Customer_VS_Number_of_Customers_Place_The_Orders")
plt.xlabel("Number_of_Orders_by_Each_Customer")
plt.ylabel("Number_of_Customers_Place_The_Orders")
fig.savefig(fig_dir + 'Orders_vs_Users.tif', format='tif', bbox_inches='tight')
# plt.show()

## 1.3 merge data together to get the ideal dataset for analysis

print('\033[1m'+"order_subdataset"+'\033[0m')
display(order_df.head())
print('\033[1m'+"order_products_prior_subdataset"+'\033[0m')
order_prior_df = pd.read_csv(order_prior_file, dtype={"order_id":np.float32,"product_id":np.float32,"add_to_cart_order":np.float32,"reordered":np.float32})
display(order_prior_df.head())
# change the type of product_name as str
print('\033[1m'+"products_subdataset"+'\033[0m')
product_df = pd.read_csv(product_file, index_col="product_id", dtype={"product_id":np.float32,'product_name': pd.StringDtype(),"aisle_id":np.float32,"department_id":np.float32})
display(product_df.head())
print('\033[1m'+"*"*100+'\033[0m')

# 1.3.1
# extract product being reordered from order_prior_df
prior_order_product = order_prior_df[order_prior_df["reordered"]==1] # reordered == 1 -> True; 0-> False

# 1.3.2
# merge order_subdataset and order_prior_subdataset together based on the order_id 
# (add product_id into the dataset)
user_order_prior = pd.merge(prior_order_product, order_df, on="order_id", how="left")

# 1.3.3
# futher add product_info into user_order_prior from products_subdataset based on the product_id
user_order_prior = pd.merge(user_order_prior, product_df, on="product_id", how="left")
# user_order_prior maps info such as product_id and product_name into coresponding orders(order_id) and users(user_id)
user_product_prior = user_order_prior[["user_id", "product_name"]]
# double check if the product_name is str, since we need to merge them based on the user_id
user_product_prior["product_name"].dtypes
# user_product_prior.head()

# 1.3.4 Merge prior product based on user_id (duplicates) to get the ideal dataset for analysis
user_product_prior = user_product_prior.groupby('user_id').agg({ 'product_name': ', '.join}).reset_index()
user_product_prior = user_product_prior.rename(columns={"user_id": "user_id", "product_name": "product_prior_set"})
# user_product_prior.head()


"""2 TF-IDF product_prior_set"""

tfidf_up = TfidfVectorizer(input='content', encoding='iso-8859-1', min_df=5, max_features=1000, sublinear_tf=True, 
                           decode_error='ignore', analyzer='word',
                           ngram_range=(1,1), stop_words='english')
# 1000 features generated from the reorder product sentences of each users
tfidf_up_model = tfidf_up.fit_transform(user_product_prior["product_prior_set"])
tfidf_up_df = pd.DataFrame(tfidf_up_model.toarray(), columns=tfidf_up.get_feature_names(), 
                           index = user_product_prior["user_id"], dtype=np.float32)
print('\033[1m'+"1000_tfidf_features"+'\033[0m')
display(tfidf_up_df.head())
print('\033[1m'+"*"*100+'\033[0m')
# corpus distribution with certain parameters such as min_df, max_features
fig = plt.figure(figsize=(12,8))
tfidf_hist = tfidf_up_df[tfidf_up_df>0].count(axis=0)
plt.hist(tfidf_hist, bins=50)
plt.title("corpus_distribution_with_certain_parameters")
fig.savefig(fig_dir + 'corpus_distribution_with_certain_parameters.tif', format='tif')
# plt.show()


""" 3 LSH for rough analysis"""

cos_bucket = MBA_LSH.cos_lsh(tfidf_up_df, 5, 5)  # r=5, b=5
# print(len(cos_bucket))
res=pd.DataFrame(cos_bucket.items(), columns = ["hash_values", "similar_user_sets"], dtype=np.float32)
print('\033[1m'+"LSH_similar_pairs_result_hashbucket"+'\033[0m')
display(res.head())
print('\033[1m'+"*"*100+'\033[0m')
res.to_csv(output_data_dir+"LSH_similar_pairs_result.csv")


""" 4 full 1005 features for KMeans"""

## 4.1 add additonal 5 features based on some user beheavor into the 1000 tfidf features
user_avg = user_order_prior.groupby('user_id')[['order_dow','order_hour_of_day', 'days_since_prior_order']].agg(np.nanmean)
order_num = user_order_prior.groupby("user_id").order_id.nunique()
product_num = user_order_prior.groupby("user_id")["product_id"].agg('count')
user_avg["order_num"] = order_num
user_avg["product_num"] = product_num
user_features = pd.merge(user_avg, tfidf_up_df, how='inner', on="user_id")
# user_features

## 4.2 standardize the 1005 features by StandardScaler
scaler = StandardScaler()
scaler.fit(user_features)
standard_user_features_data = scaler.transform(user_features)
standard_user_features = pd.DataFrame(standard_user_features_data, index = user_features.index, columns = user_features.columns, dtype=np.float32)
print('\033[1m'+"1000_tdidf_features_plus_5_user_behavor_features"+'\033[0m')
display(standard_user_features.head())
print('\033[1m'+"*"*100+'\033[0m')

## 4.3 decomposite the 1005 features into two principle components by PCA method
standard_user_features_pca = PCA(2).fit(standard_user_features.values)
reduced_standard_user_features = PCA(2).fit_transform(standard_user_features.values)
standard_user_features_df = pd.DataFrame(reduced_standard_user_features, index = standard_user_features.index, columns = ["PC1", "PC2"], dtype=np.float32)
print('\033[1m'+"decomposited_1005features_for_KMeans"+'\033[0m')
display(standard_user_features_df.head())
print('\033[1m'+"*"*100+'\033[0m')
standard_user_features_df.to_csv(output_data_dir+"decomposited_features_for_KMeans.csv")
