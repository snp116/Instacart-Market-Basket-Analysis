"""
AUTHOR: Tong Wang, CENTER FOR COMPUTATIONAL AND INTEGRATIVE BIOLOGY
DATE: 12/2021
DESCRIPTION: This script is the second part used for Instacart-Market-Basket-analysis. Here we focus on sreening
                           the optimal K value for KMeans.
USAGE(In Macbook/Amarel): (please run MBA_EDA_Features.py at first to get the decomposited features)
             1. Execute the script:
                 python MBA_Optimal_K.py Integer1 Integer2 Integer3
             Note (Commands help):
                 python MBA_Optimal_K.py -h
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

"""0. path and dictionary setting"""

parser = argparse.ArgumentParser(description='python MBA_Optimal_K.py integer1 integer2 integer3')
parser.add_argument('integers', type=int, nargs='+',
                   help='three integers for the screening of optimal K in range(left_int, right_int, step_int)')
args = parser.parse_args()

"""1. find the optimal k value for KMeans study by Elbow Method"""

output_data_dir = os.path.join('.', 'output_data/')
standard_user_features_file = os.path.join(output_data_dir, 'decomposited_features_for_KMeans.csv')
standard_user_features_df = pd.read_csv(standard_user_features_file, index_col=["user_id"])
fig_dir = os.path.join('.', 'fig/')

Sum_of_squared_distances = []
K = range(args.integers[0], args.integers[1], args.integers[2])
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(standard_user_features_df)
    Sum_of_squared_distances.append(km.inertia_)

fig = plt.figure(figsize=(12,8))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('The_number_of_k')
plt.axvline(40, color='darkorange', linestyle='dashed', linewidth=2)
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow_Method_For_Optimal_k')
fig.savefig(fig_dir + 'Elbow_Method_For_Optimal_k.tif', format='tif')
# plt.show()
