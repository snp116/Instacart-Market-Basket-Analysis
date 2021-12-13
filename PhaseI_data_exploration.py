#!/usr/bin/env python
# coding: utf-8
from zipfile import ZipFile 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
from IPython.display import display, HTML
import os

os.chdir(os.getcwd())

data_dir = os.path.join('.', 'data')
output_data_dir = os.path.join('.', 'output_data')

train_file = os.path.join(data_dir, 'order_products__train.csv')
order_file = os.path.join(data_dir, 'orders.csv')
dept_file = os.path.join(data_dir, 'departments.csv')
ailes_file = os.path.join(data_dir, 'aisles.csv')
pdt_file = os.path.join(data_dir, 'products.csv')
prior_file = os.path.join(data_dir, 'order_products__prior.csv')

train_df = pd.read_csv(train_file)
order_df = pd.read_csv(order_file)
dept_df = pd.read_csv(dept_file)
aisle_df = pd.read_csv(ailes_file)
pdt_df = pd.read_csv(pdt_file)
prior_df = pd.read_csv(prior_file)

# print("This is the dimensionality of the order's DataFrame")
# print(order_df.shape)

# print("This is the dimentionality of the product's DataFrame")
# print(pdt_df.shape)

# print("This is sample of the trained order_products DataFrame")
# print(train_df.head())

# print("This is sample of orders.csv")
# print(order_df.head(10))

# print("This is sample of departments.csv")
# print(dept_df.head())

# print("This is sample of aisles.csv")
# print(aisle_df.head())

# print("This is sample of products.csv")
# print(pdt_df.head())


##distribution to show the total orders for each customer
sns.set_style('darkgrid')
con = order_df.groupby("user_id", as_index = False)["order_number"].max()
n , bins, patches = plt.hist(con["order_number"], 15, density = True,   stacked = True,color = 'orange', alpha = 0.6)
# plt.ylabel('y-axis', fontsize=10)
# plt.xlabel('x-axis', fontsize=10)
plt.title("distribution to show the total orders for each customer")
plt.savefig(output_data_dir+'Fig1.pdf')
plt.show(block=True)
#Since the histogram appears to be skewed, its appropriate distribution is expected to be an exponential function

n, bins, patches = plt.hist(con["order_number"],15,density = True, facecolor = 'green',alpha = 0.6)
#We need to make the values of the two axes equal to proceed
bins = np.delete(bins,10)
bins = bins+5

def expo_fn(a,u,v,w):
    return u*np.exp(-v*a) + w

popt, pcov = curve_fit(expo_fn,bins,n,p0=(1,1e-6,1))

x = np.linspace(8,100,30)
y = expo_fn(x,*popt)

plt.plot(x,y,'r--')
plt.xlabel("Number_Orders")
plt.ylabel("Count_Order")
plt.title("Distribution of Order Number vs Customers")
plt.savefig(output_data_dir+'/Fig2.pdf')
plt.show(block=True)

train2_df = train_df.append(prior_df,ignore_index = True)
##Product count dataset based on order id
pdtcnt_df = train2_df.groupby("product_id",as_index= False)["order_id"].count()

pdtcnt_df.head()

pdtcnt_df.shape

##sort df based on order id
pdtcnt2_df = pdtcnt_df.sort_values("order_id", ascending = False)

#top 10 frequent products bought
print('top 10 frequent products bought')
prod_ten = pdtcnt2_df.iloc[0:11,:]
prod_ten = prod_ten.merge(pdt_df, on = "product_id")

display(prod_ten.loc[:,["product_name"]])

sns.set_style('darkgrid')
pdtcnt2_df["density"] = (pdtcnt2_df["order_id"]/np.sum(pdtcnt2_df["order_id"]))
pdtcnt2_df["rank"] = range(pdtcnt2_df.shape[0])
plt.plot(pdtcnt2_df["rank"],pdtcnt2_df["density"], color="r")
plt.title("Product count density plot")
plt.savefig(output_data_dir+'/Fig3.pdf')
plt.xlabel("rank")
plt.ylabel("density")
plt.show(block=True)

##avoid log(0) by adding 1
sns.set_style('darkgrid')
pdtcnt2_df["logRank"] = np.log(pdtcnt2_df["rank"] + 1) 
plt.title("product counts(density plot)")
plt.savefig(output_data_dir+'/Fig4.pdf')
plt.xlabel("$\log(Rank)$")
plt.ylabel("Density")
plt.plot(pdtcnt2_df["logRank"],pdtcnt2_df["density"], color="r")
plt.show(block=True)

# Distribution seems to be very steep. Hence we can perform smoothening on the sparse distribuition area.
# e^6 = 403 products define most of the distribution. Products lying under (e^6, e^12) range are not significant since their respective density is very less.

##Days of orders in a week
grouped = order_df.groupby("order_id")["order_dow"].aggregate("sum").reset_index()
grouped = grouped.order_dow.value_counts()
bars = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')

# Create bars with blue edge color
plt.bar(grouped.index, grouped.values,  edgecolor='blue')

# Create names on the x-axis
plt.xticks(grouped.index, bars)
# Show graph
# plt.show()
plt.savefig(output_data_dir+'/Fig5.pdf')
plt.ylabel('Number of orders', fontsize=10)
plt.xlabel('Days of order in a week', fontsize=10)
plt.show()

print(len(set(order_df.user_id)))