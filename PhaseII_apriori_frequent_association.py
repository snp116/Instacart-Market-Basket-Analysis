#!/usr/bin/env python
# coding: utf-8
from zipfile import ZipFile 
import itertools
import collections
import os
import pandas as pd
import numpy as np
import sys

data_dir = os.path.join('.', 'data')
output_data_dir = os.path.join('.', 'output_data')

order_file = os.path.join(data_dir, 'order_products__prior.csv')
product_file = os.path.join(data_dir, 'products.csv')

#using products.csv to get names of product_id
#using these order_products_prior to get all transactionsactions and items
transactions = pd.read_csv(order_file)
item_name   = pd.read_csv(product_file)

print("Sample dataset : ")
print(transactions.head())

print(f"The total number of transactions are {transactions.shape[0]}")

print("Sample dataset : ")
print(transactions.head())

# Making itemset, with order_id as index and item_id as value
transactions = transactions.set_index('order_id')['product_id'].rename('item_id')

def item_frequency(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("item_frequency")
    else: 
        return pd.Series(collections.Counter(iterable)).rename("item_frequency")

def get_AB_items(tran_item):
    tran_item = tran_item.reset_index().to_numpy()
    for tran_id, tran_object in itertools.groupby(tran_item, lambda x: x[0]):
        item_list = [item[1] for item in tran_object]      
        for item_pair in itertools.combinations(item_list, 2):
            yield item_pair

def merge_stats(AB_items, stats):
    return (AB_items
                .merge(stats.rename(columns={'item_frequency': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(stats.rename(columns={'item_frequency': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))            

def association_rules(tran_item, min_support):
#SUPPORT
    stats = item_frequency(tran_item).to_frame("item_frequency")
    stats['support'] = stats['item_frequency'] / len(set(tran_item.index)) * 100

    qualifying_items = stats[stats['support'] >= min_support].index
    tran_item = tran_item[tran_item.isin(qualifying_items)]
    qualifying_transactions = item_frequency(tran_item.index)[item_frequency(tran_item.index) >= 2].index
    tran_item = tran_item[tran_item.index.isin(qualifying_transactions)]

    stats = item_frequency(tran_item).to_frame("item_frequency")
    stats['support'] = stats['item_frequency'] / len(set(tran_item.index)) * 100
    
    #support for corresponding item-AB_items frequency
    AB_items = item_frequency(get_AB_items(tran_item)).to_frame("freqAB")
    AB_items['supportAB'] = AB_items['freqAB'] / len(qualifying_transactions) * 100

    AB_items = AB_items[AB_items['supportAB'] >= min_support]
    AB_items = AB_items.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    AB_items = merge_stats(AB_items, stats)
    
#CONFIDENCE
    AB_items['confidenceAtoB'] = AB_items['supportAB'] / AB_items['supportA']
    AB_items['confidenceBtoA'] = AB_items['supportAB'] / AB_items['supportB']
 
 #LIFT   
    AB_items['lift'] = AB_items['supportAB'] / (AB_items['supportA'] * AB_items['supportB'])
    
    return AB_items.sort_values('lift', ascending=False)

rules = association_rules(transactions, 0.02)

item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
rules = (rules
    .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
    .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))

rules_final = rules[columns].sort_values('lift', ascending=False)
final_result = rules_final[['itemA', 'itemB', 'freqAB', 'supportAB', 'confidenceAtoB', 'confidenceBtoA', 'lift']]

final_result.to_csv(output_data_dir+'/final_result.csv', index=False)
print('Process complete')
