"""
AUTHOR: Tong Wang, CENTER FOR COMPUTATIONAL AND INTEGRATIVE BIOLOGY
                 Shreya Patel, COMPUTER SCIENCE
DATE: 12/2021
DESCRIPTION: This script is a part imported as package for MBA_EDA_Features.py. The following is the LSH code.
USAGE(In Macbook/Amarel): import as package for MBA_EDA_Features.py
"""
import os
import numpy as np
import pandas as pd
import math
import random
import hashlib


""" Use the original id-feature dataset (tfidf_up_df) to do the feature compression, and futher cosine-LSH"""

def getMd5Hash(band):
    hashobj=hashlib.md5()
    hashobj.update(band.encode())
    hashValue = hashobj.hexdigest()
    return hashValue

def cos_lsh(feature_matrix, r, b):
    sig_matrix_size = r*b
    id_num = feature_matrix.shape[0]
    feature_num = feature_matrix.shape[1]    
    """
    get the random vector matrix, and further generate the signature matrix
    """
    random_vector_matrix = np.random.randn(feature_num, sig_matrix_size) 
    compression = np.dot(feature_matrix, random_vector_matrix)
    sig_matrix = np.where(compression>0, 1, 0).T
#     return sig_matrix
    """
    1. parameters:
    hashbucket: key: hashvalue; value: possible candidate
    each hash function start at 0, r, 2r...; end at r, 2r, 3r ...; the same hash function use for the same band
    b_index: band index (start at 1)
    
    2. logic:
    while signature_matrix rows number >= end:
        for each column:
            get the each id's sigature vector piece, make it into string and combine with band_index
            make such combination into a md5 hash value -> as a label for such vector piece
            if such hash value (label) not in the bucket:
                add the column to the corresponding key (label)
            elif hash value in the bucket, but column not in the bucket of the corresponding hash value
                add the column to it
        till signature_matrix rows number < end
    return hash bucket
    """
    hashBuckets={}
    start=0
    end=r
    b_index = 1    
    while sig_matrix.shape[0] >= end:
        for col_num in range(sig_matrix.shape[1]):
            band = str(sig_matrix[start:end, col_num])+str(b_index)
            hashValue=getMd5Hash(band)
            if hashValue not in hashBuckets:
                hashBuckets[hashValue] = [feature_matrix.index[col_num]]
            elif feature_matrix.index[col_num] not in hashBuckets[hashValue]:
                hashBuckets[hashValue].append(feature_matrix.index[col_num])
        start += r
        end += r
        b_index += 1
    return hashBuckets

