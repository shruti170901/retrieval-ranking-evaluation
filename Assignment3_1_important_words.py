import math
import os
import argparse
import json
import csv
import pandas as pd
import pickle
import numpy as np

# argument parser to read all the file names given as input from the user
'''
    data_path: stores path to the CORD-19 folder
    model_q_path: stores path to model_queries_1.bin
    alist_path: stores path to the Assignment2_1_ranked_list_A.csv
'''
parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='data_path', nargs='?', default='./Data/CORD-19', type=str,
                    help='path to the CORD-19 folder')
parser.add_argument('model_q_path', metavar='model_q_path', nargs='?', default='./model_queries_1.bin', type=str,
                    help='path to model_queries_1.bin')
parser.add_argument('alist_path', metavar='alist_path', nargs='?', default='./Assignment2_1_ranked_list_A.csv', type=str,
                    help='path to the ranked_list_A.csv')
args = parser.parse_args()

# read Assignment2_1_ranked_list_A.csv into cord_df
cord_df = pd.read_csv(args.alist_path)
# verify that cord_df has expected number of cord_ids
unique_cord_ids = list(set(cord_df["cord_id"]))
print(len(unique_cord_ids))
# load query term frequency vectors
with open('./query_tf.bin', 'rb') as pkl_handle:
    query_tf = pickle.load(pkl_handle)
# load document term frequency vectors
with open('./doc_tf.bin', 'rb') as pkl_handle:
    doc_tf = pickle.load(pkl_handle)
# load mapping to document frequency for all terms
with open('./id_df_map.bin', 'rb') as pkl_handle:
    id_df_map = pickle.load(pkl_handle)
# number of queries
Q = 35
# number of documents
N = len(doc_tf.keys())

############################ Calculating the tf-idf vectors for the queries ############################
for k in query_tf.keys():
    # for normalizing we calculate the squared sum simultaneously
    sqred_sum = 0
    # compute TF-IDF vectors for each query
    for i in range(len(query_tf[k])):
        if query_tf[k][i] != 0:
            query_tf[k][i] = 1+math.log10(query_tf[k][i])
        query_tf[k][i] *= math.log10(N/id_df_map[i])
        sqred_sum += query_tf[k][i]*query_tf[k][i]
    # normalize the vectors
    for i in range(len(query_tf[k])):
        query_tf[k][i] /= sqred_sum

################## Calculating the tf-idf vectors for the documents in the ranked list #################
for k in unique_cord_ids:
    # for normalizing we calculate the squared sum simultaneously
    sqred_sum = 0
    # compute TF-IDF vectors for each document
    for i in range(len(doc_tf[k])):
        if doc_tf[k][i] != 0:
            doc_tf[k][i] = 1+math.log10(doc_tf[k][i])
        # else it is 0
        sqred_sum += doc_tf[k][i]*doc_tf[k][i]
    # normalize the vectors
    for i in range(len(doc_tf[k])):
        doc_tf[k][i] /= sqred_sum

# load mapping to indices of vocabulary to their corresponding terms
with open('./idx_to_term.bin', 'rb') as pkl_handle:
    idx_to_term = pickle.load(pkl_handle)

# read Assignment2_1_ranked_list_A.csv into alist_df
alist_df = pd.read_csv(args.alist_path)
# open csv file to write the important words for each query
with open('./Assignment3_1_important_words.csv', 'w') as f:
    writer = csv.writer(f)
    # adding the header of the csv file
    writer.writerow(["query", "word1, word2, word3, word4, word5"]) 
    # for all queries computing the most important words
    for q in range(Q):
        # initializing the average TF-IDF vector with all zeros
        average_doc_vec = np.array([0]*len(query_tf[q]))
        # obtaining top 10 documents from the ranked list
        ranked_docs = alist_df.loc[alist_df['q_id'] == q]['cord_id'].tolist()
        ranked_docs = ranked_docs[:10]
        # adding all the TF-IDF vectors
        for doc in ranked_docs:
            average_doc_vec = average_doc_vec + np.array(doc_tf[doc])
        # averaging the sum of all TF-IDF vectors to obtain average
        average_doc_vec /= len(ranked_docs)
        # sorting the terms based on the reverse similarity vales 
        sorted_indices = sorted(list(range(20000)), key=lambda idx: average_doc_vec[idx], reverse=True)
        # obtaining the terms corresponding to the indices
        important_terms = [idx_to_term[i] for i in sorted_indices[:5]]
        # writing all the important words for the query
        writer.writerow([q, ', '.join(important_terms)]) 