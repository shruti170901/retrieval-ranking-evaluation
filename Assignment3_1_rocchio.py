import math
import os
import argparse
import json
import csv
import pandas as pd
import pickle
import numpy as np
from Assignment2_1_evaluator import AveragePrecision_k, NDCG_k

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_path', metavar='data_path', nargs='?', default='./Data/CORD-19', type=str,
                    help='path to the CORD-19 folder')
parser.add_argument('model_q_path', metavar='model_q_path', nargs='?', default='./model_queries_1.bin', type=str,
                    help='path to model_queries_1.bin')
parser.add_argument('qrel_path', metavar='qrel_path', nargs='?', default='./Data/qrels.csv', type=str,
                    help='path to the gold standard ranked_list.csv')
parser.add_argument('alist_path', metavar='alist_path', nargs='?', default='./Assignment2_1_ranked_list_A.csv', type=str,
                    help='path to the ranked_list_A.csv')
args = parser.parse_args()

# loading rank list generated in assignment 2A
cord_df = pd.read_csv(args.alist_path)
unique_cord_ids = list(set(cord_df["cord_id"]))
print(len(unique_cord_ids))

# loading query frequency vectors
with open('./query_tf.bin', 'rb') as pkl_handle:
    query_tf = pickle.load(pkl_handle)

# loading document frequency vectors
with open('./doc_tf.bin', 'rb') as pkl_handle:
    doc_tf = pickle.load(pkl_handle)

# loading mapping from index to the document frequency of the word corresponding to the index
with open('./id_df_map.bin', 'rb') as pkl_handle:
    id_df_map = pickle.load(pkl_handle)

# no. of queries
Q = 35

# size of the vocabulary
N = len(doc_tf.keys())

################## Calculating the tf-idf vectors for the queries #################
# lnc.ltc
for k in query_tf.keys():
    sqred_sum = 0
    for i in range(len(query_tf[k])):
        if query_tf[k][i] != 0:
            query_tf[k][i] = 1+math.log10(query_tf[k][i])
        query_tf[k][i] *= math.log10(N/id_df_map[i])
        sqred_sum += query_tf[k][i]*query_tf[k][i]
    for i in range(len(query_tf[k])):
        query_tf[k][i] /= sqred_sum

################## Calculating the tf-idf vectors for the documents in the ranked list #################
# lnc.ltc
for k in unique_cord_ids:
    sqred_sum = 0
    for i in range(len(doc_tf[k])):
        if doc_tf[k][i] != 0:
            doc_tf[k][i] = 1+math.log10(doc_tf[k][i])
        sqred_sum += doc_tf[k][i]*doc_tf[k][i]
    for i in range(len(doc_tf[k])):
        doc_tf[k][i] /= sqred_sum

# Returns first 20 ranked documents according to gold standards
# Used for ideal DCG calculation
def ideal_ranked_docs(q):
    cord_qrel_df = pd.read_csv(args.qrel_path)
    rel_doc = cord_qrel_df.loc[cord_qrel_df['topic-id'] == (q+1)]['judgement'].tolist()
    rel_doc.sort(reverse=True)
    
    return rel_doc[:20]

# Returns actual relevance scores for top 20 documents of rank list (from part 2A)
def pick_ranked_relevant_docs(doc_list, q, pseudo=False):
    cord_qrel_df = pd.read_csv(args.qrel_path, index_col=['topic-id', 'cord-id'])
    # doc_list SHOULD BE SORTED AS PER RELEVANCE
    if pseudo == True:
        return [1] * 10 + [0] * (len(doc_list) - 10)
    rel_doc = []
    for doc in doc_list:
        if (q + 1, doc) in cord_qrel_df.index:
            rel_doc.append(cord_qrel_df.loc[(q + 1, doc)]['judgement'])
        else:
            rel_doc.append(0)
    return rel_doc

# Returns binary relevance scores for top 20 documents of rank list (from part 2A)
def pick_relevant_docs(doc_list, q, pseudo=False):
    cord_qrel_df = pd.read_csv(args.qrel_path, index_col=['topic-id', 'cord-id'])
    # doc_list SHOULD BE SORTED AS PER RELEVANCE
    if pseudo == True:
        return [1] * 10 + [0] * (len(doc_list) - 10)
    rel_doc = []
    for doc in doc_list:
        if (q + 1, doc) in cord_qrel_df.index:
            if cord_qrel_df.loc[(q + 1, doc)]['judgement'] == 2:
                rel_doc.append(1)
            else:
                rel_doc.append(0)
        else:
            rel_doc.append(0)
    return rel_doc

# Calculating modified query vector for rochio's algorithm
def rocchio(q0, doc_list, a, b, c, pseudo = False):
    # Here a = alpha, b = beta, c = gamma
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)

    q_modified = np.asarray(query_tf[q0]) * a
    # print(q_modified)
    rel_doc = pick_relevant_docs(doc_list, q0, pseudo)
    # 0.001 is added to avoid division by 0
    b /= (rel_doc.count(1) + 0.001)
    c /= (rel_doc.count(0) + 0.001)
    for i in range(len(rel_doc)):
        r = rel_doc[i]
        if r == 1:
            q_modified += b * np.asarray(doc_tf[doc_list[i]])
        else:
            q_modified -= c * np.asarray(doc_tf[doc_list[i]])
    return q_modified

# Sorts the document list based cosine similarity with the query vector
def rank_docs(q_vec, doc_list):
    sim = {}
    for cord_id in doc_list:
        cos_sim = np.dot(q_vec, doc_tf[cord_id])
        sim[cord_id] = cos_sim

    return sorted(doc_list, key=lambda cord_id: sim[cord_id], reverse=True)

def main():
    rel_dict = {}
    ap_scores1, ap_scores2, ap_scores3, ap_scores4, ap_scores5, ap_scores6 = [], [], [], [], [], []
    ndcg_scores1, ndcg_scores2, ndcg_scores3, ndcg_scores4, ndcg_scores5, ndcg_scores6 = [], [], [], [], [], []


    for q in range(Q):
        relevance_qrel_ranked = ideal_ranked_docs(q)
        rel_dict[q] = []
        ii = 0
        # picking top 20 cord id for a query q and storing it in rel_dict
        while len(rel_dict[q]) < 20 and ii < 50:
            if cord_df.loc[50 * q + ii]['cord_id'] in doc_tf.keys():
                rel_dict[q].append(cord_df.loc[50 * q + ii]['cord_id'])
            ii += 1

        #####################   RELEVANCE FEEDBACK  ##########################
        ################ alpha = 1, beta = 1, gamma = 0.5  ###################
        q_modified1 = rocchio(q, rel_dict[q], 1, 1, 0.5)
        ranked_docs = rank_docs(q_modified1, rel_dict[q])
        relevancy_scores = pick_relevant_docs(ranked_docs, q)
        ap = AveragePrecision_k(relevancy_scores, 20)
        relevancy_scores = pick_ranked_relevant_docs(ranked_docs,q)
        ndcg_scores1.append(NDCG_k(relevancy_scores, relevance_qrel_ranked, 20))
        ap_scores1.append(ap)

        ################ alpha = 0.5, beta = 0.5, gamma = 0.5  ###############
        q_modified2 = rocchio(q, rel_dict[q], 0.5, 0.5, 0.5)
        ranked_docs = rank_docs(q_modified2, rel_dict[q])
        relevancy_scores = pick_relevant_docs(ranked_docs, q)
        ap = AveragePrecision_k(relevancy_scores, 20)
        relevancy_scores = pick_ranked_relevant_docs(ranked_docs,q)
        ndcg_scores2.append(NDCG_k(relevancy_scores, relevance_qrel_ranked, 20))
        ap_scores2.append(ap)

        ################ alpha = 1, beta = 0.5, gamma = 0  ###################
        q_modified3 = rocchio(q, rel_dict[q], 1, 0.5, 0)
        ranked_docs = rank_docs(q_modified3, rel_dict[q])
        relevancy_scores = pick_relevant_docs(ranked_docs, q)
        ap = AveragePrecision_k(relevancy_scores, 20)
        relevancy_scores = pick_ranked_relevant_docs(ranked_docs,q)
        ndcg_scores3.append(NDCG_k(relevancy_scores, relevance_qrel_ranked, 20))
        ap_scores3.append(ap)
        
        ################### PSEUDO  RELEVANCE FEEDBACK  ######################
        ################ alpha = 1, beta = 1, gamma = 0.5  ###################
        q_modified4 = rocchio(q, rel_dict[q], 1, 1, 0.5, pseudo=True)
        ranked_docs = rank_docs(q_modified4, rel_dict[q])
        relevancy_scores = pick_relevant_docs(ranked_docs, q)
        ap = AveragePrecision_k(relevancy_scores, 20)
        relevancy_scores = pick_ranked_relevant_docs(ranked_docs,q)
        ndcg_scores4.append(NDCG_k(relevancy_scores, relevance_qrel_ranked, 20))
        ap_scores4.append(ap)

        ################ alpha = 0.5, beta = 0.5, gamma = 0.5  ###############
        q_modified5 = rocchio(q, rel_dict[q], 0.5, 0.5, 0.5, pseudo=True)
        ranked_docs = rank_docs(q_modified5, rel_dict[q])
        relevancy_scores = pick_relevant_docs(ranked_docs, q)
        ap = AveragePrecision_k(relevancy_scores, 20)
        relevancy_scores = pick_ranked_relevant_docs(ranked_docs,q)
        ndcg_scores5.append(NDCG_k(relevancy_scores, relevance_qrel_ranked, 20))
        ap_scores5.append(ap)

        ################ alpha = 1, beta = 0.5, gamma = 0  ###################
        q_modified6 = rocchio(q, rel_dict[q], 1, 0.5, 0, pseudo=True)
        ranked_docs = rank_docs(q_modified6, rel_dict[q])
        relevancy_scores = pick_relevant_docs(ranked_docs, q)
        ap = AveragePrecision_k(relevancy_scores, 20)
        relevancy_scores = pick_ranked_relevant_docs(ranked_docs,q)
        ndcg_scores6.append(NDCG_k(relevancy_scores, relevance_qrel_ranked, 20))
        ap_scores6.append(ap)

    # Saving the RF results in csv file
    with open('./Assignment3_1_rocchio_RF_metrics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "beta", "gamma", "mAP@20", "NDCG@20"]) 
        writer.writerow([1, 1, 0.5, sum(ap_scores1)/len(ap_scores1), sum(ndcg_scores1)/len(ndcg_scores1)])
        writer.writerow([0.5, 0.5, 0.5, sum(ap_scores2)/len(ap_scores2), sum(ndcg_scores2)/len(ndcg_scores2)])
        writer.writerow([1, 0.5, 0, sum(ap_scores3)/len(ap_scores3), sum(ndcg_scores3)/len(ndcg_scores3)])

    # Saving the PsRF results in csv file
    with open('./Assignment3_1_rocchio_PsRF_metrics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "beta", "gamma", "mAP@20", "NDCG@20"]) 
        writer.writerow([1, 1, 0.5, sum(ap_scores4)/len(ap_scores4), sum(ndcg_scores4)/len(ndcg_scores4)])
        writer.writerow([0.5, 0.5, 0.5, sum(ap_scores5)/len(ap_scores5), sum(ndcg_scores5)/len(ndcg_scores5)])
        writer.writerow([1, 0.5, 0, sum(ap_scores6)/len(ap_scores6), sum(ndcg_scores6)/len(ndcg_scores6)])

if __name__ == '__main__':
    main()
