import math
import os
import argparse
import json
import csv
import nltk
import pandas as pd
import pickle 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle 
# Helper function to calcuate average_precision@k
def AveragePrecision_k(Relevancy_score_list,k):
    AP = 0
    relevance_docs = 0
    # Considering non-zero relevancy scores as relevant
    for i in range(k):
        if Relevancy_score_list[i]>0:
            relevance_docs+=1
            AP+=(relevance_docs/(i+1))
    # If no relevant documents, AP is zero
    if relevance_docs==0:
        return 0
    AP/=(relevance_docs)
    # Rounding to two decimals
    return round(AP,2)

# Helper function to calculate NDCG@k
def NDCG_k(evaluated_scores_list,ideal_scores_list,k):
    evaluated_DCG=0
    ideal_DCG=0
    for idx in range(k):
        if idx==0:
            evaluated_DCG = evaluated_scores_list[idx]
            ideal_DCG = ideal_scores_list[idx]
        else:
            evaluated_DCG += evaluated_scores_list[idx]/math.log2(idx+1)
            ideal_DCG += ideal_scores_list[idx]/math.log2(idx+1)
    ################ ASSUMING NDCG IS 0 WHEN IDEAL DCG IS 0  ###################
    if ideal_DCG == 0:
        return 0
    # Rounding to two decimals
    return round(evaluated_DCG/ideal_DCG,2)


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('gold_standard', metavar='gold_standard', nargs='?', default='./Data/qrels.csv', type=str,
                        help='path to the gold standard ranked list')
    parser.add_argument('evaluated_list', metavar='evaluated_list', nargs='?', default='./Assignment2_1_ranked_list_C.csv', type=str,
                        help='path to evaluated ranked list')
    args = parser.parse_args()
    
    path_to_gold_standard = args.gold_standard
    path_to_evaluated_list = args.evaluated_list

    
    ''' the gold standard file qrels.csv has marked some documents with relevance score = 2, 
        which either do not exist or have an empty abstract Not considering them for calculations '''

    # Reading the queries file
    df = pd.read_csv('./Data/queries.csv')
    queries = df["query"]

    query_id=[]
    query_text =[]
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    for idx in range(len(df)):
        query = queries[idx]
        word_tokens = word_tokenize(query)
        # Removing stop words
        filtered_sentence = [w.lower() for w in word_tokens if w.lower() not in stop_words]
        # Joining as space separated 
        filtered_sentence = ' '.join(filtered_sentence)
        filtered_sentence = tokenizer.tokenize(filtered_sentence)
        filtered_sentence = ' '.join(filtered_sentence)
        # performing lemmatization (without POS tags) to generate tokens from the corpus
        lemmatized_text = []
        for word in filtered_sentence.split():
            lemmatized_text.append(lemmatizer.lemmatize(word))
        
        query_id.append(idx)
        query_text.append(' '.join(lemmatized_text))

    # Storing into a text file
    query_df= pd.DataFrame()
    query_df["query_id"] = query_id
    query_df["query_text"]=query_text
    # Finding unique words in query
    query_vocab = set(' '.join(query_text).split())
    # Loading the model queries bin path
    with open('./model_queries_1.bin', 'rb') as pkl_handle:
        inverted_index = pickle.load(pkl_handle)

    # Building new inverted index with 20000 words
    new_inv_idx = {}
    i = 0
    # First adding the query words into new inverted index
    for q in query_vocab:
        new_inv_idx[q] = inverted_index[q]
        del inverted_index[q]  # Deleting to avoid repetition
        i += 1
    # Taking words from leftover inverted index to make new inverted index of size 20000
    for k,_ in inverted_index.items():
        new_inv_idx[k] = inverted_index[k]
        i += 1
        if i >= 20000:
            break
    # Clearing to avoid memory error
    inverted_index.clear()
    
    # Finding the documents in the new inverted index to consider new corpus of documents
    existing_cord_ids = []
    for k,v in new_inv_idx.items():
        for x in new_inv_idx[k]:
            existing_cord_ids.append(x[0])
    existing_cord_ids = list(set(existing_cord_ids))
    # Size of the updated document corpus
    N = len(existing_cord_ids)
    print(N)
    

    ######################## CALCULATING THE ACTUAL RELEVANCY SCORES ############################

    # Reading the gold standard file
    df = pd.read_csv(path_to_gold_standard) 
    
    # Stores the gold standard relevance scores
    actual_rel ={}

    # initializing relevance scores list for each query
    for t_id in set(df["topic-id"]):
        actual_rel[t_id]={}

    # Resolving the conflict and choosing the relevance score when iteration value is high 
    for t_id,c_id,ite,rel in zip(df["topic-id"],df["cord-id"],df["iteration"],df["judgement"]):
        if c_id in existing_cord_ids:
            if c_id in actual_rel[t_id]:
                if actual_rel[t_id][c_id][1] < ite :
                   actual_rel[t_id][c_id][0] = rel
                   actual_rel[t_id][c_id][1] = ite
            else :
                actual_rel[t_id][c_id] ={}
                actual_rel[t_id][c_id][0] = rel
                actual_rel[t_id][c_id][1] = ite 
    
    ######################## CALCULATING THE PREDICTION RELEVANCY SCORES #########################

    # Loading the predicted ranked lists
    predicted_ranked_list = pd.read_csv(path_to_evaluated_list)
    
    # creating the relevancy scores variable for predicted ranked list
    relevancy_scores = {}

    for t_id in set(predicted_ranked_list["q_id"]):
        # Initializing relevancy scores for each query to empty list 
        relevancy_scores[(t_id+1)] = []
        for q_id,c_id in zip(predicted_ranked_list["q_id"],predicted_ranked_list["cord_id"]):
            # Finding  the relevancy scores for the ranked lists  
            if q_id==t_id:
                if (q_id+1) in actual_rel and c_id in actual_rel[(q_id+1)]:
                    relevancy_scores[q_id+1].append(actual_rel[(q_id+1)][c_id][0])
                else :
                    relevancy_scores[q_id+1].append(0)

    # storing the ideal relevant scores in sorted order
    ideal_relevant_scores = {}
    for t_id in set(predicted_ranked_list["q_id"]):
        temp =[]
        for c_id in actual_rel[t_id+1]:
            temp.append(actual_rel[t_id+1][c_id][0])
        ideal_relevant_scores[t_id+1] = sorted(temp , reverse=True) 


    ######################## CALCULATING THE AVERAGE PRECISION AND NDCG ##################################

    AveragePrecision_10 =[]
    AveragePrecision_20 =[]
    NDCG_10 =[]
    NDCG_20 =[]

    for t_id in set(predicted_ranked_list["q_id"]):
        AveragePrecision_10.append(AveragePrecision_k(relevancy_scores[t_id+1][:10],10))
        AveragePrecision_20.append(AveragePrecision_k(relevancy_scores[t_id+1][:20],20))
        NDCG_10.append(NDCG_k(relevancy_scores[t_id+1][:10],ideal_relevant_scores[t_id+1][:10],10))
        NDCG_20.append(NDCG_k(relevancy_scores[t_id+1][:20],ideal_relevant_scores[t_id+1][:20],20))

    mAP_10 = sum(AveragePrecision_10)/len(AveragePrecision_10)
    mAP_20 = sum(AveragePrecision_20)/len(AveragePrecision_20)
    averNDCG_10 = sum(NDCG_10)/len(NDCG_10)
    averNDCG_20 = sum(NDCG_20)/len(NDCG_20)

    
    ######################## WRITING THE RESULTS INTO A FILE#############################################

    result_df = pd.DataFrame()
    #result_df.columns = ["topic-id","AveragePrecision_10","NDCG_10","AveragePrecision_20","NDCG_20"]
    result_df["topic_id"] = [i+1 for i in range(len(set(predicted_ranked_list["q_id"])))]
    result_df["AveragePrecision_10"] = AveragePrecision_10
    result_df["AveragePrecision_20"] = AveragePrecision_20
    result_df["NDCG_10"] = NDCG_10
    result_df["NDCG_20"] = NDCG_20
    result_df.loc[len(result_df.index)] = ["mean" , mAP_10 ,  mAP_20 , averNDCG_10 ,averNDCG_20 ]
    #result_df = result_df.append(["mean" , mAP_10 ,  mAP_20 , averNDCG_10 ,averNDCG_20 ])
    url = path_to_evaluated_list[:-17]+"metrics_"+path_to_evaluated_list[-5][0]+".txt"
    result_df.to_csv(url, header = ["topic-id","AveragePrecision_10","AveragePrecision_20","NDCG_10","NDCG_20"], sep = "\t", index = False)
    

if __name__ == '__main__':
    main()