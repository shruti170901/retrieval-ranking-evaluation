import math
import os
import argparse
import json
import csv
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle 

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', metavar='data_path', nargs='?', default='./Data/CORD-19', type=str,
                        help='path to the CORD-19 folder')
    parser.add_argument('bin_path', metavar='bin_path', nargs='?', default='./model_queries_1.bin', type=str,
                        help='path to the model bin file')
    args = parser.parse_args()
    bin_path = args.bin_path
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
    with open(bin_path, 'rb') as pkl_handle:
        inverted_index = pickle.load(pkl_handle)

    # Building new inverted index with 20000 words
    new_inv_idx = {}
    # Maps each of the 20000 words to corresponding index
    term_to_idx = {}
    # Maps each index (0-19999) to the document frequency of the word corresponding to that index
    id_df_map = {}
    # Maps each index (0-19999) to the corresponding word in the vocabulary
    idx_to_term = {}

    i = 0
    # First adding the query words into new inverted index
    for q in query_vocab:
        new_inv_idx[q] = inverted_index[q]
        del inverted_index[q]  # Deleting to avoid repetition
        term_to_idx[q] = i
        idx_to_term[i] = q
        id_df_map[i] = len(new_inv_idx[q])  # Size of posting list of a word is its document frequency
        i += 1
    # Taking words from leftover inverted index to make new inverted index of size 20000
    for k,_ in inverted_index.items():
        term_to_idx[k] = i
        idx_to_term[i] = k
        new_inv_idx[k] = inverted_index[k]
        id_df_map[i] = len(new_inv_idx[k])  # Size of posting list of a word is its document frequency
        i += 1
        if i >= 20000:
            break
    # Clearing to avoid memory error
    inverted_index.clear()
    
    # Finding the documents in the new inverted index to consider new corpus of documents
    cord_ids = []
    for k,v in new_inv_idx.items():
        for x in new_inv_idx[k]:
            cord_ids.append(x[0])
    cord_ids = list(set(cord_ids))
    print(len(cord_ids))
    # Size of the updated document corpus
    N = len(cord_ids)
    
    # Initializing the document frequency vector
    doc_tf = {}
    for c in cord_ids:
        # doc_tf[c] = {}
        doc_tf[c] = [0]*len(new_inv_idx)

    # Building the document term frequency vector
    for k,v in new_inv_idx.items():
        for x in new_inv_idx[k]:
            doc_tf[x[0]][term_to_idx[k]] = x[1]

    # Building the query frequency vector
    query_tf = {}
    for q in query_id:
        query_tf[q] = [0]*len(new_inv_idx)
        for w in query_text[q].split():
            query_tf[q][term_to_idx[w]] += 1


    # Storing the term frequency vectors to load for each ranked list
    with open('./doc_tf.bin', 'wb') as pkl_handle:
        pickle.dump(doc_tf, pkl_handle)
    with open('./query_tf.bin', 'wb') as pkl_handle:
        pickle.dump(query_tf, pkl_handle)
    with open("./id_df_map.bin",'wb') as pkl_handle:
        pickle.dump(id_df_map,pkl_handle)
    with open("./idx_to_term.bin",'wb') as pkl_handle:
        pickle.dump(idx_to_term,pkl_handle)
    
    print('Initial preprocessing done')

    ####################################################  lnc.ltc  ##########################################
    
    # Loading the query term frequency vector
    with open('./query_tf.bin', 'rb') as pkl_handle:
        query_tf = pickle.load(pkl_handle)
    
    # Calculating the query vectors
    for k in query_tf.keys():
        sqred_sum = 0
        for i in range(len(query_tf[k])):
            # term frequency using logarithmic
            if query_tf[k][i] != 0:
                query_tf[k][i] = 1+math.log10(query_tf[k][i])
            # IDF weighting
            query_tf[k][i] *= math.log10(N/id_df_map[i])
            sqred_sum += query_tf[k][i]*query_tf[k][i]
        # normalization using cosine    
        for i in range(len(query_tf[k])):
            query_tf[k][i] /= sqred_sum

    # Initializing the cosine similarity to empty dicts
    cos_sim = {}
    for q in query_tf.keys():
        cos_sim[q] = {}
        
    # Loading the document frequency
    with open('./doc_tf.bin', 'rb') as pkl_handle:
        doc_tf = pickle.load(pkl_handle)

    # Calculating the non-zero terms in each query
    non_zero_query_ids ={}
    for q in query_tf.keys():
        non_zero_query_ids[q]=[]
        for idx in range(len(query_tf[q])):
            if query_tf[q][idx]!=0:
                non_zero_query_ids[q].append(idx)
    
    # Calculating the document vector
    ppp = 0
    for k in doc_tf.keys():
        if ppp%500==0:
            print(ppp)
        ppp += 1
        sqred_sum = 0
        for i in range(len(doc_tf[k])):
            # term frequency using logarithmic
            if doc_tf[k][i] != 0:
                doc_tf[k][i] = 1+math.log10(doc_tf[k][i])
            sqred_sum += doc_tf[k][i]*doc_tf[k][i]
        # Normalization
        for i in range(len(doc_tf[k])):
            doc_tf[k][i] /= sqred_sum

        # Calculating the cosine similarity
        for q in query_tf.keys():
            cos_sim[q][k] = 0
            for j in non_zero_query_ids[q]:
                cos_sim[q][k] += doc_tf[k][j]*query_tf[q][j]
        # Deleting the document tf to avoid memory error
        doc_tf[k] = []

    # Sorting the documents according to similarity scores and writing top 50 ranked documents
    with open('./Assignment2_1_ranked_list_A.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(["q_id","cord_id"]) 
        for q,t in query_tf.items():
            rank_list_a = cos_sim[q]
            rank_list_a = sorted(rank_list_a.items(), key=lambda item: item[1] , reverse = True)
            l = [k for k,v in rank_list_a][:50]
            rank_list_a.clear()
            for x in l:
                writer.writerow([q,x])
    
    print("Rank List A generated")
    ####################################################  Lnc.Lpc  ##########################################

    # Loading the query term frequency vector
    with open('./query_tf.bin', 'rb') as pkl_handle:
        query_tf = pickle.load(pkl_handle)

    # Calculating the query vectors
    for k in query_tf.keys():
        # Initializing the squared sum of tf-idf to 0
        sqred_sum = 0
        # Dimension of the vector
        n = len(query_tf[k])
        # Total term frequencies of all terms 
        total = 0
        # Number of unique terms 
        terms_count=0
        for i in range(n):
            total += query_tf[k][i]
            if query_tf[k][i]!=0:
                terms_count+=1
        average_tf = total/terms_count
        # Calculating LOGARITHMIC AVERAGE
        for i in range(n):
            if query_tf[k][i] != 0:
                query_tf[k][i] = 1+math.log10(query_tf[k][i])
            query_tf[k][i] /= (1+math.log10(average_tf))
            ## Probablistic 
            query_tf[k][i] *= max(0,(math.log10((N-id_df_map[i])/id_df_map[i])))
            # Calculating the squared sum
            sqred_sum += query_tf[k][i]*query_tf[k][i]

        # Normalization with cosine
        for i in range(len(query_tf[k])):
            query_tf[k][i] /= sqred_sum

    # Calculating the non-zero terms in each query 
    non_zero_query_ids ={}
    for q in query_tf.keys():
        non_zero_query_ids[q]=[]
        for idx in range(len(query_tf[q])):
            if query_tf[q][idx]!=0:
                non_zero_query_ids[q].append(idx)

    # Initializing the cosine similarity to empty dicts
    cos_sim = {}
    for q in query_tf.keys():
        cos_sim[q] = {}

    # Loading the document term frequency vector
    with open('./doc_tf.bin', 'rb') as pkl_handle:
        doc_tf = pickle.load(pkl_handle)

    sqred_sum = 0
    ppp = 0
    for k in doc_tf.keys():
        if ppp%500 == 0:
            print(ppp)
        ppp += 1
        # Dimension of the vector
        n = len(doc_tf[k])
        # Total term frequencies of all terms 
        total = 0
        # Number of unique terms
        terms_count=0
       
        for i in range(n):
            total += doc_tf[k][i]
            if doc_tf[k][i]!=0:
                terms_count+=1
        average_tf = total/terms_count
        # Logarithmic Average 
        for i in range(n):
            if doc_tf[k][i] != 0:
                doc_tf[k][i] = 1+math.log10(doc_tf[k][i])
            doc_tf[k][i] /= 1+math.log10(average_tf)
            # Calculating squared sum
            sqred_sum += doc_tf[k][i]*doc_tf[k][i]
        # Normalization using cosine
        for i in range(len(doc_tf[k])):
            doc_tf[k][i] /= sqred_sum
  
        # Calculating the cosine similarity
        for q in query_tf.keys():
            cos_sim[q][k] = 0
            for j in non_zero_query_ids[q]:
                cos_sim[q][k] += doc_tf[k][j]*query_tf[q][j]

        # Deleting the doc_tf[k] to avoid memory error
        doc_tf[k] = []

    # Sorting the documents according to similarity scores and writing top 50 ranked documents
    with open('./Assignment2_1_ranked_list_B.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(["q_id","cord_id"]) 
        for q,t in query_tf.items():
            rank_list_a = cos_sim[q]
            rank_list_a = sorted(rank_list_a.items(), key=lambda item: item[1] , reverse = True)
            l = [k for k,v in rank_list_a][:50]
            rank_list_a.clear()
            for x in l:
                writer.writerow([q,x])
            
    print("RankList B generated")

    # # ####################################################  anc.apc  ##########################################

    # Loading the query term frequency vector
    with open('./query_tf.bin', 'rb') as pkl_handle:
        query_tf = pickle.load(pkl_handle)

    # Calculating the query vectors
    for k in query_tf.keys():
        # Initializing the squared sum of tf-idf to 0
        sqred_sum = 0
        # Dimension of vector
        n = len(query_tf[k])
        # Calculating the maximum term frequency in query vector
        max_tf = 0
        for i in range(n):
            max_tf = max(query_tf[k][i], max_tf)

        # Augmented term frequency
        for i in range(n):
            query_tf[k][i] = 0.5+(0.5*query_tf[k][i])/max_tf
        # Probablistic idf 
            query_tf[k][i] *= max(0,(math.log10((N-id_df_map[i])/id_df_map[i])))
        
            sqred_sum += query_tf[k][i]*query_tf[k][i]
        # Normalization using cosine
        for i in range(len(query_tf[k])):
            query_tf[k][i] /= sqred_sum

    # Initializing the cosine similarity to empty dicts
    cos_sim = {}
    for q in query_tf.keys():
        cos_sim[q] = {}

    # Loading the document term frequency vectors
    with open('./doc_tf.bin', 'rb') as pkl_handle:
        doc_tf = pickle.load(pkl_handle)

    sqred_sum = 0
    ppp = 0


    for k in doc_tf.keys():
        if ppp%50 == 0:
            print(ppp)
        ppp += 1
        # Dimension of vector
        n = len(doc_tf[k])
        # Calculating the maximum term frequency in document vector
        max_tf = 0
        for i in range(n):
            max_tf = max(max_tf, doc_tf[k][i])
        ## Augmented for term frequency
        for i in range(n):
            doc_tf[k][i] = 0.5+(0.5*doc_tf[k][i])/max_tf
            sqred_sum += doc_tf[k][i]*doc_tf[k][i]
        # Normalization using cosine
        for i in range(len(doc_tf[k])):
            doc_tf[k][i] /= sqred_sum
        
        # Calculating the cosine similarity
        for q in query_tf.keys():
            cos_sim[q][k] = 0
            for j in range(len(doc_tf[k])):
                cos_sim[q][k] += doc_tf[k][j]*query_tf[q][j]
        # Deleting the doc_tf[k] to avoid memory error
        doc_tf[k] = []

    # Sorting the documents according to similarity scores and writing top 50 ranked documents
    with open('./Assignment2_1_ranked_list_C.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(["q_id","cord_id"]) 
        for q,t in query_tf.items():
            rank_list_a = cos_sim[q]
            rank_list_a = sorted(rank_list_a.items(), key=lambda item: item[1] , reverse = True)
            l = [k for k,v in rank_list_a][:50]
            rank_list_a.clear()
            for x in l:
                writer.writerow([q,x])
            

if __name__ == '__main__':
    main()



   