Group number: 1
Part: 3

Library requirements:
python version used: py 3.7
math
argparse
csv
pandas 
pickle
numpy

The directory structure should look like:

[Main Code Directory]
|
|-------- Assignment2_1_evaluator.py
|
|-------- Assignment3_1_rochhio.py
|
|-------- Assignment3_1_important_words.py
|
|-------- query_tf.bin
|
|-------- doc_tf.bin
|
|-------- id_df_map.bin
|
|-------- idx_to_term.bin
|
|-------- Data
| |
| |---------- CORD-19
| |
| |---------- queries.csv
| |
| |---------- id_mapping.csv
| |
| |---------- qrels.csv


Steps to run:
3A) Assignment3_1_rochhio.py
arguments for parser:
	data_path   : stores path to the CORD-19 folder
			: default: ./Data/CORD-19
	model_q_path: stores path to model_queries_1.bin
			: default: ./model_queries_1.bin
	qrel_path	: path to the gold standard qrels.csv
			: default: ./Data/qrels.csv
	alist_path  : stores path to the Assignment2_1_ranked_list_A.csv
			: default: ./Assignment2_1_ranked_list_A.csv

usage instructions:
python Assignment3_1_rochhio.py <data_path> <model_q_path> <qrel_path> <alist_path>
example usage:
python Assignment3_1_rochhio.py ./Data/CORD-19 ./model_queries_1.bin ./Data/qrels.csv ./Assignment2_1_ranked_list_A.csv
default usage:
python Assignment3_1_rochhio.py

files generated:
./Assignment3_1_rocchio_RF_metrics.csv	: stores metrics calculated for Relevance feedback
./Assignment3_1_rocchio_PsRF_metrics.csv	: stores metrics calculated for Pseudo Relevance feedback

3B) Assignment3_1_important_words.py

arguments for parser:
	data_path   : stores path to the CORD-19 folder
			: default: ./Data/CORD-19
	model_q_path: stores path to model_queries_1.bin
			: default: ./model_queries_1.bin
	alist_path  : stores path to the Assignment2_1_ranked_list_A.csv
			: default: ./Assignment2_1_ranked_list_A.csv

usage instructions:
python Assignment3_1_important_words.py <data_path> <model_q_path> <alist_path>
example usage:
python Assignment3_1_important_words.py ./Data/CORD-19 ./model_queries_1.bin ./Assignment2_1_ranked_list_A.csv
default usage:
python Assignment3_1_important_words.py

files generated:
./Assignment3_1_important_words.csv: stores top 5 important words for every query
