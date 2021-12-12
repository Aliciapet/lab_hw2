from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import entropy
import json
import pickle
from evaluate import evaluate
from nltk.stem.snowball import SnowballStemmer
import warnings
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.min_rows',50)
pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)
from collections import Counter
stemmer = SnowballStemmer("english")


def remove_digits(s):
    if not isinstance(s, str):
        return
    else:
        return ''.join([i for i in s if not i.isdigit()])


def stem(s):
    warnings.filterwarnings("ignore")
    if not isinstance(s, str):  # handle nan values
        return "empty"
    else:
        stemmed = ' '.join([stemmer.stem(y) for y in s.split()])
        return stemmed if len(stemmed) else s


labeled_path = 'labeled.csv'
unlabeled_path = 'unlabeled.csv'
warnings.filterwarnings("ignore")
labeled = pd.read_csv(labeled_path, nrows=50000)
unlabeled = pd.read_csv(unlabeled_path, nrows=5)
# labeled.drop(columns=['country', 'region', 'locality', 'founded', 'size'], inplace=True)
# unlabeled.drop(columns=['country', 'region', 'locality', 'founded', 'size'], inplace=True)
# labeled_splits = [labeled.loc[0:29999,:]]+[labeled.loc[i:i+9999,:] for i in range(30000, labeled.shape[0], 10000)]
texts, industry = (labeled['text']), (labeled['industry'])
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# record_embeddings = model.encode(texts)
# labeled['embedding'] = record_embeddings.tolist()
# # print(labeled)
# # labeled['sim2_0'] = labeled.apply(lambda row: cosine_similarity(record_embeddings[1].reshape((-1,1)), np.array(row['embedding'])).reshape((-1,1)), axis=1)
# # sims = cosine_similarity(
# #     [record_embeddings[1]],
# #     record_embeddings[:]
# # )
# # labeled['sim2_0'] = sims[0].tolist()
# # print(labeled['text'].loc[1]) # [np.argmax(sims)]
# # print(labeled[['industry','sim2_0']].sort_values(['sim2_0'], axis=0, ascending=False))
# rows_already_read = 0
industry2vec = {}
# emb_sum_by_industry = {}
# n_per_class = {}
# for i, r in labeled[['industry','embedding']].iloc[rows_already_read:].iterrows():
#     label = r['industry']
#     if label not in emb_sum_by_industry.keys():
#         emb_sum_by_industry[label] = np.array(r['embedding'])
#         n_per_class[label] = 1
#     else:
#         emb_sum_by_industry[label] += np.array(r['embedding'])
#         n_per_class[label] += 1
#     rows_already_read += 1
# for label, n in n_per_class.items():
#     industry2vec[label] = (emb_sum_by_industry[label] / n)
# print(f'{n_per_class=}')
# # print(f'{emb_sum_by_industry=}')
# # print(f'{industry2vec=}')
# print(f'{rows_already_read=}')
# print(len(n_per_class))
# with open('temp_sum.json', 'w') as f:
#     json.dump({'emb sum by industry': {k: vec.tolist() for k, vec in emb_sum_by_industry.items()},
#               'industry2vec': {k: vec.tolist() for k, vec in industry2vec.items()},
#                'n per class': n_per_class, 'rows already read': rows_already_read}, f, indent=1)
# # .to_records()


# labeled['text'] = labeled['text'].apply(remove_digits)
# labeled['text'] = labeled['text'].apply(stem)  # Stem every word.
# print('stemmed')

# tf_vectorizer = CountVectorizer(max_df=.60, min_df=50, stop_words='english', max_features=30000)
# # create a vocabulary based on one fragment
# tf_vectorizer.fit(labeled['text'])
# tfidf_transformer = TfidfTransformer(use_idf=True)
#
# # transformed_train_tf_documents = tf_vectorizer.fit_transform(labeled['text'])
# # transformed_train_tfidf_documents = tfidf_transformer.fit_transform(transformed_train_tf_documents)
# swapped_vocabulary = dict((v,k) for k,v in tf_vectorizer.vocabulary_.items())
# vocab_size = len(swapped_vocabulary)
# print('vocab-size = ', vocab_size)
# # find label names and a penalty for each term
# label_names, penalty_per_term, num_of_industries = fit(first_fragment, transformed_train_tf_documents)

# with open('HW1_205552599_205968043/205552599_205968043.json', 'r') as f:
#      snippets = json.load(f)

with open('snippets50000.json', 'r') as f:
     snippets = json.load(f)

cnt = Counter()
for snippet in snippets:
     for term in snippet['snippet']:
          cnt[term] += 1
FREQ_THRESH = 50
frequent_terms = {k:cnt[k] for k in dict(cnt).keys() if cnt[k]>= FREQ_THRESH}
print(frequent_terms)
print(f'{len(frequent_terms)}=')
for frequent_term in (frequent_terms.keys()):
     labeled[f'has {str(frequent_term)} snippet'] = 0
labeled.set_index('id', inplace=True)
print(labeled)
for snippet in snippets:
     for term in snippet['snippet']:
          if term in frequent_terms.keys() and frequent_terms[term] >= FREQ_THRESH:
               labeled.at[snippet['id'], f'has {str(term)} snippet'] = 1

pd.set_option('display.max_rows', None)
grouped = labeled.dropna(subset=['founded', 'size']).groupby('industry')
print(grouped.count())
mean_founded_by_industry = grouped.founded.mean().sort_values(axis=0, ascending=False)
mode_size_by_industry = grouped['size'].agg(lambda x: x.value_counts().index[0]).apply(lambda x: int(str(x).split('-')[0].replace('+','')))
industry2vec_df = pd.DataFrame()
industry2vec_df['size'] = mode_size_by_industry
industry2vec_df['founded'] = mean_founded_by_industry
snippet_cols = [f'has {str(term)} snippet' for term in frequent_terms.keys()]
for t in snippet_cols:
    print(t)
    sum = grouped[t].apply(lambda x : x.astype(int).sum())
    industry2vec_df = pd.concat(
        [
            industry2vec_df,
            sum
        ], axis=1
    )

print(industry2vec_df)
print(mode_size_by_industry.value_counts())
X = industry2vec_df.to_numpy()
from k_means_constrained import KMeansConstrained
clf = KMeansConstrained(
     n_clusters=20,
     size_min=np.ceil(147*0.03),
     size_max=np.floor(147*0.1),
     random_state=0
)
clf.fit_predict(X)
industry2vec_df['kmeans.labels'] = clf.labels_
print(industry2vec_df['kmeans.labels'])
# print(industry2vec_df['kmeans.labels'].value_counts())


labeled['founded'].fillna((labeled['founded'].mean()), inplace=True)
labeled['size'].fillna((labeled['size'].mode()), inplace=True)
# for ind in mean_founded_by_industry.keys():



"""
#clustering:

# cluster companies to 20 clusters - each cluster contains at least 5 industries and at most 14.
# method:   A. represent industries as vectors using BERT, https://www.analyticsvidhya.com/blog/2021/05/measuring-text-similarity-using-bert/
#           B. Apply a clustering algorithm
# save the embeddings for later usage
# write the industry2cluster.csv

# mapping
# map each record in unlabeled dataset to a cluster ID
# Method:
# A. maybe use snippet from last assignment
# B. convert text to vector
# classify to the most similiar clusterID (cosine similiarity) according to centroid
# Ask in forum: does the clustering process needs to be similiar for labeled and unlabeled
# Write company2cluster.csv file

# Evaluation:
#
"""