# from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import entropy
import json
from sklearn.linear_model import SGDClassifier
from nltk.stem.snowball import SnowballStemmer
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from k_means_constrained import KMeansConstrained
pd.set_option('display.min_rows',50)
pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)
from collections import Counter
stemmer = SnowballStemmer("english")

RANDOM_STATE = 42


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
labeled = pd.read_csv(labeled_path, nrows=210000)
unlabeled = pd.read_csv(unlabeled_path, nrows=50000)
# labeled.drop(columns=['country', 'region', 'locality', 'founded', 'size'], inplace=True)
# unlabeled.drop(columns=['country', 'region', 'locality', 'founded', 'size'], inplace=True)
texts, industry = (labeled['text']), (labeled['industry'])
# emb_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# record_embeddings = model.encode(texts)
# labeled['embedding'] = record_embeddings.tolist()
# # print(labeled)
# # labeled['sim2_0'] = labeled.apply(lambda row: cosine_similarity(record_embeddings[1].reshape((-1,1)), np.array(row['embedding'])).reshape((-1,1)), axis=1)

# # labeled['sim2_0'] = sims[0].tolist()
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

# labeled['text'] = labeled['text'].apply(remove_digits)
# labeled['text'] = labeled['text'].apply(stem)  # Stem every word.
# print('stemmed')

# with open('205552599_205968043.json', 'r') as f:
#      snippets = json.load(f)
with open('snippets50000.json', 'r') as f:
    snippets = json.load(f)
# with open('C:/Users/אריאל/PycharmProjects/lab_hw1/HW1_205552599_205968043/205552599_205968043.json', 'r') as f: # todo change pat
#      snippets = json.load(f)
     # snippets = snippets[:2100]

cnt = Counter()
for snippet in snippets:
     for term in snippet['snippet']:
          cnt[term] += 1
FREQ_THRESH = 500# 5000
VERY_COMMON_THRESH = 500# 5000
frequent_terms = {k:count for k,count in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)[:FREQ_THRESH]}
# frequent_terms = {k:cnt[k] for k in dict(cnt).keys() if cnt[k] >= FREQ_THRESH}
# very_common = {k:cnt[k] for k in dict(cnt).keys() if cnt[k] >= VERY_COMMON_THRESH}
very_common = {k:count for k,count in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)[:VERY_COMMON_THRESH]}
not_very_common_lst = [k for k in frequent_terms.keys() if k not in very_common.keys()]
common_cols = [f'has {str(term)} snippet' for term in frequent_terms.keys()]
print(f'{len(frequent_terms)=}')
print(f'{len(very_common)=}')

print('loaded')
# normalize columns
labeled['founded'] = (labeled['founded'] - labeled['founded'].min()) / (
            labeled['founded'].max() - labeled['founded'].min())
labeled['size'] = labeled['size'].apply(lambda x: np.log(int(str(x).split('-')[0].replace('+', ''))))
labeled['size'] = (labeled['size'] - labeled['size'].min()) / (labeled['size'].max() - labeled['size'].min())
unlabeled['founded'] = (unlabeled['founded'] - unlabeled['founded'].min()) / (
            unlabeled['founded'].max() - unlabeled['founded'].min())
unlabeled['size'] = unlabeled['size'].apply(lambda x: np.log(int(str(x).split('-')[0].replace('+', ''))))
unlabeled['size'] = (unlabeled['size'] - unlabeled['size'].min()) / (unlabeled['size'].max() - unlabeled['size'].min())
labeled_splits = [labeled.loc[i:i+999,:] for i in range(0, labeled.shape[0], 1000)]
unlabeled_splits = [unlabeled.loc[i:i+9999,:] for i in range(0, unlabeled.shape[0], 10000)]
first_unlabeled_id, last_unlabeled_id = unlabeled.loc[0,"id"], unlabeled.loc[unlabeled.shape[0]-1,"id"]
NB_clf = MultinomialNB()
linear_clf = SGDClassifier()


def stem(s):
    warnings.filterwarnings("ignore")
    if not isinstance(s, str):  # handle nan values
        return "empty"
    else:
        stemmed = ' '.join([stemmer.stem(y) for y in s.split()])
        return stemmed if len(stemmed) else s


def make_feature_columns(fragment, for_train_kmeans=False):
    # if for_train_kmeans:
    #     print('stemming')
    #     fragment['text'] = fragment['text'].apply(stem)  # Stem every word.
    #     # fragment['text'].fillna('', inplace=True)
    #     text_embeddings = emb_model.encode(fragment['text'])
    #     print('embedding added')
    #     fragment = pd.concat(
    #         [
    #             fragment,
    #             pd.DataFrame(text_embeddings)
    #         ], axis=1
    #     )
    fragment.drop(['text'], inplace=True, axis=1)
    for frequent_term in (frequent_terms.keys()):
         fragment[f'has {str(frequent_term)} snippet'] = 0
    fragment.set_index('id', inplace=True)
    for snippet in snippets[fragment.index[0]-1:fragment.index[-1]]:
         for term in snippet['snippet']:
              if term in frequent_terms.keys():
                   fragment.at[snippet['id'], f'has {str(term)} snippet'] = 1
    # print(fragment.loc[fragment.index[0]])
    pd.set_option('display.max_rows', None)
    return fragment


# create intustry2clusterid
train_kmeans = labeled.loc[:100000,:].copy()#.drop(columns=['text'])
train_kmeans = make_feature_columns(train_kmeans, for_train_kmeans=True)
# print(train_kmeans)
print(train_kmeans.shape)
grouped = train_kmeans.dropna(subset=['founded', 'size']).groupby('industry')

del train_kmeans
mean_founded_by_industry = grouped.founded.mean().sort_values(axis=0, ascending=False)
mode_size_by_industry = grouped['size'].agg(lambda x: x.value_counts().index[0])
industry2vec_df = pd.DataFrame()
industry2vec_df['size'] = mode_size_by_industry
industry2vec_df['founded'] = mean_founded_by_industry
snippet_cols = [f'has {str(term)} snippet' for term in frequent_terms.keys()]
not_very_common_cols = [f'has {str(term)} snippet' for term in not_very_common_lst]
# for t in snippet_cols:
# sums = grouped[snippet_cols].apply(lambda x : x.astype(int).sum())
for col in snippet_cols:
    sum = grouped[col].apply(lambda x: x.astype(int).sum()/x.count())
    industry2vec_df = pd.concat(
        [
            industry2vec_df,
            sum
        ], axis=1
)
X = industry2vec_df.to_numpy()
clf = KMeansConstrained(
     n_clusters=20,
     size_min=np.ceil(147*0.03),
     size_max=np.floor(147*0.1),
     random_state=0
)
clf.fit_predict(X)
industry2vec_df['clusterID'] = clf.labels_
print(f'{industry2vec_df["clusterID"]}')
# todo check how many rows to write to csv file, names of columns, file name
industry2vec_df.to_csv('industery2cluster.csv', columns=['clusterID'])
# industry2vec_df = pd.read_csv('industery2cluster.csv', index_col='industry')

test = labeled_splits[-1]
test = make_feature_columns(test)
test['founded'].fillna((labeled['founded'].mean()), inplace=True)
test['size'].fillna((labeled['size'].mode()), inplace=True)
print(f'{test.shape}')
X_test, y_test = csc_matrix(
            test.drop(columns=['country', 'region', 'locality', 'industry'] + not_very_common_cols).to_numpy()) \
            , test['industry'].apply(lambda y: industry2vec_df['clusterID'].loc[y])

for fragment_idx in (range(len(labeled_splits)-1)):
    fragment = labeled_splits[fragment_idx]
    # create column for snippet terms
    fragment = make_feature_columns(fragment)
    # print(fragment[common_cols].sum(axis=1))
    # fill NANs
    fragment['founded'].fillna((labeled['founded'].mean()), inplace=True)
    fragment['size'].fillna((labeled['size'].mode()), inplace=True)
    pd.set_option('display.max_rows', 50)
    # predict cluster membership for unknown instances
    # todo when to drop text? drop 'country'?
    X_train, y_train = csc_matrix(fragment.drop(columns=['country', 'region', 'locality', 'industry'] + not_very_common_cols).to_numpy()),\
                       fragment['industry'].apply(lambda y: industry2vec_df['clusterID'].loc[y])
    NB_clf.partial_fit(X_train, y_train, classes=list(range(20)))
    # linear_clf.partial_fit(X_train, y_train, classes=list(range(20)))
    if fragment_idx == 0:
        print(f"{NB_clf.score(X_train, NB_clf.predict(X_train))=}")
    y_hat = NB_clf.predict(X_test)
    print(f"{NB_clf.score(X_test,y_test)=}")
    labeled_splits[fragment_idx] = None
    del fragment

# test
y_hat = NB_clf.predict(X_test)
print(f"{NB_clf.score(X_test,y_test)=}")
# y_hat = linear_clf.predict(X_test)
# print(f"{linear_clf.score(X_test,y_test)=}")

# todo same process for unlabeled...
unlabeled['clusterID'] = 0
unlabeled.set_index('id', inplace=True)
for fragment_idx in (range(len(unlabeled_splits)-1)):
    fragment = unlabeled_splits[fragment_idx]
    # create column for snippet terms
    fragment = make_feature_columns(fragment)
    # fill NANs
    fragment['founded'].fillna((labeled['founded'].mean()), inplace=True)
    fragment['size'].fillna((labeled['size'].mode()), inplace=True)
    # predict cluster membership for unknown instances
    X_comp = csc_matrix(fragment.drop(columns=['country', 'region', 'locality', 'industry'] + not_very_common_cols).to_numpy())
    y_hat = NB_clf.predict(X_comp)
    # todo put prediction in dataframe
    unlabeled.loc[fragment.index[0]:fragment.index[-1],'clusterID'] = y_hat
    unlabeled_splits[fragment_idx] = None
# create company2cluster
unlabeled.to_csv('company2cluster.csv', columns=['clusterID'])





"""other classifiers"""
# classifier = RandomForestClassifier(random_state=RANDOM_STATE)
# linear_clf.partial_fit(X_train1, y_train1, classes=list(range(20)))
# y_hat = linear_clf.predict(X_test)
# print(f"{linear_clf.score(X_test,y_test)=}")
# linear_clf.partial_fit(X_train2, y_train2, classes=list(range(20)))
# y_hat = linear_clf.predict(X_test)
# print(f"{linear_clf.score(X_test,y_test)=}")
# print(f"{NB_clf.score(X_train1, NB_clf.predict(X_train1))=}")

# classifier.fit(X_train1, y_train1)
# print('fit done')
# y_hat = classifier.predict(X_test)
# print(f"{classifier.score(X_test,y_test)=}")
# print(f"{classifier.score(X_train1,classifier.predict(X_train1))=}")



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