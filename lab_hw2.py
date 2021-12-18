import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import json
from sklearn.linear_model import SGDClassifier
from nltk.stem.snowball import SnowballStemmer
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from k_means_constrained import KMeansConstrained
# pd.set_option('display.min_rows',50)
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
# labeled_path = 'C:/Users/אריאל/PycharmProjects/lab_hw1/labeled.csv' # todo remove
# unlabeled_path = 'C:/Users/אריאל/PycharmProjects/lab_hw1/unlabeled.csv'
warnings.filterwarnings("ignore")
cols_to_use_unlabeled = ['id','country', 'founded', 'size']
cols_to_use_labeled = cols_to_use_unlabeled + ['industry']
train_kmeans = pd.read_csv(labeled_path, nrows=80001, usecols=cols_to_use_labeled)  # small sample
# labeled.drop(columns=['country', 'region', 'locality', 'founded', 'size'], inplace=True)
# unlabeled.drop(columns=['country', 'region', 'locality', 'founded', 'size'], inplace=True)
print('train data loaded')

with open('205552599_205968043.json', 'r') as f:
     snippets = json.load(f)
# with open('C:/Users/אריאל/PycharmProjects/lab_hw1/HW1_205552599_205968043/205552599_205968043.json', 'r') as f: # todo change pat
#      snippets = json.load(f)
print('snippets loaded')

# count each term frequency
cnt = Counter()
for snippet in snippets:
     for term in snippet['snippet']:
          cnt[term] += 1
FREQ_THRESH = 5000
VERY_COMMON_THRESH = 5000
frequent_terms = {k:count for k,count in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)[:FREQ_THRESH]}
# frequent_terms = {k:cnt[k] for k in dict(cnt).keys() if cnt[k] >= FREQ_THRESH}
# very_common = {k:cnt[k] for k in dict(cnt).keys() if cnt[k] >= VERY_COMMON_THRESH}
very_common = {k:count for k,count in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)[:VERY_COMMON_THRESH]}
not_very_common_lst = [k for k in frequent_terms.keys() if k not in very_common.keys()]
common_cols = [f'has {str(term)} snippet' for term in frequent_terms.keys()]
print(f'{len(frequent_terms)=}')
print(f'{len(very_common)=}')


# normalize columns
def normalize(df):
    df['founded'] = (df['founded'] - df['founded'].min()) / (
            df['founded'].max() - df['founded'].min())
    df['size'] = df['size'].apply(lambda x: np.log(int(str(x).split('-')[0].replace('+', ''))))
    df['size'] = (df['size'] - df['size'].min()) / (df['size'].max() - df['size'].min())
    print('normalized')


normalize(train_kmeans)
print('fragments split')
NB_clf = MultinomialNB()
linear_clf = SGDClassifier()


def stem(s):
    warnings.filterwarnings("ignore")
    if not isinstance(s, str):  # handle nan values
        return "empty"
    else:
        stemmed = ' '.join([stemmer.stem(y) for y in s.split()])
        return stemmed if len(stemmed) else s


def make_feature_columns(fragment):
    for frequent_term in (frequent_terms.keys()):
         fragment[f'has {str(frequent_term)} snippet'] = 0
    fragment.set_index('id', inplace=True)
    for snippet in snippets[fragment.index[0]-1:fragment.index[-1]]:
         for term in snippet['snippet']:
              if term in frequent_terms.keys():
                   fragment.at[snippet['id'], f'has {str(term)} snippet'] = 1
    pd.set_option('display.max_columns', None)
    return pd.get_dummies(fragment, columns=['country'])


# create intustry2clusterid
print('begin clustering')
# n_rows_for_kmeans = 70000
# train_kmeans = labeled.loc[:n_rows_for_kmeans,:].copy()#.drop(columns=['text'])
train_kmeans = make_feature_columns(train_kmeans)
# print(train_kmeans)
print(train_kmeans.shape)
print('features ready')
grouped = train_kmeans.dropna(subset=['founded', 'size']).groupby('industry')
del train_kmeans
print('grouped by industry. begin aggregation')
mean_founded_by_industry = grouped.founded.mean()#.sort_values(axis=0, ascending=False)
mode_size_by_industry = grouped['size'].agg(lambda x: x.value_counts().index[0])
industry2vec_df = pd.DataFrame()
industry2vec_df['size'] = mode_size_by_industry
industry2vec_df['founded'] = mean_founded_by_industry
snippet_cols = [f'has {str(term)} snippet' for term in frequent_terms.keys()]
not_very_common_cols = [f'has {str(term)} snippet' for term in not_very_common_lst]
# for t in snippet_cols:
# sums = grouped[snippet_cols].apply(lambda x : x.astype(int).sum())
for col in snippet_cols:
    mean = grouped[col].apply(lambda x: x.astype(int).mean())
    industry2vec_df = pd.concat(
        [
            industry2vec_df,
            mean
        ], axis=1
)
del grouped
print('begin k-means')
X = industry2vec_df.to_numpy()
clf = KMeansConstrained(
     n_clusters=20,
     size_min=np.ceil(147*0.03),
     size_max=np.floor(147*0.1),
     random_state=0
)
clf.fit_predict(X)
industry2vec_df['clusterID'] = clf.labels_
industry2vec_df = industry2vec_df.loc[:, ['clusterID']]
print(f'{industry2vec_df["clusterID"]}')
# todo check how many rows to write to csv file, names of columns, file name
industry2vec_df.to_csv('industery2cluster.csv', columns=['clusterID'])
# industry2vec_df = pd.read_csv('industery2cluster.csv', index_col='industry')

labeled = pd.read_csv(labeled_path, usecols=cols_to_use_labeled)  # full train set
print('train loaded')
normalize(labeled)
labeled_splits = [labeled.loc[i:i+999,:] for i in range(0, labeled.shape[0], 1000)]
test = labeled_splits[-1]
test = make_feature_columns(test)
test['founded'].fillna((labeled['founded'].mean()), inplace=True)
test['size'].fillna((labeled['size'].mode()), inplace=True)
print(f'{test.shape}')
X_test, y_test = csc_matrix(
            test.drop(columns=['industry'] + not_very_common_cols).to_numpy()) \
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
    X_train, y_train = csc_matrix(fragment.drop(columns=['industry'] + not_very_common_cols).to_numpy()),\
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

mean_founded = labeled['founded'].mean()
mode_size = labeled['size'].mode()
del labeled
# same process for unlabeled...
unlabeled = pd.read_csv(unlabeled_path, usecols=cols_to_use_unlabeled)
# normalize columns
normalize(unlabeled)
unlabeled_splits = [unlabeled.loc[i:i+999,:] for i in range(0, unlabeled.shape[0], 1000)]
first_unlabeled_id, last_unlabeled_id = unlabeled.loc[0,"id"], unlabeled.loc[unlabeled.shape[0]-1,"id"]
unlabeled['clusterID'] = 0
unlabeled.set_index('id', inplace=True)
for fragment_idx in (range(len(unlabeled_splits)-1)):
    fragment = unlabeled_splits[fragment_idx]
    # create column for snippet terms
    fragment = make_feature_columns(fragment)
    # fill NANs
    fragment['founded'].fillna(mean_founded, inplace=True)
    fragment['size'].fillna(mode_size, inplace=True)
    # predict cluster membership for unknown instances
    X_comp = csc_matrix(fragment.drop(columns= not_very_common_cols).to_numpy())
    y_hat = NB_clf.predict(X_comp)
    # put prediction in dataframe
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
# texts, industry = (labeled['text']), (labeled['industry'])
# emb_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# record_embeddings = model.encode(texts)
# labeled['embedding'] = record_embeddings.tolist()
# # print(labeled)
# # labeled['sim2_0'] = labeled.apply(lambda row: cosine_similarity(record_embeddings[1].reshape((-1,1)), np.array(row['embedding'])).reshape((-1,1)), axis=1)

# # labeled['sim2_0'] = sims[0].tolist()
# # print(labeled[['industry','sim2_0']].sort_values(['sim2_0'], axis=0, ascending=False))
# rows_already_read = 0
# industry2vec = {}
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
# fragment.drop(['text'], inplace=True, axis=1)