import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import json
import warnings
from sklearn.naive_bayes import MultinomialNB
from k_means_constrained import KMeansConstrained
pd.set_option('display.min_rows',40)
pd.set_option('display.max_columns', 50)
# pd.set_option('display.width', 1000)
from collections import Counter


RANDOM_STATE = 42

labeled_path = 'labeled.csv'
unlabeled_path = 'unlabeled.csv'

warnings.filterwarnings("ignore")
cols_to_use_unlabeled = ['id','country', 'region', 'founded', 'size']
cols_to_use_labeled = cols_to_use_unlabeled + ['industry']
train_kmeans = pd.read_csv(labeled_path, nrows=50000, usecols=cols_to_use_labeled)  # small sample
train_kmeans.set_index('id', inplace=True)
regions = sorted([x for x,v in train_kmeans['region'].value_counts().iteritems()], reverse=True)[:0]
countries = ['united states', 'united kingdom', 'canada', 'australia'][:0]
print('train data loaded')

with open('205552599_205968043.json', 'r') as f:
     snippets = json.load(f)
print('snippets loaded')

# count each term frequency
cnt = Counter()
for snippet in snippets:
     for term in snippet['snippet']:
          cnt[term] += 1
FREQ_THRESH = 7000
COMMON_THRESH = 7000
frequent_terms = {k:count for k,count in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)[:FREQ_THRESH]}
common = {k:count for k,count in sorted(dict(cnt).items(), key=lambda item: item[1], reverse=True)[:COMMON_THRESH]}
snippet_cols = [f'has {str(term)} snippet' for term in frequent_terms.keys()]
country_cols = [f'from {c}' for c in countries]
region_cols = [f'from {r}' for r in regions]
print(f'{len(frequent_terms)=}')
print(f'{len(common)=}')


# normalize columns 'founded' and 'size'
def normalize(df):
    df['founded'] = (df['founded'] - df['founded'].min()) / (
            df['founded'].max() - df['founded'].min())
    df['size'] = df['size'].apply(lambda x: np.log(int(str(x).split('-')[0].replace('+', ''))))
    df['size'] = (df['size'] - df['size'].min()) / (df['size'].max() - df['size'].min())
    print('normalized')


normalize(train_kmeans)
print('fragments split')
NB_clf = MultinomialNB()


def make_feature_columns(fragment):
    for frequent_term in (frequent_terms.keys()):
         fragment[f'has {str(frequent_term)} snippet'] = 0
    for snippet in snippets[fragment.index[0]-1:fragment.index[-1]]:
         for term in snippet['snippet']:
              if term in frequent_terms.keys():
                   fragment.at[snippet['id'], f'has {str(term)} snippet'] = 1
    pd.set_option('display.max_columns', None)
    for country in countries:
        fragment[f'from {country}'] = fragment['country'].apply(lambda x: int(x == country))
    for r in regions:
        fragment[f'from {r}'] = fragment['region'].apply(lambda x: int(x == r))
    return fragment.drop(['country', 'region'], axis=1)

def extract_data_for_clustering(train_kmeans):
    # create intustry2clusterid
    print('begin clustering')
    industry2vec_df = pd.DataFrame()
    train_kmeans = make_feature_columns(train_kmeans)
    train_kmeans = train_kmeans.dropna(subset=['founded', 'size'])
    print(train_kmeans.shape)
    grouped = train_kmeans.groupby('industry')
    del train_kmeans
    print('grouped by industry. begin aggregation')
    mean_founded_by_industry = grouped.founded.mean()
    mode_size_by_industry = grouped['size'].agg(lambda x: x.value_counts().index[0])
    industry2vec_df['size'] = mode_size_by_industry
    industry2vec_df['founded'] = mean_founded_by_industry
    mean = grouped[snippet_cols + country_cols + region_cols].apply(lambda x: x.astype(int).mean())
    industry2vec_df = pd.concat(
        [
            industry2vec_df,
            mean
        ], axis=1)
    class_sizes = grouped['size'].count()
    class_proportion = class_sizes/class_sizes.sum()
    industry2vec_df = pd.concat(
            [
                industry2vec_df,
                class_proportion
            ], axis=1
    )
    industry2vec_df = ((industry2vec_df - industry2vec_df.min()) / (industry2vec_df.max() - industry2vec_df.min()) ).dropna(axis='columns')
    del grouped
    return industry2vec_df


industry2vec_df = extract_data_for_clustering(train_kmeans)
X = industry2vec_df.to_numpy()
print(f'{X.shape=}')
labeled = pd.read_csv(labeled_path, usecols=cols_to_use_labeled)  # full train set
labeled.set_index('id', inplace=True)
normalize(labeled)
labeled_splits = [labeled.loc[i:i+999,:] for i in range(0, labeled.shape[0], 1000)]

print('begin k-means')
clf = KMeansConstrained(
     n_clusters=20,
     size_min=np.ceil(147*0.03),
     size_max=np.floor(147*0.1),
     random_state=0
)
clf.fit_predict(X)
industry2vec_df['clusterID'] = clf.labels_ + 1
industry2vec_df = industry2vec_df.loc[:, ['clusterID']]
print(f'{industry2vec_df["clusterID"]}')
# write to csv file
industry2vec_df.to_csv('industery2cluster_205552599_205968043.csv', columns=['clusterID'])

test = labeled_splits[-1]
frequent_terms = common
test = make_feature_columns(test)
test['founded'].fillna((labeled['founded'].mean()), inplace=True)
test['size'].fillna((labeled['size'].mode()), inplace=True)
print(f'{test.shape}')
X_test, y_test = csc_matrix(
            test.drop(columns=['industry']).to_numpy()) \
            , test['industry'].apply(lambda y: industry2vec_df['clusterID'].loc[y])

for fragment_idx in (range(len(labeled_splits)-1)):
    print(f'{fragment_idx=}')
    fragment = labeled_splits[fragment_idx]
    # create column for snippet terms
    fragment = make_feature_columns(fragment)
    # fill NANs
    fragment['founded'].fillna((labeled['founded'].mean()), inplace=True)
    fragment['size'].fillna((labeled['size'].mode()), inplace=True)
    # predict cluster membership for unknown instances
    X_train, y_train = csc_matrix(fragment.drop(columns=['industry'] ).to_numpy()),\
                       fragment['industry'].apply(lambda y: industry2vec_df['clusterID'].loc[y])
    NB_clf.partial_fit(X_train, y_train, classes=list(range(20)))
    if fragment_idx == 0:
        print(f"{NB_clf.score(X_train, y_train)=}")
    print(f"{NB_clf.score(X_test,y_test)=}")
    labeled_splits[fragment_idx] = None
    del fragment

# test
y_hat = NB_clf.predict(X_test)
print(f"{NB_clf.score(X_test,y_test)=}")

mean_founded = labeled['founded'].mean()
mode_size = labeled['size'].mode()
del labeled
# same process for unlabeled...
unlabeled = pd.read_csv(unlabeled_path, usecols=cols_to_use_unlabeled)
unlabeled.set_index('id', inplace=True)
# normalize columns
normalize(unlabeled)
unlabeled_splits = [unlabeled.iloc[i:i+1000,:] for i in range(0, unlabeled.shape[0], 1000)]
unlabeled['clusterID'] = 0
for fragment_idx in range(len(unlabeled_splits)):
    fragment = unlabeled_splits[fragment_idx]
    # create column for snippet terms
    fragment = make_feature_columns(fragment)
    # fill NANs
    fragment['founded'].fillna(mean_founded, inplace=True)
    fragment['size'].fillna(mode_size, inplace=True)
    # predict cluster membership for unknown instances
    X_comp = csc_matrix(fragment.to_numpy())
    y_hat = NB_clf.predict(X_comp)
    # put prediction in dataframe
    unlabeled.loc[fragment.index[0]:fragment.index[-1],'clusterID'] = y_hat
    unlabeled_splits[fragment_idx] = None
# create company2cluster
unlabeled.to_csv('company2cluster_205552599_205968043.csv', columns=['clusterID'])
