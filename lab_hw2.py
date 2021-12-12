from sentence_transformers import SentenceTransformer

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
