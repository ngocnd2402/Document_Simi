''' Dataset for Sematic Textual Similarity task '''
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import spacy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import nltk
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_sm')
with open('LP50.txt', 'r') as f:
    documents = [line.strip() for line in f]
print(len(documents))

gt_scores_df = pd.read_csv('LP50_averageScores.csv')
gt_scores = gt_scores_df['average'].values

''' Function to calculate similarities '''
def calc_similarity_scores_ordered(documents, metric_function):
    scores = []
    for index1, doc1 in enumerate(documents):
        for index2, doc2 in enumerate(documents):
            if index1 < index2:
                score = metric_function(doc1, doc2)
                scores.append(score)
    return scores


# @@@ Dưới đây là một số metric để tính similarity, và sau đó tính hệ số tương quan hạng Spearman @@@

'''---------------Word2vec-------------'''
word2vec = KeyedVectors.load_word2vec_format(r'Doc2Vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
def word2vec_similarity(text1 :str, text2: str) -> float:
    text1 = nlp(text1.lower())
    text2 = nlp(text2.lower())
    text1 = [token.lemma_ for token in text1 if not token.is_stop and not token.is_punct]
    text2 = [token.lemma_ for token in text2 if not token.is_stop and not token.is_punct]
    vector1 = np.mean([word2vec[word] for word in text1 if word in word2vec.key_to_index], axis=0)
    vector2 = np.mean([word2vec[word] for word in text2 if word in word2vec.key_to_index], axis=0)
    cosine_similarity_value = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return float(cosine_similarity_value)

'''----------------LSA-----------------'''
with open(r"TF_IDF\corpus.txt", "r", encoding='utf-8') as file:
    lines = file.readlines()
    corpus = [line.strip() for line in lines]

def lsa_similarity(text1 :str, text2 :str) -> float:
    corpus.append(text1)
    corpus.append(text2)
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    tfidf = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(70)
    lsa = make_pipeline(svd, Normalizer(copy=False, norm='l2'))
    tfidf_lsa = lsa.fit_transform(tfidf)
    similarity = cosine_similarity(tfidf_lsa[-2].reshape(1, -1), tfidf_lsa[-1].reshape(1, -1))
    return abs(similarity[0][0])

def convert_tag(tag):
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

def doc_to_synsets(doc):
    tokens = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in pos]
    wntag = [convert_tag(tag) for tag in tags]
    ans = list(zip(tokens,wntag))
    sets = [wn.synsets(x,y) for x,y in ans]
    final = [val[0] for val in sets if len(val) > 0]
    return final

def similarity_score(s1, s2):
    s =[]
    for i1 in s1:
        r = []
        scores = [x for x in [i1.path_similarity(i2) for i2 in s2] if x is not None]
        if scores:
            s.append(max(scores))
    if len(s) == 0:
        return 0  # return a default value when no matches found
    else:
        return sum(s)/len(s)

def document_path_similarity(doc1, doc2):
    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)
    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

word2vec_scores = calc_similarity_scores_ordered(word2vec_similarity)
lsa_scores = calc_similarity_scores_ordered(documents, lsa_similarity)
document_path_scores = calc_similarity_scores_ordered(documents, document_path_similarity)

print('Spearman between Word2vec and groundtruth: ',spearmanr(word2vec_scores, gt_scores)[0])
print('Spearman between LSA and groundtruth: ',spearmanr(lsa_scores, gt_scores)[0])
print('Spearman between Synset and groundtruth: ',spearmanr(document_path_scores, gt_scores)[0])

# Code chay 
def spearman_correlation(x, y):
    n = len(x)
    if len(y) != n:
        raise ValueError("Input lists must have the same length")
    
    rank_x = pd.Series(x).rank()
    rank_y = pd.Series(y).rank()
    
    diff = rank_x - rank_y
    squared_diff = diff ** 2
    
    return 1 - (6 * squared_diff.sum()) / (n * (n**2 - 1))

word2vec_spearman = spearman_correlation(word2vec_scores, gt_scores)
lsa_spearman = spearman_correlation(lsa_scores, gt_scores)
synset_spearman = spearman_correlation(document_path_scores, gt_scores)

print('Spearman between Word2vec and groundtruth: ', word2vec_spearman)
print('Spearman between LSA and groundtruth: ', lsa_spearman)
print('Spearman between Synset and groundtruth: ', synset_spearman)
