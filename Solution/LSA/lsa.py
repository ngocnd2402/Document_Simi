from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize


def lsa_similarity(text1, text2, num_svd_components):
    with open("TF_IDF/corpus.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()
    corpus = [line.strip() for line in lines]
    corpus.append(text1)
    corpus.append(text2)
    vectorizer = CountVectorizer(stop_words='english')
    term_freq_matrix = vectorizer.fit_transform(corpus)
    l2_norm_term_freq_matrix = normalize(term_freq_matrix, norm='l2', axis=1)
    svd = TruncatedSVD(num_svd_components)
    lsa = make_pipeline(svd, Normalizer(copy=False, norm='l2'))
    lsa_matrix = lsa.fit_transform(l2_norm_term_freq_matrix)
    similarity = cosine_similarity(lsa_matrix[-2].reshape(1, -1), lsa_matrix[-1].reshape(1, -1))
    return abs(similarity[0][0])

text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(lsa_similarity(text1, text2))