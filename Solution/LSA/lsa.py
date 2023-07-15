from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize

# Dùng thư viện
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

# Code chay
# import numpy as np
# from scipy import linalg
# from scipy.spatial.distance import cosine
# from collections import Counter
# import re

# def count_vectorize(text, dictionary):
#     word_freq = Counter(text.split())
#     vector = [word_freq[word] for word in dictionary]
#     return vector

# def lsa_similarity_custom(text1, text2, num_svd_components):
#     with open("TF_IDF/corpus.txt", "r", encoding='utf-8') as file:
#         lines = file.readlines()
#     corpus = [re.sub(r'\W+', ' ', line.lower().strip()) for line in lines]
#     corpus.append(text1.lower())
#     corpus.append(text2.lower())
#     dictionary = list(set(' '.join(corpus).split()))
#     term_freq_matrix = np.array([count_vectorize(text, dictionary) for text in corpus])
#     l2_norm_term_freq_matrix = term_freq_matrix / np.linalg.norm(term_freq_matrix, axis=1, keepdims=True)
#     u, s, vh = np.linalg.svd(l2_norm_term_freq_matrix, full_matrices=False)
#     lsa_matrix = np.dot(l2_norm_term_freq_matrix, vh[:num_svd_components, :].T)
#     lsa_matrix = lsa_matrix / np.linalg.norm(lsa_matrix, axis=1, keepdims=True)
#     similarity = 1 - cosine(lsa_matrix[-2], lsa_matrix[-1])
#     return abs(similarity)

# text1 = 'I have a dog and i love it'
# text2 = 'i have a cat and i hate it'
# print(lsa_similarity_custom(text1, text2, 2))


text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(lsa_similarity(text1, text2))