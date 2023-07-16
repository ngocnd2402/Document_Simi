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
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    tfidf = vectorizer.fit_transform(corpus)
    svd = TruncatedSVD(num_svd_components, random_state=500)
    tfidf_lsa = svd.fit_transform(tfidf)
    normalizer = Normalizer(copy=False, norm='l2')
    tfidf_lsa = normalizer.fit_transform(tfidf_lsa)
    similarity = cosine_similarity(tfidf_lsa[-2].reshape(1, -1), tfidf_lsa[-1].reshape(1, -1))
    return abs(similarity[0][0])


text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(lsa_similarity(text1, text2, 5))