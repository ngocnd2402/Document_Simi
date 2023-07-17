from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import Normalizer
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import spacy
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
nlp = spacy.load('en_core_web_sm')

# Load Doc2Vec model
word2vec = KeyedVectors.load_word2vec_format(r'D:/UIT/Năm 2/Kỳ 4/Tính toán đa phương tiện/Document similarity/Doc2Vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)

oov_vector = np.zeros((300,))
def word2vec_similarity(text1, text2):
    # B1: Preprocessing
    text1 = nlp(text1.lower())
    text2 = nlp(text2.lower())

    # tokenize and remove stop words and punctuation
    text1_tokens = []
    for token in text1:
        if not token.is_stop and not token.is_punct:
            text1_tokens.append(token.lemma_)

    text2_tokens = []
    for token in text2:
        if not token.is_stop and not token.is_punct:
            text2_tokens.append(token.lemma_)

    # B2: Create vector representation
    vector1 = []
    for word in text1_tokens:
        if word in word2vec.key_to_index:
            vector1.append(word2vec[word])
        else:
            vector1.append(oov_vector)
    vector1 = np.mean(vector1, axis=0)

    vector2 = []
    for word in text2_tokens:
        if word in word2vec.key_to_index:
            vector2.append(word2vec[word])
        else:
            vector2.append(oov_vector)
    vector2 = np.mean(vector2, axis=0)

    # B3: Compare
    vector1 = vector1.reshape(1, -1)
    vector1 = normalize(vector1, norm='l1')

    vector2 = vector2.reshape(1, -1)
    vector2 = normalize(vector2, norm='l1')

    cosine_similarity_value = cosine_similarity(vector1, vector2)[0][0]

    return float(cosine_similarity_value)



word1 = 'king'
word2 = 'man'
word3 = 'woman'
word4 = 'queen'
vectora = word2vec[word1] - word2vec[word2] + word2vec[word3]
vectorb = word2vec[word4]
cosine_similarity_value = cosine_similarity(vectora.reshape(1, -1), vectorb.reshape(1, -1))[0][0]
print("Cosine similarity between vector a and vector b:", cosine_similarity_value)
