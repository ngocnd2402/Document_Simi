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

# Function for preprocessing
def preprocess_text(text):
    text = nlp(text.lower())
    tokens = []
    for token in text:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)
    return tokens

# Function for creating vector representation
def create_vector(tokens):
    vector = []
    for word in tokens:
        if word in word2vec.key_to_index:
            vector.append(word2vec[word])
        else:
            vector.append(oov_vector)
    vector = np.mean(vector, axis=0)
    vector = vector.reshape(1, -1)
    vector = normalize(vector, norm='l1')
    return vector

# Function for comparison
def compare_vectors(vector1, vector2):
    cosine_similarity_value = cosine_similarity(vector1, vector2)[0][0]
    return float(cosine_similarity_value)

# Use the functions
text1 = "your first text here"
text2 = "your second text here"
text1_tokens = preprocess_text(text1)
text2_tokens = preprocess_text(text2)
vector1 = create_vector(text1_tokens)
vector2 = create_vector(text2_tokens)
similarity = compare_vectors(vector1, vector2)
print(f"The cosine similarity between the two texts is: {similarity}")

# Calculate and print similarity of vectors
word1 = 'king'
word2 = 'man'
word3 = 'woman'
word4 = 'queen'
vectora = word2vec[word1] - word2vec[word2] + word2vec[word3]
vectorb = word2vec[word4]
cosine_similarity_value = cosine_similarity(vectora.reshape(1, -1), vectorb.reshape(1, -1))[0][0]
print("Cosine similarity between vector a and vector b:", cosine_similarity_value)