import math
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def bow_vec(document, vocabulary):
    vector = {term: document.split().count(term) for term in vocabulary}
    return vector

def cosine_simi(vec1, vec2):
    dot_product = sum(vec1[key] * vec2.get(key, 0) for key in vec1)
    magnitude1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    return dot_product / (magnitude1 * magnitude2)

def preprocess_text(text):
    text = text.lower() 
    text = re.sub('[' + string.punctuation + ']', '', text)  
    text = ' '.join(text.split())
    return text

def bow_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    corpus = [text1, text2]
    vocabulary = set()
    for document in corpus:
        vocabulary.update(document.split())
    vocabulary = sorted(list(vocabulary))
    vector1 = bow_vec(text1, vocabulary)
    vector2 = bow_vec(text2, vocabulary)
    cosine_sim = cosine_simi(vector1, vector2)
    return cosine_sim

def bow_similarity_lib(text1, text2):
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
    corpus = [text1, text2]
    vectorized_corpus = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(vectorized_corpus)
    similarity = similarity_matrix[0, 1]
    return similarity

text1 = 'I have a dog and i love it'
text2 = 'i have a cat and i hate it'
print(bow_similarity(text1, text2))
print(bow_similarity_lib(text1, text2))
