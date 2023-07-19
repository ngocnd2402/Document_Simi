import math
# code sử dụng thư viện
# from nltk import ngrams
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# def ngram_similarity(text1, text2,n_gram = 3):
#     corpus = [text1, text2]
#     if n_gram > 1:
#         # Tạo danh sách các n-gram cho cả hai đoạn văn bản
#         ngrams_corpus = []
#         for doc in corpus:
#             grams = [' '.join(gram) for gram in ngrams(doc.split(), n_gram)]
#             ngrams_corpus.append(' '.join(grams))
#         corpus = ngrams_corpus

#     vectorizer = CountVectorizer()
#     vectorized_corpus = vectorizer.fit_transform(corpus)
#     similarity_matrix = cosine_similarity(vectorized_corpus)
#     similarity = similarity_matrix[0, 1]
#     return similarity

def tokenize_text(text, n):
    words = text.lower().split()
    tokens = [' '.join(words[i:i+n]) for i in range(len(words)-(n-1))]
    return tokens

def create_vocabulary(tokens1, tokens2):
    vocabulary = set(tokens1 + tokens2)
    return vocabulary

def document_representation(ngrams, vocabulary):  
    document_vector = {}
    for ngram in vocabulary:
        if ngram in ngrams:
            if ngram in document_vector:
                document_vector[ngram] += 1
            else:
                document_vector[ngram] = 1
    
    return document_vector

def cosine_simi(vec1, vec2):
    dot_product = sum(vec1[key] * vec2.get(key, 0) for key in vec1)
    magnitude1 = math.sqrt(sum(value ** 2 for value in vec1.values()))
    magnitude2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    return dot_product / (magnitude1 * magnitude2)

def ngram_similarity(text1, text2, n):
    token1= tokenize_text(text1, n)
    token2= tokenize_text(text2, n)
    vocab = create_vocabulary(token1, token2)
    vec1 = document_representation(token1, vocab)
    vec2 = document_representation(token2, vocab)
    return cosine_simi(vec1, vec2)

text1 = 'I have a dog and I love it'
text2 = 'I have a cat and I hate it'
n = 3

similarity_score = ngram_similarity(text1, text2, n)
print("Similarity Score:", similarity_score)
