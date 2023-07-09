import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# def compute_tf(text):
#     # Tính toán tần suất thuật ngữ (term frequency)
#     tf_text = Counter(text.split())
#     for i in tf_text:
#         tf_text[i] = tf_text[i]/float(len(text.split()))
#     return tf_text

# def compute_idf(word, corpus):
#     # Tính toán đảo tần suất thuật ngữ độc lập (inverse document frequency)
#     return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i.split()]))

# def compute_tf_idf(corpus):
#     # Tính toán TF-IDF cho tất cả các từ trong tập hợp các văn bản
#     documents_list = []
#     for text in corpus:
#         tf_idf_dictionary = {}
#         computed_tf = compute_tf(text)
#         for word in computed_tf:
#             tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
#         documents_list.append(tf_idf_dictionary)
#     return documents_list

# def cosine_similarity(vec1, vec2):
#     intersection = set(vec1.keys()) & set(vec2.keys())
#     numerator = sum([vec1[x] * vec2[x] for x in intersection])

#     sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
#     sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
#     denominator = math.sqrt(sum1) * math.sqrt(sum2)

#     if not denominator:
#         return 0.0
#     return float(numerator) / denominator

# def tfidf_similarity(text1, text2):
#     with open("TF_IDF/corpus.txt", "r", encoding='utf-8') as file:
#         lines = file.readlines()

#     corpus = [line.strip() for line in lines]
#     corpus.append(text1)
#     corpus.append(text2)
#     tfidf_matrix = compute_tf_idf(corpus)
#     return cosine_similarity(tfidf_matrix[-2], tfidf_matrix[-1])

def tfidf_similarity(text1, text2):
    with open(r"D:\UIT\Năm 2\Kỳ 4\Tính toán đa phương tiện\Doc Simi\TF_IDF\corpus.txt", "r", encoding='utf-8') as file:
        lines = file.readlines()

    corpus = [line.strip() for line in lines]
    corpus.append(text1)
    corpus.append(text2)
    vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_sim = cosine_similarity(tfidf_matrix[-2], tfidf_matrix[-1])[0][0]
    return cosine_sim

text1 = 'I have a dog and i love it'
text2 = 'I have a dog and i love it'
print(tfidf_similarity(text1, text2))
