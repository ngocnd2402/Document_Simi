import re

def jaccard_preprocess(document):
    """Preprocess the document: remove non-letter characters, convert to lower case"""
    words = re.sub(r'\W', ' ', document).lower().split()
    return words

def jaccard_unique_terms(doc1, doc2):
    """Create sets of unique terms for two documents"""
    terms1 = list(set(jaccard_preprocess(doc1)))
    terms2 = list(set(jaccard_preprocess(doc2)))
    return terms1, terms2

def jaccard_similarity(doc1, doc2):
    """Calculate Jaccard coefficient"""
    terms1, terms2 = jaccard_unique_terms(doc1, doc2)

    intersection = [value for value in terms1 if value in terms2]

    union = terms1.copy()
    for term in terms2:
        if term not in union:
            union.append(term)

    jaccard_coefficient = float(len(intersection)) / len(union)

    return jaccard_coefficient

text1 = 'I have a dog and i love it'
text2 = 'I have a dog and i love it'
print(jaccard_similarity(text1, text2))
