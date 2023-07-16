import re

def preprocess(document):
    """Preprocess the document: remove non-letter characters, convert to lower case"""
    words = re.sub(r'\W', ' ', document).lower().split()
    return words

def create_unique_terms(doc1, doc2):
    """Create sets of unique terms for two documents"""
    terms1 = list(set(preprocess(doc1)))
    terms2 = list(set(preprocess(doc2)))
    return terms1, terms2

def dice_similarity(doc1, doc2):
    """Calculate Jaccard and Dice coefficients"""
    terms1, terms2 = create_unique_terms(doc1, doc2)

    intersection = [value for value in terms1 if value in terms2]

    union = terms1.copy()
    for term in terms2:
        if term not in union:
            union.append(term)

    dice_coefficient = float(2 * len(intersection)) / (len(terms1) + len(terms2))

    return dice_coefficient


text1 = 'I have a dog and i love it'
text2 = 'I have a dog and i love it'
print(dice_similarity(text1, text2))