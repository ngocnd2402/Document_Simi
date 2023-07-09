import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import TfidfVectorizer

def read_words_from_file(filename):
    word_set = set()
    with open(filename, 'r',encoding = 'utf-8',errors='ignore') as file:
        for line in file:
            words = line.strip().split()
            word_set.update(words)
    return word_set

filename = 'corpus.txt'
with open(filename, 'r') as corpus:
    corpus_content = corpus.read()


def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.replace('?', ' ? ')
    text = ' '.join(text.split()) # Xóa dấu " " dư thừa
    return text



text1 = '''Climate change poses an urgent threat to our planet. Rising temperatures and extreme weather events demand immediate action. We must transition to renewable energy and adopt sustainable practices to secure a livable future.'''
text2 = '''The evidence is clear: climate change is a pressing issue. We must reduce emissions, promote sustainability, and invest in clean energy. It's time for global cooperation to protect our planet and future generations.'''
preprocess_corpus_content = preprocess_text(corpus_content)
text1 = preprocess_text(text1)
text2 = preprocess_text(text2)
updated_corpus_content = preprocess_corpus_content + '\n' + text1 + '\n' + text2

with open(filename, 'w') as corpus:
    corpus.write(updated_corpus_content)
def create_from_file(words) :
    # Tạo ma trận 2D với tất cả giá trị ban đầu là 0
    matrix = np.zeros((len(words), len(words)))

    # Đọc qua file corpus.txt
    with open('corpus.txt', 'r') as file:
        # Duyệt qua từng dòng trong file
        for line in file:
            # Tách các từ trong dòng
            tokens = line.strip().split()

            # Duyệt qua các từ trong dòng
            for i in range(len(tokens)):
                word = tokens[i]

                # Nếu từ đang xét có trong danh sách từ
                if word in words:
                    # Duyệt qua các từ gần kề
                    for j in range(i - 1, i + 2 , 2):
                        # Kiểm tra giới hạn chỉ số
                        if j >= 0 and j < len(tokens) and tokens[j] in words:
                            # Tăng giá trị tại vị trí [word, word1] lên 1
                            matrix[words.index(word), words.index(tokens[j])] += 1

    return matrix

def create_from_text(text,words) :
    # Tạo ma trận 2D với tất cả giá trị ban đầu là 0
    matrix = np.zeros((len(words), len(words)))

    # Duyệt qua từng dòng trong file
    tokens = text.split()

    # Duyệt qua các từ trong dòng
    for i in range(len(tokens)):
        word = tokens[i]

        # Nếu từ đang xét có trong danh sách từ
        if word in words:
            # Duyệt qua các từ gần kề
            for j in range(i - 1, i + 2 , 2):
                # Kiểm tra giới hạn chỉ số
                if j >= 0 and j < len(tokens) and tokens[j] in words:
                    # Tăng giá trị tại vị trí [word, word1] lên 1
                    matrix[words.index(word), words.index(tokens[j])] += 1

    return matrix


def create_token(text):
    token = set(text.split())
    return token

word1 = list(sorted(create_token(text1)))
word2 = list(sorted(create_token(text2)))

matrix1 = create_from_text(text1,word1)
matrix2 = create_from_text(text2,word2)

#words = read_words_from_file(filename)
words = read_words_from_file(filename)
#print(len(words))
# Danh sách các từ
# words = ['word1', 'word2', 'word3', 'word4']
# words = set(words)
words = sorted(words)
words = list(words)
words.extend(word1)
words.extend(word2)

X = create_from_file(words)
la = np.linalg
U,s,Vh = la.svd(X,full_matrices=False)
type(U)
num_rows = len(U)
num_cols = len(U[0])  # Giả sử các hàng có cùng số cột
# Create an empty graph

with open(filename, 'w') as corpus:
    corpus.write(corpus_content)

G1 = nx.Graph()
for i in range(len(word1)):
    G1.add_node(word1[i], label = word1[i])
G2 = nx.Graph()
for i in range(len(word2)):
    G2.add_node(word2[i], label = word2[i])
def distance(A,B):
  i = words.index(A)
  j = words.index(B)
  point1 = (U[i,0],U[i,1])
  point2 = (U[j,0],U[j,1])
  return euclidean(point1, point2)

# Add edges based on the relationships in the matrices
for i in range(len(word1)):
    for j in range(i+1, len(word1)):
        #words[i] ss words[j]
        if matrix1[i][j] != 0:
            dis = distance(word1[i],word1[j])
            G1.add_edge(word1[i], word1[j])

for i in range(len(word2)):
    for j in range(i+1, len(word2)):
        #words[i] ss words[j]
        if matrix2[i][j] != 0:
            dis = distance(word2[i],word2[j])
            G2.add_edge(word2[i], word2[j])

'''print(word1,word2)
for i in range(len(word1)):
    for j in range(len(word2)):
        print(word1[i],word2[j],distance(word1[i],word2[j]))'''
# Draw the graph
from sklearn.neighbors import NearestNeighbors
def get_closest_nodes(G1, G2):
    closest_nodes = {}
    X_knn = []
    for i in range(len(words)):
        X_knn.append([U[i,0],U[i,1]])
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X_knn)
    
    for node1 in G1.nodes:
        distances, indices = nbrs.kneighbors([U[words.index(node1), :2]])
        closest_nodes[node1] = distances
        
    return closest_nodes

closest_nodes = get_closest_nodes(G1, G2)
#print(closest_nodes)
def node_match(node1, node2):
    #print(node1,node2)
    if distance(node1['label'],node2['label']) in closest_nodes[node1['label']]:
        #print(node1,node2)
        return True
    else:
        return False
print(len(G1), len(G2))
ged = nx.graph_edit_distance(G1, G2, node_match=node_match)
total_cost_graph1 = len(G1.nodes) + len(G1.edges)
total_cost_graph2 = len(G2.nodes) + len(G2.edges)
normalization_factor = max(total_cost_graph1, total_cost_graph2)
count = 0

similarity = 1 - float(ged)/normalization_factor
print(similarity,ged)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
pos1 = nx.spring_layout(G1)  # Layout algorithm to position the nodes
edge_labels1 = nx.get_edge_attributes(G1, 'weight')
nx.draw_networkx(G1, pos=pos1, with_labels=True, node_color='lightblue')
#nx.draw(G, pos, with_labels=True, node_size=200, font_size=8)
nx.draw_networkx_edge_labels(G1, pos=pos1, edge_labels=edge_labels1)

plt.subplot(1, 2, 2)
pos2 = nx.spring_layout(G2)  # Layout algorithm to position the nodes
edge_labels2 = nx.get_edge_attributes(G2, 'weight')
nx.draw_networkx(G2, pos=pos2, with_labels=True, node_color='lightblue')
#nx.draw(G, pos, with_labels=True, node_size=200, font_size=8)
nx.draw_networkx_edge_labels(G2, pos=pos2, edge_labels=edge_labels2)

plt.show()