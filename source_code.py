from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

text = open('sample_text2.txt').read()
#sentences[:5]

sentences = sent_tokenize(text)

sentences = [s.lower() for s in sentences]
sentences = [re.sub(r'[^a-zA-Z0-9\s]','',s) for s in sentences]
word_corpus = [sent.split() for sent in sentences]

model = Word2Vec(word_corpus, min_count=1, size=300)
#model.wv['i']
#model.most_similar('i')
#word_vectors = model.wv

vectorizer = TfidfVectorizer(token_pattern=r'\w{1,}')
X = vectorizer.fit_transform([" ".join(sent) for sent in word_corpus])
tfidf_Vector = X.toarray()

tfidf_map = vectorizer.vocabulary_

#finding vectors
vectors = []
for sent_no in range(len(word_corpus)):
    vec_sent = []
    tf_idf_sent_sum = 0
    for word in word_corpus[sent_no]:
        vec_sent.append(X.toarray()[sent_no][tfidf_map[word]]*model.wv[word])
        tf_idf_sent_sum += X.toarray()[sent_no][tfidf_map[word]]
    vectors.append(sum(vec_sent)/tf_idf_sent_sum)
                                                       
# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(vectors[i].reshape(1,300), vectors[j].reshape(1,300))[0,0]

#making weighted graph using similarity matrix
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph,max_iter=300)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)