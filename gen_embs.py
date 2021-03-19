import pandas as pd
import networkx as nx
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed, dump, load
from karateclub import Graph2Vec, GL2Vec, FeatherGraph
from constants import *


data = pd.read_csv('data/labels.csv')
paths = [f'data/scg-karateclub/{name}.adjlist' for name in data.values[:, 0]]


def load_graph(path):
    G = nx.read_adjlist(path)
    G = nx.Graph(G)
    G = nx.relabel_nodes(G, mapping=dict(zip(G, range(len(G)))))
    return G


graphs = Parallel(n_jobs=N_JOBS)(
    delayed(load_graph)(path) for path in tqdm(paths))

print('Running Graph2vec...')
graph2vec = Graph2Vec(
    wl_iterations=3,
    epochs=50
)
graph2vec.fit(graphs)
embs = graph2vec.get_embedding()
dump(embs, 'data/graph2vec_embs.lz4')

# print('Running GL2Vec...')
# gl2vec = GL2Vec(
#     wl_iterations=3,
#     epochs=50
# )
# gl2vec.fit(graphs)
# embs = gl2vec.get_embedding()
# dump(embs, 'data/gl2vec_embs.lz4')

# print('Running Feather-G...')
# featherg = FeatherGraph()
# featherg.fit(graphs)
# embs = featherg.get_embedding()
# dump(embs, 'data/featherg_embs.lz4')
