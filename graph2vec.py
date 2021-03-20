import json
import networkx as nx
import subprocess
import pandas as pd
import numpy as np

from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
from constants import *
from Classifiers import Classifiers

DST_PATH = 'data/scg-for-graph2vec/'


def run(path):
    name = path.split('/')[-1].replace('.adjlist', '')
    graph_path = DST_PATH + name + '.json'
    if glob(graph_path):
        return 1

    G = nx.read_adjlist(path)
    G = nx.relabel_nodes(G, mapping=dict(zip(G, range(len(G)))))
    with open(graph_path, 'w') as f:
        json.dump({
            'edges': list(G.edges),
        }, f)
    return 1


# output = Parallel(n_jobs=N_JOBS)(
#     delayed(run)(path) for path in tqdm(SCG_PATHS))
# print(f'Generated {sum(output)} SCGs (graph2vec).')

# cmd = 'python graph2vec/src/graph2vec.py --input-path data/scg-for-graph2vec/ --output-path data/graph2vec.csv --epochs 50 --wl-iterations 3'
# p = subprocess.Popen(cmd, shell=True)
# p.wait()

embs = pd.read_csv('data/graph2vec.csv')
data = pd.read_csv('data/labels.csv')
data = pd.merge(data, embs, left_on='name', right_on='type')

bashlite = data.loc[data['label'] == 'bashlite', 'x_0':].values
mirai = data.loc[data['label'] == 'mirai', 'x_0':].values
others = data.loc[data['label'] == 'others', 'x_0':].values
benign = data.loc[data['label'] == 'benign', 'x_0':].values

classifiers = Classifiers()

# Scenario 1
X_train = np.concatenate((bashlite, others, benign[:2000]))
X_test = np.concatenate((mirai, benign[2000:]))
y_train = [1]*(len(bashlite)+len(others)) + [0]*2000
y_test = [1]*len(mirai) + [0]*(len(benign)-2000)

report, roc = classifiers.run(X_train, X_test, y_train, y_test)
with open('results/1/clf-graph2vec.txt', 'w') as f:
    f.write(report)
roc.savefig('results/1/roc-graph2vec.png', dpi=300)

# Scenario 2
X_train = np.concatenate((bashlite, mirai, benign[:2000]))
X_test = np.concatenate((others, benign[2000:]))
y_train = [1]*(len(bashlite)+len(mirai)) + [0]*2000
y_test = [1]*len(others) + [0]*(len(benign)-2000)

report, roc = classifiers.run(X_train, X_test, y_train, y_test)
with open('results/2/clf-graph2vec.txt', 'w') as f:
    f.write(report)
roc.savefig('results/2/roc-graph2vec.png', dpi=300)

# Scenario 3
X_train = np.concatenate((mirai, others, benign[:2000]))
X_test = np.concatenate((bashlite, benign[2000:]))
y_train = [1]*(len(mirai)+len(others)) + [0]*2000
y_test = [1]*len(bashlite) + [0]*(len(benign)-2000)

report, roc = classifiers.run(X_train, X_test, y_train, y_test)
with open('results/3/clf-graph2vec.txt', 'w') as f:
    f.write(report)
roc.savefig('results/3/roc-graph2vec.png', dpi=300)
