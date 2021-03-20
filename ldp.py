import networkx as nx
import numpy as np
import pandas as pd
import json

from constants import *
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from ldp.ldp import LDP
from Classifiers import Classifiers

paths = glob('data/scg-json/*')


def gen_emb(path):
    name = path.split('/')[-1].replace('.json', '')
    edges = json.load(open(path))['edges']
    G = nx.from_edgelist(edges)
    ldp = LDP()
    emb = ldp.fit_a_graph(G)
    emb = [name] + list(emb)
    return emb


# embs = Parallel(n_jobs=N_JOBS)(delayed(gen_emb)(path) for path in tqdm(paths))
# columns = ['name'] + [f'x_{i}' for i in range(160)]
# pd.DataFrame(embs, columns=columns).to_csv('data/ldp.csv', index=None)

embs = pd.read_csv('data/ldp.csv')
data = pd.read_csv('data/labels.csv')
data = pd.merge(data, embs, on='name')

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
with open('results/1/clf-ldp.txt', 'w') as f:
    f.write(report)
roc.savefig('results/1/roc-ldp.png', dpi=300)

# Scenario 2
X_train = np.concatenate((bashlite, mirai, benign[:2000]))
X_test = np.concatenate((others, benign[2000:]))
y_train = [1]*(len(bashlite)+len(mirai)) + [0]*2000
y_test = [1]*len(others) + [0]*(len(benign)-2000)

report, roc = classifiers.run(X_train, X_test, y_train, y_test)
with open('results/2/clf-ldp.txt', 'w') as f:
    f.write(report)
roc.savefig('results/2/roc-ldp.png', dpi=300)

# Scenario 3
X_train = np.concatenate((mirai, others, benign[:2000]))
X_test = np.concatenate((bashlite, benign[2000:]))
y_train = [1]*(len(mirai)+len(others)) + [0]*2000
y_test = [1]*len(bashlite) + [0]*(len(benign)-2000)

report, roc = classifiers.run(X_train, X_test, y_train, y_test)
with open('results/3/clf-ldp.txt', 'w') as f:
    f.write(report)
roc.savefig('results/3/roc-ldp.png', dpi=300)
