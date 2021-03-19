import pandas as pd
from constants import *


labels = []
paths = ELF_PATHS
for path in paths:
    name = path.split('/')[-1].replace('.adjlist', '')
    label = path.split('/')[-2]
    graph_path = 'data/scg/' + name + '.adjlist'
    if glob(graph_path):
        labels.append({
            'name': name,
            'label': label
        })
pd.DataFrame(labels).to_csv('data/labels.csv', index=None)
