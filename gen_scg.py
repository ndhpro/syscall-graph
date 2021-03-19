import json
import networkx as nx
from glob import glob
from joblib import Parallel, delayed
from constants import *


def scg(rp_path, pid, u):
    edges = []
    path = rp_path + 'strace' + pid + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    for sc in data:
        if sc['name'] == 'execve' and sc['return'] != '0':
            return edges
        v = pid + ':' + sc['name'] + '(' + sc['arguments'] + ')'
        v = v.replace(' ', '_')
        if u:
            edges.append([u, v])
        u = v
        if sc['name'] == 'fork' or sc['name'] == 'clone':
            pid_child = sc['return']
            if glob(rp_path + 'strace' + pid_child + '.json'):
                edges.extend(scg(rp_path, pid_child, u))
    return edges


def extract_graph(path):
    file_name = path.split('/')[-2].split('_')[0]
    graph_path = 'data/scg/' + file_name + '.adjlist'
    kc_path = 'data/scg_kc/' + file_name + '.adjlist'
    if glob(kc_path):
        return 1

    edges = []
    sc_paths = glob(path + 'strace*.json')
    sc_paths.sort()
    if sc_paths:
        pid = sc_paths[0].split('strace')[-1].split('.json')[0]
        edges = scg(path, pid, None)

    if edges:
        G = nx.Graph()
        G.add_edges_from(edges)
        G = nx.relabel_nodes(G, mapping=dict(zip(G, range(len(G)))))
        nx.write_adjlist(G, kc_path)
        return 1
    return 0


output = Parallel(n_jobs=N_JOBS, verbose=50)(
    delayed(extract_graph)(path) for path in RP_PATHS)
print(f'Generated {sum(output)} SCGs.')
