from glob import glob
import json
import pandas as pd
import os


def scg(fpath, pid, u):
    edges = []
    path = f'{fpath}strace{pid}.json'
    with open(path, 'r') as f:
        data = json.load(f)
    for sc in data:
        if sc['name'] == 'execve' and sc['return'] != '0':
            return edges
        node_id = sc['name'] + sc['arguments'] + pid
        nodes[node_id] = nodes.get(node_id, len(nodes))
        v = nodes[node_id]
        if u:
            edges.append([u, v])
        u = v
        if sc['name'] == 'fork':
            pidf = sc['return']
            if os.path.exists(f'{fpath}strace{pidf}.json'):
                edgesf = scg(fpath, pidf, u)
                edges.extend(edgesf)
    return edges


file_list = glob(
    '/media/server1/data/vsandbox_report/final_report_malware/*/')
file_list.extend(
    glob('/media/server1/data/vsandbox_report/final_report_benign/*/'))

labels = []
for file_path in file_list:
    file_name = file_path.split('/')[-2].split('_')[0]
    nodes = {}
    edges = []
    sc_paths = glob(file_path + 'strace*.json')
    sc_paths.sort()
    if sc_paths:
        pid = sc_paths[0].split('strace')[-1].split('.json')[0]
        edges = scg(file_path, pid, None)

    if edges:
        G = {'edges': edges}
        with open(f'graph/{file_name}.json', 'w') as f:
            json.dump(G, f)
        labels.append({
            'file': file_name,
            'label': 1 if 'malware' in file_path else 0
        })

pd.DataFrame(labels).to_csv('data/label.csv', index=None)
