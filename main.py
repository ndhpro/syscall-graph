from glob import glob
import os
import pandas as pd
import numpy as np
from Graph2vec import Graph2vec
from Classifiers import Classifiers

BENIGN_RP_PATH = glob(
    '/media/server1/data/vsandbox_report/final_report_benign/*/')


def get_graph_paths():
    benign = []
    for path in BENIGN_RP_PATH:
        hash_str = path.split('/')[-2].split('_')[0]
        if os.path.exists(f'graph/{hash_str}.json'):
            benign.append(hash_str)

    stats = pd.read_csv('data/stats.csv')
    bashlite = list(stats.loc[stats['family'] ==
                              'Gaygyt/BASHLITE (2014)', 'file_name'].values)
    mirai = list(stats.loc[stats['family'] ==
                           'Mirai (2016)', 'file_name'].values)
    tsunami = list(stats.loc[stats['family'] ==
                             'Tsunami/Kaiten (2013)', 'file_name'].values)
    print(len(bashlite), len(mirai), len(tsunami), len(benign))

    X_train = bashlite + mirai + benign[:2000]
    X_test = tsunami + benign[2000:]

    X_train = [f'graph/{x}.json' for x in X_train]
    X_test = [f'graph/{x}.json' for x in X_test]

    y_train = list(np.ones(len(mirai) + len(bashlite))) + \
        list(np.zeros(2000))
    y_test = list(np.ones(len(tsunami))) + list(np.zeros(len(benign) - 2000))
    return X_train, X_test, y_train, y_test


def graph2vec(X_train, X_test):
    graph2vec = Graph2vec()
    X_train = graph2vec.train(X_train)
    X_test = graph2vec.apply(X_test)

    X_train.to_csv('data/train.csv', index=None)
    X_test.to_csv('data/test.csv', index=None)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_graph_paths()

    graph2vec(X_train, X_test)

    classifiers = Classifiers()
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    X_train = train.loc[:, 'x_0':].values
    X_test = test.loc[:, 'x_0':].values

    classifiers.run(X_train, X_test, y_train, y_test)
