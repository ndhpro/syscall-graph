from Graph2vec import Graph2vec
from Classifiers import Classifiers
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

import logging
import coloredlogs
coloredlogs.install()


# Load data
# labels = pd.read_csv('data/label.csv')
# X_mal = labels.loc[labels['label'] == 1, 'file'].values
# X_beg = labels.loc[labels['label'] == 0, 'file'].values

# X_path, y = [], []
# X_new_path = []
# stat = pd.read_csv('data/stats.csv')
# for x in tqdm(X_mal, desc='Split data'):
#     fam = stat.loc[stat['file_name'] == x, 'family'].values[0]
#     try:
#         year = int(fam.split()[-1][1:-1])
#         if year < 2016:
#             X_path.append(f'graph/{x}.json')
#             y.append(1)
#         else:
#             X_new_path.append(f'graph/{x}.json')
#     except:
#         continue

# X_path.extend([f'graph/{x}.json' for x in X_beg])
# y.extend([0] * len(X_beg))

# X_train_path, X_test_path, y_train, y_test = train_test_split(
#     X_path, y, test_size=0.3, random_state=2020, stratify=y)

# Run Graph2vec
# graph2vec = Graph2vec(dimensions=1024)

# X_train = graph2vec.train(X_train_path)
# X_train['label'] = y_train
# X_train.to_csv('data/X_train.csv', index=None)
# graph2vec.save_model()

# X_test = graph2vec.apply(X_test_path)
# X_test['label'] = y_test
# X_test.to_csv('data/X_test.csv', index=None)

# graph2vec.load_model('model/graph2vec.model')
# X_test_new = graph2vec.apply(X_new_path)
# X_test_new['label'] = [1] * len(X_test_new)
# X_test_new.to_csv('data/X_test_new.csv', index=None)

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
X_test_new = pd.read_csv('data/X_test_new.csv')

y_train = X_train.values[:, -1].astype('int')
X_train = X_train.values[:, 1:-1].astype('float')
y_test = X_test.values[:, -1].astype('int')
X_test = X_test.values[:, 1:-1].astype('float')
y_test_new = X_test_new.values[:, -1].astype('int')
X_test_new = X_test_new.values[:, 1:-1].astype('float')

# Run Classification
logging.info(f'{X_train.shape} {X_test.shape} {X_test_new.shape}')

clf = Classifiers()

# clf.train(X_train, y_train)
# clf.save_model()
clf.load_model()

report = clf.test(X_test, y_test)
logging.info(report)
report.to_csv('result/clf_result.csv', index=None)

report = clf.test(X_test_new, y_test_new)
logging.info(report)
report.to_csv('result/clf_result_new.csv', index=None)
