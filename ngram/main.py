import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from Classifiers import Classifiers
from joblib import dump, load

import logging
import coloredlogs
coloredlogs.install()

N_GRAM = 1

logging.info('Loading data')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
new = pd.read_csv('data/new.csv')

train_path = [f'corpus/{i}.txt' for i in train.values[:, 0]]
test_path = [f'corpus/{i}.txt' for i in test.values[:, 0]]
new_path = [f'corpus/{i}.txt' for i in new.values[:, 0]]

train_label = train.values[:, 1].astype('int')
test_label = test.values[:, 1].astype('int')
new_label = new.values[:, 1].astype('int')

logging.info('Vectorizing data')
vec = CountVectorizer(input='filename', ngram_range=[N_GRAM, N_GRAM])
train_vec = vec.fit_transform(train_path).toarray()
test_vec = vec.transform(test_path).toarray()
new_vec = vec.transform(new_path).toarray()

dump(vec, f'model/vec_{N_GRAM}-gram.joblib')

logging.info('Classifying')
clf = Classifiers()
clf.train(train_vec, train_label)
clf.save_model()

result = clf.test(test_vec, test_label)
logging.info(result)
result.to_csv(f'result/{N_GRAM}-gram.csv')

result = clf.test(new_vec, new_label)
logging.info(result)
result.to_csv(f'result/new_{N_GRAM}-gram.csv')
