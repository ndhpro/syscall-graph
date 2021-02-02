from Graph2vec import Graph2vec
from Classifiers import Classifiers
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


stats = pd.read_csv('data/stats.csv')
