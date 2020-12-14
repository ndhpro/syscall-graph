import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from imblearn.over_sampling import SMOTE

import logging
import coloredlogs
coloredlogs.install()

CLASSIFIERS = {
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(random_state=2020, class_weight="balanced"),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM': SVC(random_state=2020, class_weight="balanced"),
    'RF': RandomForestClassifier(random_state=2020, class_weight="balanced", n_jobs=-1)
}

HYPER_GRID = {
    'NB': {},
    'DT': {"criterion": ["gini", "entropy"]},
    'KNN': {"n_neighbors": [5, 100, 500], "weights": ["uniform", "distance"]},
    'SVM': {"C": np.logspace(-2, 2, 5), "gamma": np.logspace(-2, 2, 5)},
    'RF': {"n_estimators": [10, 100, 1000]},
}

HEADERS = ["Classifier", "Accuracy", "ROC AUC", "FPR", "Precision", "Recall",
           "F1-score", "TP", "FP", "TN", "FN"]
COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]


class Classifiers:
    def __init__(self):
        self.scaler = Normalizer()
        self.clf = {}
        for clf_name in CLASSIFIERS:
            self.clf[clf_name] = GridSearchCV(
                CLASSIFIERS[clf_name],
                HYPER_GRID[clf_name],
                cv=5,
                n_jobs=-1
            )

    def train(self, X_train, y_train):
        self.fs = SelectFromModel(
            LinearSVC(penalty="l1", dual=False, random_state=2020).fit(X_train, y_train), prefit=True)
        X_train = self.fs.transform(X_train)
        X_train = self.scaler.fit_transform(X_train)

        for clf_name in self.clf:
            logging.info(f'Training {clf_name}...')
            self.clf[clf_name].fit(X_train, y_train)

    def test(self, X_test, y_test):
        X_test = self.fs.transform(X_test)
        X_test = self.scaler.transform(X_test)

        report = []
        for clf_name in self.clf:
            y_pred = self.clf[clf_name].predict(X_test)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cnf_matrix.ravel()
            FPR = FP / (FP + TN)
            try:
                roc_auc = round(100 * metrics.roc_auc_score(y_test, y_pred), 2)
            except Exception as e:
                logging.error(e)
                roc_auc = None
            row = [
                str(clf_name),
                round(100 * metrics.accuracy_score(y_test, y_pred), 2),
                roc_auc,
                round(100 * FPR, 2),
                round(100 * metrics.precision_score(y_test, y_pred), 2),
                round(100 * metrics.recall_score(y_test, y_pred), 2),
                round(100 * metrics.f1_score(y_test, y_pred), 2),
            ]
            row.extend([TP, FP, TN, FN])
            report.append(row)

        report = pd.DataFrame(report, columns=HEADERS)
        return report

    def save_model(self):
        dump(self.fs, 'model/fs.joblib')
        dump(self.scaler, 'model/scaler.joblib')
        for clf_name in self.clf:
            dump(self.clf[clf_name], f'model/{clf_name}.joblib')

    def load_model(self):
        self.fs = load('model/fs.joblib')
        self.scaler = load('model/scaler.joblib')
        for clf_name in self.clf:
            self.clf[clf_name] = load(f'model/{clf_name}.joblib')
