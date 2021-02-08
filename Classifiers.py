import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from imblearn.over_sampling import SMOTE


ALGO = ['DT', 'KNN', 'SVM', 'RF']
CLF_NAME = {
    'DT': 'Decision Tree',
    'KNN': 'K-Nearest Neighbors',
    'SVM': 'SVM',
    'RF': 'Random Forest'
}
CLASSIFIERS = {
    'DT': DecisionTreeClassifier(random_state=2020, class_weight="balanced"),
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'SVM': SVC(kernel='rbf', random_state=2020, class_weight="balanced", probability=True),
    'RF': RandomForestClassifier(random_state=2020, class_weight="balanced", n_jobs=-1)
}
HYPER_GRID = {
    'DT': {"criterion": ["gini", "entropy"]},
    'KNN': {"n_neighbors": [5, 100, 500], "weights": ["uniform", "distance"]},
    'SVM': {"C": np.logspace(-2, 2, 5), "gamma": np.logspace(-2, 2, 5)},
    'RF': {"n_estimators": [10, 100, 1000]},
}

COLORS = ['purple', 'orange', 'green', 'red']


class Classifiers:
    def __init__(self, algo=ALGO, model_path=None):
        self.model_path = model_path
        self.clf = {}
        for clf_name in algo:
            self.clf[clf_name] = GridSearchCV(
                CLASSIFIERS[clf_name],
                HYPER_GRID[clf_name],
                cv=5,
                n_jobs=-1
            )

    def run(self, X_train, X_test, y_train, y_test):
        ftsl = SelectFromModel(
            LinearSVC(penalty="l1", dual=False, random_state=2020).fit(X_train, y_train), prefit=True)
        X_train = ftsl.transform(X_train)
        X_test = ftsl.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if self.model_path:
            dump(ftsl, self.model_path + 'ftsl.joblib')
            dump(scaler, self.model_path + 'scaler.joblib')

        print(X_train.shape, X_test.shape)

        roc_auc = {}
        for clf_name, color in zip(self.clf, COLORS):
            print(clf_name)
            self.clf[clf_name].fit(X_train, y_train)

            if self.model_path:
                dump(ftsl, self.model_path + 'ftsl.joblib')

            y_pred = self.clf[clf_name].predict(X_test)
            y_prob = self.clf[clf_name].predict_proba(X_test)[:, 1]

            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cnf_matrix.ravel()
            FPR = FP / (FP + TN)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
            auc = metrics.roc_auc_score(y_test, y_prob)

            roc_auc[CLF_NAME[clf_name]] = [auc, fpr, tpr]

            print(metrics.classification_report(y_test, y_pred, digits=4))
            print(cnf_matrix)
            print('FPR: %.4f' % FPR)
            print('ROC AUC: %.4f' % auc)
            print('-' * 80)

        self.draw_roc(roc_auc)

    def draw_roc(self, roc_auc):
        roc_auc = dict(sorted(roc_auc.items(), key=lambda k: k[1][0]))
        for name, color in zip(roc_auc, COLORS):
            auc, fpr, tpr = roc_auc[name]
            plt.plot(fpr, tpr, color=color, marker='.',
                     label="%s (AUC = %0.4f)" % (name, auc))
            plt.plot([0, 1], [0, 1], "b--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("1-Specificity(False Positive Rate)")
            plt.ylabel("Sensitivity(True Positive Rate)")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            plt.savefig(f"result/roc.png", dpi=300)
