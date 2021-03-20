import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from imblearn.over_sampling import SMOTE


SEED = 42
ALGO = ['KNN', 'DT', 'RF', 'SVM']
CLF_NAME = {
    'KNN': 'K-Nearest Neighbors',
    'DT': 'Decision Tree',
    'RF': 'Random Forest',
    'SVM': 'SVM',
}
CLASSIFIERS = {
    'KNN': KNeighborsClassifier(n_jobs=-1),
    'DT': DecisionTreeClassifier(random_state=SEED),
    'RF': RandomForestClassifier(random_state=SEED, n_jobs=-1),
    'SVM': SVC(random_state=SEED, probability=True),
}
HYPER_GRID = {
    'KNN': {"n_neighbors": [10, 100, 1000], "weights": ["uniform", "distance"]},
    'DT': {"criterion": ["gini", "entropy"], "splitter": ["best", "random"]},
    'RF': {"n_estimators": [10, 100, 1000]},
    'SVM': {"C": np.logspace(-1, 1, 3), "gamma": np.logspace(-1, 1, 3)},

}

COLORS = ['purple', 'orange', 'green', 'red']


class Classifiers:
    def __init__(self, algo=ALGO):
        self.clf = {}
        for clf_name in algo:
            self.clf[clf_name] = GridSearchCV(
                CLASSIFIERS[clf_name],
                HYPER_GRID[clf_name],
                cv=5,
                n_jobs=-1
            )

    def run(self, X_train, X_test, y_train, y_test):
        report = f'Original shape: {X_train.shape} {X_test.shape}\n'

        smote = SMOTE(random_state=SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        ftsl = SelectFromModel(
            LinearSVC(penalty="l1", dual=False, random_state=SEED).fit(X_train, y_train), prefit=True)
        X_train = ftsl.transform(X_train)
        X_test = ftsl.transform(X_test)

        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        report += f'Processed shape: {X_train.shape} {X_test.shape}\n'

        roc_auc = {}
        for clf_name in self.clf:
            print(CLF_NAME[clf_name] + '... ', end='', flush=True)
            self.clf[clf_name].fit(X_train, y_train)

            y_pred = self.clf[clf_name].predict(X_test)
            y_prob = self.clf[clf_name].predict_proba(X_test)[:, 1]

            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cnf_matrix.ravel()
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
            auc = metrics.roc_auc_score(y_test, y_prob)
            other_metrics = pd.DataFrame({
                'TPR': '%.4f' % TPR,
                'FPR': '%.4f' % FPR,
                'ROC AUC': '%.4f' % auc,
            }, index=[0]).to_string(col_space=9, index=False)
            roc_auc[CLF_NAME[clf_name]] = [auc, fpr, tpr]
            report += '-' * 80 + '\n'
            report += CLF_NAME[clf_name] + '\n'
            report += f'{metrics.classification_report(y_test, y_pred, digits=4)}\n'
            report += f'{cnf_matrix}\n\n'
            report += f'{other_metrics}\n'

            print('%.4f' % metrics.accuracy_score(y_test, y_pred))

        roc = self.draw_roc(roc_auc)
        return report, roc

    def draw_roc(self, roc_auc):
        roc_auc = dict(sorted(roc_auc.items(), key=lambda k: k[1][0]))
        plt.figure()
        for name, color in zip(roc_auc, COLORS):
            auc, fpr, tpr = roc_auc[name]
            plt.plot(fpr, tpr, color=color, marker=',',
                     label="%s (AUC = %0.4f)" % (name, auc))
            plt.plot([0, 1], [0, 1], "b--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("1-Specificity(False Positive Rate)")
            plt.ylabel("Sensitivity(True Positive Rate)")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
        return plt
