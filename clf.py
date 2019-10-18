import numpy as np
import matplotlib.pyplot as plt
import time
import logging

from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC


g = open('log/result.txt', mode='w+')

# Load data
X, y = load_npz('data/data.npz'), np.loadtxt('data/label.txt')
g.write('Size of data before Feature selection: ' + str(X.shape) + '\n')
g.write('Number of Malware samples: ' + str(X[y == 1].shape[0]) + '\n')
g.write('Number of Benign samples: ' + str(X[y == -1].shape[0]) + '\n')

# Feature selection
tick = time.time()
g.write('-' * 54 + '\n')
model = SelectFromModel(
    LinearSVC(penalty="l1", dual=False, random_state=42).fit(X, y), prefit=True)
X = model.transform(X)
g.write('Size of data after Feature selection: ' + str(X.shape) + '\n')
g.write('Feature selection done in %0.2f sec\n' % (time.time() - tick))
g.write('-' * 54 + '\n')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42)


names = ["RBF SVM", "Decision Tree", "Random Forest",
         "Bagging", "k-Nearest Neighbors"]

classifiers = [
    SVC(kernel='rbf', class_weight='balanced'),
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    BaggingClassifier(random_state=42),
    KNeighborsClassifier()
]

hyperparam = [
    {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-3, 3, 7)},
    {'criterion': ['gini', 'entropy']},
    {'criterion': ['gini', 'entropy'], 'n_estimators': [
        10, 100, 1000], 'bootstrap': [True, False]},
    {'n_estimators': [10, 100, 1000]},
    {'n_neighbors': [5, 100, 500], 'weights': ['uniform', 'distance']}
]

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

for name, est, color, hyper in zip(names, classifiers, colors, hyperparam):
    try:
        g.write(name + ':\n\n')
        print(name)
        clf = GridSearchCV(est, param_grid=hyper, cv=5, iid=False)

        tick = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - tick

        tick = time.time()
        y_true, y_pred = y_test, clf.predict(X_test)
        test_time = time.time() - tick

        g.write('Classification report:\n' +
                str(classification_report(y_true, y_pred)) + '\n')
        cnf_matrix = confusion_matrix(y_true, y_pred)
        g.write('Confusion matrix:\n' + str(cnf_matrix) + '\n\n')

        TN, FP, FN, TP = cnf_matrix.ravel()
        # FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        # TP = np.diag(cnf_matrix)
        # TN = cnf_matrix.sum() - (FP + FN + TP)
        # FP = FP.astype(float)
        # FN = FN.astype(float)
        # TP = TP.astype(float)
        # TN = TN.astype(float)
        fpr = FP / (FP + TN)

        g.write('Accuracy: %0.3f\n' % metrics.accuracy_score(y_true, y_pred))
        g.write('ROC AUC: %0.3f\n' % metrics.roc_auc_score(y_true, y_pred))
        g.write('False Positive Rate: %0.3f\n' % fpr)
        # with np.printoptions(precision=3, suppress=True):
        #     g.write('False Positive Rate for each class: ' + str(fpr) + '\n')
        g.write(
            'Hyperparameter tuning (with Grid search & 5-fold CV) done in %0.2f sec\n' % train_time)
        g.write('Testing done in %0.2f sec\n' % test_time)
        g.write('-' * 54 + '\n')

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, color=color,
                 label='%s ROC (area = %0.3f)' % (name, auc))
    except Exception as error:
        logging.error(error)

# display
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('img/roc')
plt.show()
