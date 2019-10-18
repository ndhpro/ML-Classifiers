import numpy as np
import matplotlib.pyplot as plt
import time
import logging

from scipy.sparse import load_npz, vstack
from sklearn import metrics, utils
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2


g = open('log/result_oc.txt', mode='w+')

# Load data
X, y = load_npz('data/data.npz'), np.loadtxt('data/label.txt')
g.write('Size of data before Feature selection: ' + str(X.shape) + '\n')
g.write('Number of Malware samples: ' + str(X[y == 1].shape[0]) + '\n')
g.write('Number of Benign samples: ' + str(X[y == -1].shape[0]) + '\n')
g.write('-' * 54 + '\n')

# Feature selection
X = StandardScaler(with_mean=False).fit_transform(X)
X = SelectKBest(chi2, k=1024).fit_transform(X, y)
g.write('Size of data after Feature selection: ' + str(X.shape) + '\n')
g.write('-' * 54 + '\n')

# Split data
X_mal = X[y == 1]
X_beg = X[y == -1]
utils.shuffle(X_mal)
utils.shuffle(X_beg)
X_train, X_mal_test = train_test_split(X_mal, test_size=.3, random_state=42)
y_train = np.ones(X_train.shape[0])


X_test = vstack([X_mal_test, X_beg])
y_test = np.hstack([np.ones(X_mal_test.shape[0]), -np.ones(X_beg.shape[0])])

# Apply
clf = OneClassSVM(nu=.1, gamma='scale')
# clf = LocalOutlierFactor(novelty=True, contamination=.2)
clf.fit(X_train)
y_true, y_pred = y_test, clf.predict(X_test)

# Report
g.write('Classification report:\n' +
        str(classification_report(y_true, y_pred)) + '\n')

cnf_matrix = confusion_matrix(y_true, y_pred)
g.write('Confusion matrix:\n' + str(cnf_matrix) + '\n\n')
g.write('Accuracy: %0.3f\n' % metrics.accuracy_score(y_true, y_pred))
g.write('ROC AUC: %0.3f\n' % metrics.roc_auc_score(y_true, y_pred))

TN, FP, FN, TP = cnf_matrix.ravel()
fpr = FP / (FP + TN)
g.write('False Positive Rate: %0.3f\n' % fpr)

# display
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
auc = metrics.roc_auc_score(y_true, y_pred)
plt.plot(fpr, tpr, color='red',
         label='%s ROC (area = %0.3f)' % ('OC-SVM', auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
