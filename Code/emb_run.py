import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from plotting import plot_confusion_matrix

# load data
X_train = np.load('../Data/nolocDoc2Vec/%s.npy' % 'xtrain')
X_test = np.load('../Data/nolocDoc2Vec/%s.npy' % 'xtest')
y_train = np.load('../Data/nolocDoc2Vec/%s.npy' % 'ytrain')
y_test = np.load('../Data/nolocDoc2Vec/%s.npy' % 'ytest')

# instantiate classifiers...
hyp = {'n_estimators': [1], 'max_depth': [1, 3]}
# hyp = {'n_estimators': [10, 300, 500],
       # 'max_depth': [None, 1, 3, 10, 50, 100]}

rf = RandomForestClassifier(n_jobs=-1, verbose=2)


gridcv = GridSearchCV(rf, hyp, verbose=2)

# train and evaluate them
gridcv.fit(X_train, y_train)
bestest = gridcv.best_estimator_

score = bestest.score(X_test, y_test)
print "SCORE: ", score
print "BEST ESTIMATOR: ", bestest

preds = bestest.predict(X_test)
cnfmat = confusion_matrix(y_test, preds,
                          labels=np.arange(np.max(y_train) + 1))

conf_file = '../Figures/RF_EMB_CONF_MTX_' +\
            time.strftime("%Y%m%d-%H%M%S") + '.png'

plot_confusion_matrix(cnfmat, 'RF', 'EMB', conf_file,
                      'emoji', False)
