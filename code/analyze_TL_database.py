"""

Methods to apply:
    Random Forest


TODO: apply some kind of cut
TODO: mean subtraction and std division
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# this is something like 4.5 gigs of memory!
databasepath = '/Users/cpd/Desktop/batches/database.csv'
data_all = pd.read_csv(databasepath)
print('data loaded')
# we also need to filter out entries who are of type test which are
# indeterminate
data = data_all[data_all['category'] == 'training']

# extract features
feature_cols = ['nn{0}'.format(i) for i in xrange(500)]
X = data[feature_cols].values

# now we have several ways of choosing y from our data
# let us just go with alpha = 1 or not (whether the cutout is a simulated lens or not)
y = data['alpha'] == 1

print(sum(y))

print(X.shape)
# train on svm and rf
clf_rf = RandomForestClassifier()
#scores_rf = cross_val_score(clf_rf, X, y)
#print('rf: ', scores_rf)

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2)
clf_rf.fit(X_train, y_train)
y_score =  clf_rf.predict_proba(X_test)
# predict_proba gives prob for both columns
# so we just want it for the column 2
y_score = y_score[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
print(clf_rf.score(X_test, y_test))
# make a plot on the features
plt.figure()
plt.xlabel('Feature Number')
plt.ylabel('Feature Importance')
plt.plot(clf_rf.feature_importances_, 'k--')


# repeat for stage1 and stage 2
X_1 = X[(data['stage'] == 1).values]
y_1 = y[data['stage'] == 1]
X_train_1, X_test_1, y_train_1, y_test_1, = train_test_split(X_1, y_1, test_size=0.2)
clf_rf_1 = RandomForestClassifier()
clf_rf_1.fit(X_train_1, y_train_1)
y_score_1 =  clf_rf.predict_proba(X_test_1)
y_score_1 = y_score_1[:, 1]
fpr_1, tpr_1, _ = roc_curve(y_test_1, y_score_1)

X_2 = X[(data['stage'] == 2).values]
y_2 = y[data['stage'] == 2]
X_train_2, X_test_2, y_train_2, y_test_2, = train_test_split(X_2, y_2, test_size=0.2)
clf_rf_2 = RandomForestClassifier()
clf_rf_2.fit(X_train_2, y_train_2)
y_score_2 =  clf_rf.predict_proba(X_test_2)
y_score_2 = y_score_2[:, 1]
fpr_2, tpr_2, _ = roc_curve(y_test_2, y_score_2)


# now compare against the mean_probability
# this isn't totally correct because the classifications are done by subject,
# and here subjects are now represented multiple numbers of times unevenly
y_test_sw = data['kind'] == 'sim'
y_score_sw = data['mean_probability']

cond_sw1 = data['stage'] == 1
cond_sw2 = data['stage'] == 2

y_test_sw1 = y_test_sw[cond_sw1]
y_score_sw1 = y_score_sw[cond_sw1]
y_test_sw2 = y_test_sw[cond_sw2]
y_score_sw2 = y_score_sw[cond_sw2]


# this is somewhat unfair because sw comes from the image,
# not the cutout
fpr_sw1, tpr_sw1, _ = roc_curve(y_test_sw1, y_score_sw1)
fpr_sw2, tpr_sw2, _ = roc_curve(y_test_sw2, y_score_sw2)


plt.figure()
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr, tpr, '-', color='Red', linewidth=3, label='Convnet Random Forest All')
plt.plot(fpr_1, tpr_1, '--', color='Blue', linewidth=3, label='Convnet Random Forest Stage 1')
plt.plot(fpr_2, tpr_2, '--', color='DarkOrange', linewidth=3, label='Convnet Random Forest Stage 2')
plt.plot(fpr_sw1, tpr_sw1, linestyle='-', color='Blue', linewidth=3, label='SpaceWarps Stage1')
plt.plot(fpr_sw2, tpr_sw2, linestyle='-', color='DarkOrange', linewidth=3, label='SpaceWarps Stage2')
plt.legend(loc='lower right')

plt.show()


# what are the number of objects with alpha = 1 etc?


import ipdb; ipdb.set_trace()
