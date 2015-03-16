"""

Methods to apply:
    Random Forest


TODO: apply some kind of cut
TODO: mean subtraction and std division
TODO: save!
TODO: train test split on cutouts, not the data augmented
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier
from scipy.stats import randint as sp_randint  # for searching for best hyperparameters!
from sklearn.grid_search import GridSearchCV

# this is something like 4.5 gigs of memory!
project_dir = '/Users/cpd/Projects/strongcnn/'
datapath = '/Users/cpd/Desktop/batches/'

data = pd.read_csv(datapath + 'database.csv',
                       index_col=522)

# load up catalog for cutouts
cat = pd.read_csv(project_dir + 'catalog/cluster_catalog.csv')

print('data loaded')
# we also need to filter out entries who are of type test which are
# indeterminate
data = data[data['category'] == 'training']
cat = cat[cat['category'] == 'training']

# extract features
feature_cols = ['nn{0}'.format(i) for i in xrange(500)]
X = data[feature_cols].values

# now we have several ways of choosing y from our data
# let us just go with alpha = 1 or not (whether the cutout is a simulated lens or not)
y = data['alpha'] == 1

print(sum(y))
print(X.shape)


# get cuts on cutoutname for stage1, stage2, and all
cut_train, cut_test = train_test_split(cat['cutoutname'].values, test_size=0.2)
cut_train_1, cut_test_1 = train_test_split(cat[cat['stage'] == 1]['cutoutname'].values, test_size=0.2)
cut_train_2, cut_test_2 = train_test_split(cat[cat['stage'] == 2]['cutoutname'].values, test_size=0.2)

"""
Need:
    X_test, X_train
    y_test, y_train
for all, _1, _2
"""

cuts = {'all': (cut_train, cut_test),
        1: (cut_train_1, cut_test_1),
        2: (cut_train_2, cut_test_2),}

train = {# 'all': (data.loc[cut_train][feature_cols].values,
         #         data.loc[cut_train]['alpha'] == 1),
         1: (data[data['stage'] == 1].loc[cut_train_1][feature_cols].values,
             data[data['stage'] == 1].loc[cut_train_1]['alpha'] == 1),
         2: (data[data['stage'] == 2].loc[cut_train_2][feature_cols].values,
             data[data['stage'] == 2].loc[cut_train_2]['alpha'] == 1),
        }
test = {# 'all': (data.loc[cut_test][feature_cols].values,
        #          data.loc[cut_test]['alpha'] == 1),
         1: (data[data['stage'] == 1].loc[cut_test_1][feature_cols].values,
             data[data['stage'] == 1].loc[cut_test_1]['alpha'] == 1),
         2: (data[data['stage'] == 2].loc[cut_test_2][feature_cols].values,
             data[data['stage'] == 2].loc[cut_test_2]['alpha'] == 1),
        }


mult_dict = {'dud': 0, 'lensed quasar': 1, 'lensing cluster': 2, 'lensed galaxy': 3}
# y_train_mult = np.array([y_train[i] * mult_dict[data.loc[cut_train]['flavor'][i]] for i in xrange(len(y_train))])

classifiers = {
    'RandomForest': RandomForestClassifier,
    'Softmax': SGDClassifier,
    'SVM': SGDClassifier,}
classifier_kwrgs = {
    'RandomForest': dict(n_estimators=50, verbose=1, n_jobs=-1),
    'Softmax': dict(loss='log', verbose=1, n_jobs=-1,
        n_iter = 1000),
    'SVM': dict(loss='hinge', verbose=1, n_jobs=-1,
        n_iter = 1000)
    }
parameter_searches = {
    'Softmax': {'alpha': 10.0**-np.arange(1,7)},
    'SVM': {'alpha': 10.0**-np.arange(1,7)},
    }

fpr = {}
tpr = {}
thresh = {}
scores = {}

for classifier in classifiers:
    for stage in train:
        print('Training of {0}, stage {1} starting'.format(classifier, stage))
        kwrgs = classifier_kwrgs[classifier].copy()
        clf = classifiers[classifier](**kwrgs)
        if classifier in parameter_searches:
            # do grid search
            param_grid = parameter_searches[classifier]
            if 'n_iter' in kwrgs:
                # don't need to do full iterations to get which hyperparam best
                n_iter_old = kwrgs['n_iter']
                kwrgs['n_iter'] = 100
            grid_search = GridSearchCV(classifiers[classifier](**kwrgs), param_grid=param_grid)
            grid_search.fit(*train[stage])
            # update kwrgs with best fit of grid search
            clf = grid_search.best_estimator_
            if 'n_iter' in kwrgs:
                if stage == 1:
                    clf.n_iter = 10000
                elif stage == 2:
                    clf.n_iter = 2000
        # train the data
        clf.fit(*train[stage])
        print('Training of {0}, stage {1} complete'.format(classifier, stage))
        # get probs from trained results and test too
        key_train = (classifier, stage, 'train')
        key_test = (classifier, stage, 'test')
        if classifier == 'SVM':
            scores[key_train] = clf.decision_function(train[stage][0])
            scores[key_test] = clf.decision_function(test[stage][0])
        else:
            scores[key_train] = clf.predict_proba(train[stage][0])[:, 1]
            scores[key_test] = clf.predict_proba(test[stage][0])[:, 1]
        fpr[key_train], tpr[key_train], thresh[key_train] = roc_curve(train[stage][1], scores[key_train])
        fpr[key_test], tpr[key_test], thresh[key_test] = roc_curve(test[stage][1], scores[key_test])


# now compare against the mean_probability
# this isn't totally correct because the classifications are done by subject,
# and here subjects are now represented multiple numbers of times unevenly
y_test_sw = cat['kind'] == 'sim'
y_score_sw = cat['mean_probability']

cond_sw1 = cat['stage'] == 1
cond_sw2 = cat['stage'] == 2

y_test_sw1 = y_test_sw[cond_sw1]
y_score_sw1 = y_score_sw[cond_sw1]
y_test_sw2 = y_test_sw[cond_sw2]
y_score_sw2 = y_score_sw[cond_sw2]

# this is somewhat unfair because sw comes from the image,
# not the cutout
fpr_sw1, tpr_sw1, _ = roc_curve(y_test_sw1, y_score_sw1)
fpr_sw2, tpr_sw2, _ = roc_curve(y_test_sw2, y_score_sw2)

classifier_colors = {'RandomForest': 'r', 
                     'Softmax': 'g',
                     'SVM': 'm'}
testtrain_style = {'test': '--',
                   'train': ':'}

for stage in [1, 2]:
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if (stage == 1):
        plt.plot(fpr_sw1, tpr_sw1, linestyle='-', color='Blue', linewidth=3, label='SpaceWarps Stage 1')
    if (stage == 2):
        plt.plot(fpr_sw2, tpr_sw2, linestyle='-', color='DarkOrange', linewidth=3, label='SpaceWarps Stage 2')
    for testtrain in ['test', 'train']:
        for classifier in classifiers:
            key = (classifier, stage, testtrain)
            label = '{0} Stage {1} {2}'.format(*key)
            plt.plot(fpr[key], tpr[key], 
                     linestyle=testtrain_style[testtrain], 
                     color=classifier_colors[classifier],
                     linewidth=3, label=label)
    plt.legend(loc='lower right')
    plt.xlim(-0.005, 0.5)
    plt.ylim(0.5, 1.0)
    plt.savefig(project_dir + 'doc/roc_curve_stage_{0}.pdf'.format(stage))


# # for the scores, compare now against the spacewarps scores
# for stage in [1, 2]:
#     plt.figure()
#     plt.xlabel('Space Warps Probability')
#     plt.ylabel('Classifier Probability')
#     for key in scores:
#         if key[-1] == 'test':
#             x = data[data['stage'] == stage].loc[cuts[stage][1]]['mean_probability'].values
#         elif key[-1] == 'train':
#             x = data[data['stage'] == stage].loc[cuts[stage][0]]['mean_probability'].values
# 
#         y = scores[key]
#         label = '{0} Classifier Stage {1} {2}ing set'.format(*key)
#         plt.plot(x, y, '.', label=label)
# plt.legend(loc='lower right')
# plt.show()


bindir = project_dir + 'binarys/'
np.save(bindir + 'fpr', fpr)
np.save(bindir + 'tpr', tpr)
np.save(bindir + 'cuts', cuts)
np.save(bindir + 'scores', scores)
np.save(bindir + 'thresh', thresh)

import ipdb; ipdb.set_trace()
