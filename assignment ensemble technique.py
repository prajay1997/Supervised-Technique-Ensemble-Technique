################## Q1) Solution of Dibeted_ensemble ##################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


# Load the dataset

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Ensemble Technique\datasets\Diabeted_Ensemble.csv")

data.shape
data.columns
a = data.describe()
data.isnull().sum()
data.duplicated().sum()


# standarddization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_norm = scaler.fit_transform(data.iloc[:,0:8])

data['Class'].unique()
data['Class'].value_counts()

# splitting the data into train and test data 

X_train, X_test, Y_train, Y_test = train_test_split(data_norm, data.Class, test_size = 0.2)

#######################################################################################
                          
                        ####### GridSearchCV #############

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=50)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, Y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(Y_test, cv_rf_clf_grid.predict(X_test))
accuracy_score(Y_test, cv_rf_clf_grid.predict(X_test))

# Evaluation on Training Data
confusion_matrix(Y_train, cv_rf_clf_grid.predict(X_train))
accuracy_score(Y_train, cv_rf_clf_grid.predict(X_train))
###########################################################################

######################### voting #################################

from sklearn.ensemble import VotingClassifier

# Instantiate the learners (classifiers)

learner1 = GaussianNB(priors=None, var_smoothing=1e-09)
learner2 = KNeighborsClassifier(n_neighbors=5)
learner3 = DecisionTreeClassifier(criterion = 'entropy', random_state=50)

# Instantiate the voting Classifier
voting = VotingClassifier([("Naive_bayes",learner1),
                ("KNN",learner2),
                ("DT", learner3)])
# Fit Classifier in the training data
voting.fit(X_train,Y_train)

# predict the most voted class
 hard_prediction = voting.predict(X_test)

# Accuracy of the hard voting

print('Hard voting', accuracy_score(Y_test, hard_prediction))
print('Hard Voting:', accuracy_score(Y_train, voting.predict(X_train)))

## Soft Voting
# Instantiate the learners( classifiers)
learner4 = GaussianNB(priors=None, var_smoothing=1e-09)
learner5 = KNeighborsClassifier(n_neighbors=5)
learner6 = DecisionTreeClassifier(criterion = 'entropy',max_depth=4, random_state=50)

# Instantiate the voting Classifier
voting1 = VotingClassifier([("Naive_bayes",learner4),
                ("KNN",learner5),
                ("DT", learner6)],
                 voting = 'soft')

# fit classifier in the train data

voting1.fit(X_train,Y_train)
learner4.fit(X_train,Y_train)
learner5.fit(X_train,Y_train)
learner6.fit(X_train,Y_train)

# predict the most voted class
soft_predictions = voting1.predict(X_test) 

# Get the base learner predictions
predictions4 = learner4.predict(X_test)
predictions5 = learner5.predict(X_test)
predictions6 = learner6.predict(X_test)

# Accuracies of base learners
print('L4:', accuracy_score(Y_test, predictions4))
print('L5:', accuracy_score(Y_test, predictions5))
print('L6:', accuracy_score(Y_test, predictions6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(Y_test, soft_predictions))

print('Soft Voting:', accuracy_score(Y_train, voting1.predict(X_train)))

#####################################################################################

#####################################Stacking################################# 

# Create the ensemble's base learners and meta learner
# Append base learners to a list
base_learners = []

# naive bayes classifier model
naive_bayes = GaussianNB()
base_learners.append(naive_bayes)


# KNN classifier model 
knn = KNeighborsClassifier(n_neighbors=5)
base_learners.append(knn)

# Decision Tree model 

DT = DecisionTreeClassifier(criterion ="entropy", max_depth =4, random_state = 500)
base_learners.append(DT)

# Meta model using Knn classifier
meta_learner = KNeighborsClassifier(n_neighbors=3)

# Create the cross-validation folds
KF = KFold(n_splits = 5)
meta_index = 0
for train_indices, test_indices in KF.split(X_train):
    # Train each learner on the K-1 folds and create meta data for the Kth fold
    for i in range(len(base_learners)):
        learner = base_learners[i]

        learner.fit(X_train[train_indices], Y_train[train_indices])
        predictions = learner.predict_proba(X_train[test_indices])[:,0]

        meta_data[i][meta_index:meta_index+len(test_indices)] = predictions

    meta_targets[meta_index:meta_index+len(test_indices)] = Y_train[test_indices]
    meta_index += len(test_indices)


# Transpose the meta data to be fed into the meta learner
meta_data = meta_data.transpose()

# Create the meta data for the test set and evaluate the base learners
test_meta_data = np.zeros((len(base_learners), len(test_x)))
base_acc = []

for i in range(len(base_learners)):
    learner = base_learners[i]
    learner.fit(train_x, train_y)
    predictions = learner.predict_proba(test_x)[:,0]
    test_meta_data[i] = predictions

    acc = metrics.accuracy_score(test_y, learner.predict(test_x))
    base_acc.append(acc)
    
test_meta_data = test_meta_data.transpose()

# Fit the meta learner on the train set and evaluate it on the test set
meta_learner.fit(meta_data, meta_targets)
ensemble_predictions = meta_learner.predict(test_meta_data)

acc = metrics.accuracy_score(test_y, ensemble_predictions)

# Print the results
for i in range(len(base_learners)):
    learner = base_learners[i]

    print(f'{base_acc[i]:.2f} {learner.__class__.__name__}')
    
print(f'{acc:.2f} Ensemble')

##################################################################################################
##########################################  Bagging ######################################

from sklearn import tree
clftree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = clftree , n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 52)

bag_clf.fit(X_train, Y_train)

# Evaluation on Testing Data
confusion_matrix(Y_test, bag_clf.predict(X_test))
accuracy_score(Y_test, bag_clf.predict(X_test))test)

# Evaluation on Training Data
confusion_matrix(Y_train, bag_clf.predict(Y_train))
accuracy_score(Y_train, bag_clf.predict(X_train))

#######################################################################################

###############################Boosting###################################

               ################ Adaboost  ###################

from sklearn.ensemble import AdaBoostRegressor

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 500)

ada_clf.fit(X_train, Y_train)

# Evaluate the model on test datasets

confusion_matrix(Y_test, ada_clf.predict(X_test))
accuracy_score(Y_test, ada_clf.predict(X_test))

# Evaluate the model on train datasets

confusion_matrix(Y_train, ada_clf.predict(X_train))
accuracy_score(Y_train, ada_clf.predict(X_train))

                    ###### Gradient Boosting #########
                    
from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(X_train, Y_train)

# Evaluate the model on test datasets
confusion_matrix(Y_test, boost_clf.predict(X_test))
accuracy_score(Y_test, boost_clf.predict(X_test))

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 100, max_depth = 2)
boost_clf2.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(Y_test, boost_clf2.predict(X_test))
accuracy_score(Y_test, boost_clf2.predict(X_test))

# Evaluation on Training Data
accuracy_score(Y_train, boost_clf2.predict(X_train))



                         ############ XGboost##################33

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

# n_jobs – Number of parallel threads used to run xgboost.
# learning_rate (float) – Boosting learning rate (xgb’s “eta”)


xgb_clf.fit(X_train, Y_train)


# Evaluation on Testing Data
confusion_matrix(Y_test, xgb_clf.predict(X_test))
accuracy_score(Y_test, xgb_clf.predict(X_test))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, Y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(Y_test, cv_xg_clf.predict(X_test))
grid_search.best_params_


#######################

from numpy import mean
from numpy import std

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot

X = X_train
y = Y_train
# get a list of models to evaluate
def get_models():
	models = dict()
	models['knn'] = KNeighborsClassifier()
	models['DT'] = DecisionTreeClassifier()
	models['bayes'] = GaussianNB()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

"""
>lr 0.759 (0.034)
>knn 0.700 (0.050)
>cart 0.694 (0.059)
>svm 0.745 (0.039)
>bayes 0.750 (0.043)
"""
# make a prediction with a stacking ensemble
from sklearn.ensemble import StackingClassifier
# define dataset
# define the base models
level0 = list()
level0.append(('knn', KNeighborsClassifier()))
level0.append(('DT', DecisionTreeClassifier()))
level0.append(('bayes', GaussianNB()))
# define meta learner model
level1 = KNeighborsClassifier()
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# fit the model on all available data
model.fit(X, y)

# make a prediction on test data
pred_test = model.predict(X_test)

# Evaluation on Testing Data
confusion_matrix(Y_test, pred_test)
accuracy_score(Y_test,pred_test)

# 66 % accuracy

# Evaluation on Training Data
confusion_matrix(Y_train, model.predict(X_train))
accuracy_score(Y_train, model.predict(X_train))

# 82 % Accuracy


################################# END ###########################################



