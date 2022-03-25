###################### Solution of Ensemble_password_strenghtdataset  #############

import pandas as pd 
import numpy as np
import seaborn as sns

# load the data 

data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Ensemble Technique\datasets\Ensemble_Password_Strength.xlsx")
# Univariate Analysis of the dataset
data.head()
data.shape
data.columns
data.describe()
data.isnull().sum()
data.duplicated().sum()
data.characters_strength.unique()
data.characters_strength.value_counts()

sns.countplot(data['characters_strength'])

# make the password and strength into tuples so I can access easily

pass_tup = np.array(data)

#Suffling randomly the data

import random
random.shuffle(pass_tup)

x = [labels[0] for labels in pass_tup]
y = [labels[1] for labels in pass_tup]

# Creating a function to split the character

def word_divider(words):
    char = []
    for i in words:
        char.append(i)
    return char

 word_divider('prajay@123')

 #  TF-IDF vectorizer to convert String data into numerical data
 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer= word_divider)

# convert all the rows to string object

corpus = [str(item) for item in x]

X = vectorizer.fit_transform(corpus)
X.shape

vectorizer.get_feature_names()

first_document_vector = X[0]
first_document_vector

first_document_vector.T.todense()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X ,y , test_size = 0.2, random_state =10)

############  Bagging ######################33

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
clf = tree.DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

bag_clf = BaggingClassifier(base_estimator = rf , n_estimators = 100,
                            bootstrap = True, n_jobs = 1, random_state = 50)

# fitting the model on train datasets
bag_clf.fit(X_train, Y_train)

# Evaluation on the test datasets

confusion_matrix(Y_test, bag_clf.predict(X_test))
accuracy_score(Y_test, bag_clf.predict(X_test))

# 95% accuracy

# Evalution of the model on train dataset

confusion_matrix(Y_train, bag_clf.predict(X_train))
accuracy_score(Y_train, bag_clf.predict(X_train))

# 98% accuracy

##############################################################################

######################### Boosting ###############################

######################## Adaboosting ################################3333

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 1000)

# fit the model on train datasets
ada_clf.fit(X_train, Y_train)

# Evaluation of model on test datastes

confusion_matrix(Y_test, ada_clf.predict(X_test))
accuracy_score(Y_test, ada_clf.predict(X_test))
## 87% accuracy

# Evaluation on training datasets

confusion_matrix(Y_train, ada_clf.predict(X_train))
accuracy_score(Y_train, ada_clf.predict(X_train))

# 87% accuracy

############################# Gradient boosting##############################333

from sklearn.ensemble import GradientBoostingClassifier

gd_clf = GradientBoostingClassifier(learning_rate = 0.02, n_estimators =400) 


gd_clf.fit(X_train, Y_train)

# Evaluate the model on test datasets
confusion_matrix(Y_test, gd_clf.predict(X_test))
accuracy_score(Y_test, gd_clf.predict(X_test))

# 93% accuracy

# Hyperparameters
gd_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 100, max_depth = 2)
gd_clf2.fit(X_train, Y_train)

# Evaluation on Testing Data
confusion_matrix(Y_test, gd_clf2.predict(X_test))
accuracy_score(Y_test, gd_clf2.predict(X_test))

# 86 % accuracy

# Evaluation on Training Data
accuracy_score(Y_train, gd_clf2.predict(X_train))

# 87% accuracy

######################### XGBoosting###############################

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 100, learning_rate = 0.3, n_jobs = -1)

# Fit the model in the traiing datasets

xgb_clf.fit(X_train, Y_train)


# Evaluation on Testing Data
confusion_matrix(Y_test, xgb_clf.predict(X_test))
accuracy_score(Y_test, xgb_clf.predict(X_test))
# 98%
accuracy_score(Y_train, xgb_clf.predict(X_train))
xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 50, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, Y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
confusion_matrix(Y_test, cv_xg_clf.predict(X_test))
accuracy_score(Y_test, cv_xg_clf.predict(X_test))
grid_search.best_params_

# 97 % accuracy

accuracy_score(Y_train, cv_xg_clf.predict(X_train))
# 99% accuracy

################################################################################################

################################ Stacking #####################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=10)),
    ('gbdt',GradientBoostingClassifier())
]

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(
    estimators=estimators, 
    final_estimator = DecisionTreeClassifier(),
    cv=10)

# fit the model on  train data

clf.fit(X_train, Y_train)

# evaluate the model on test data

pred_test = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,pred_test)
confusion_matrix(Y_test,pred_test)
# 92 % accuracy

############## Another code for stacking ##########

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
	models['RFC'] = RandomForestClassifier()
	models['knn'] = KNeighborsClassifier()
	models['gbc'] = GradientBoostingClassifier()
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


# make a prediction with a stacking ensemble
from sklearn.ensemble import StackingClassifier
# define dataset
# define the base models
level0 = list()
level0.append(('RFC', RandomForestClassifier()))
level0.append(('knn', KNeighborsClassifier()))
level0.append(('Gbc', GradientBoostingClassifier()))
# define meta learner model
level1 = DecisionTreeClassifier()
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# fit the model on all available data
model.fit(X, y)

# make a prediction on test data
pred_test = model.predict(X_test)

# Evaluation on Testing Data
confusion_matrix(Y_test, pred_test)
accuracy_score(Y_test,pred_test)

# 93 % accuracy

# Evaluation on Training Data
confusion_matrix(Y_train, model.predict(X_train))
accuracy_score(Y_train, model.predict(X_train))
# 99%
############################### Voting #################################

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
voting.fit(X_train.toarray(),Y_train)

# predict the most voted class
 hard_prediction = voting.predict(X_test.toarray())

# Accuracy of the hard voting

print('Hard voting', accuracy_score(Y_test, hard_prediction))
#93%
print('Hard Voting:', accuracy_score(Y_train, voting.predict(X_train.toarray())))
#  97%

################################ Soft Voting ###########################

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

voting1.fit(X_train.toarray(),Y_train)
learner4.fit(X_train.toarray(),Y_train)
learner5.fit(X_train.toarray(),Y_train)
learner6.fit(X_train.toarray(),Y_train)

# predict the most voted class
soft_predictions = voting1.predict(X_test.toarray()) 

# Get the base learner predictions
predictions4 = learner4.predict(X_test.toarray())
predictions5 = learner5.predict(X_test.toarray())
predictions6 = learner6.predict(X_test.toarray())

# Accuracies of base learners
print('L4:', accuracy_score(Y_test, predictions4))
print('L5:', accuracy_score(Y_test, predictions5))
print('L6:', accuracy_score(Y_test, predictions6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(Y_test, soft_predictions))
# 86%

print('Soft Voting:', accuracy_score(Y_train, voting1.predict(X_train.toarray())))
# 91%

