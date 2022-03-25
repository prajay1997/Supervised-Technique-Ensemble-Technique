###################### Solution of Coca_datasets Ensemble Methods#############

import pandas as pd 
import numpy as np

# load the data 

data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Ensemble Technique\datasets\Coca_Rating_Ensemble.xlsx")

data.drop(["REF", "Review"], axis =1, inplace = True)

# Univariate Analysis of the dataset
data.shape
data.columns
a = data.describe()


# changing the rating column float to int

data['Rating'] = data['Rating'].astype(int)

data["Rating"].unique()
data["Rating"].value_counts()

# Data cleaning the Company column
print(data['Company'].unique())
data['Company'].replace('Na�ve','Naive',inplace=True)
print(data['Company'].unique())

# Data cleaning the Company_Location column
print(data['Company_Location'].unique())
data['Company_Location'].replace({'Domincan Republic':'Dominican Republic','Niacragua':'Nicaragua','Eucador':'Ecuador'},inplace=True)
print(data['Company_Location'].unique())

# Data cleaning the Bean Type column
print(data['Bean_Type'].unique())

data['Bean_Type'].replace('Forastero (Arriba) ASSS', 'Forastero',inplace=True)
data['Bean_Type'].replace('Forastero (Arriba) ASS', 'Forastero',inplace=True)
data['Bean_Type'].replace('Forastero (Arriba)', 'Forastero',inplace=True)
data['Bean_Type'].replace('Forastero (Nacional)', 'Forastero',inplace=True)
data['Bean_Type'].replace('Criollo, +','Criollo',inplace=True)
data['Bean_Type'].replace('Blend-Forastero,Criollo','Blend',inplace=True)
data['Bean_Type'].replace('Forastero(Arriba, CCN)','Forastero',inplace=True)
data['Bean_Type'].replace('Forastero (Amelonado)','Forastero',inplace=True)
data['Bean_Type'].replace('Trinitario, Nacional','Trinitario',inplace=True)
data['Bean_Type'].replace('Trinitario (Amelonado)','Trinitario',inplace=True)
data['Bean_Type'].replace('Trinitario, TCGA','Trinitario',inplace=True)
data['Bean_Type'].replace('Criollo (Amarru)','Criollo',inplace=True)
data['Bean_Type'].replace('Criollo, Trinitario','Blend',inplace=True)
data['Bean_Type'].replace('Criollo (Porcelana)','Criollo',inplace=True)
data['Bean_Type'].replace('Trinitario (85% Criollo)','Blend',inplace=True)
data['Bean_Type'].replace('Forastero (Catongo)','Forastero',inplace=True)
data['Bean_Type'].replace('Forastero (Parazinho)','Forastero',inplace=True)
data['Bean_Type'].replace('Trinitario, Criollo','Blend',inplace=True)
data['Bean_Type'].replace('Criollo (Ocumare)','Criollo',inplace=True)
data['Bean_Type'].replace('Criollo (Ocumare 61)','Criollo',inplace=True)
data['Bean_Type'].replace('Criollo (Ocumare 77)','Criollo',inplace=True)
data['Bean_Type'].replace('Criollo (Ocumare 67)','Criollo',inplace=True)
data['Bean_Type'].replace('Criollo (Wild)','Criollo',inplace=True)
data['Bean_Type'].replace('Trinitario, Forastero','Blend',inplace=True)
data['Bean_Type'].replace('Trinitario (Scavina)','Trinitario',inplace=True)
data['Bean_Type'].replace('Criollo, Forastero','Blend',inplace=True)
data['Bean_Type'].replace('Forastero, Trinitario','Blend',inplace=True)
data.replace('\xa0','Unkown',inplace=True)
data['Bean_Type'].replace(np.nan,'Unkown',inplace=True)

print (data['Bean_Type'].unique())

def bean_type(bean):
    if bean =='Unkown':
        return 'Unkown'
    elif bean == 'Criollo':
        return 'Criollo'
    elif bean == 'Trinitario':
        return 'Trinitario'
    elif bean == 'Forastero':
        return 'Forastero'
    elif bean =='Blend':
        return 'Blend'
    else:
        return 'Other'

data['Bean_Type']=data['Bean_Type'].map(bean_type)
data['Bean_Type'].unique()
data['Bean_Type'].value_counts()

# Data cleaning the Bean Origin column

print(data['Origin'].unique())
data['Origin'] = data['Origin'].replace('Domincan Republic', 'Dominican Republic')
data['Origin'] = data['Origin'].replace('Carribean(DR/Jam/Tri)', 'Carribean')
data['Origin'] = data['Origin'].replace('Trinidad-Tobago', 'Trinidad, Tobago')
data['Origin'] = data['Origin'].replace("Peru, Mad., Dom. Rep.", "Peru, Madagascar, Dominican Republic")
data['Origin'] = data['Origin'].replace("Central and S. America", "Central and South America")
data['Origin'] = data['Origin'].replace("PNG, Vanuatu, Mad", "Papua New Guinea, Vanuatu, Madagascar")
data['Origin'] = data['Origin'].replace("Ven., Trinidad, Mad.", "Venezuela, Trinidad, Madagascar")
data['Origin'] = data['Origin'].replace("Ven.,Ecu.,Peru,Nic.", "Venezuela, Ecuador, Peru, Nicaragua")
data['Origin'] = data['Origin'].replace("Ven, Trinidad, Ecuador","Venezuela, Trinidad, Ecuador")
data['Origin'] = data['Origin'].replace("Ghana, Domin. Rep", "Ghana, Dominican Republic")
data['Origin'] = data['Origin'].replace("Ecuador, Mad., PNG","Ecuador, Madagascar, Papua New Guinea")
data['Origin'] = data['Origin'].replace("Mad., Java, PNG","Madagascar, Java, Papua New Guinea")
data['Origin'] = data['Origin'].replace("Gre., PNG, Haw., Haiti, Mad", "Grenada, Papua New Guinea, Hawaii, Haiti, Madagascar")

print (data['Origin'].unique())
data['Origin'].value_counts()

print(data['Name'].unique())
data['Name'] = data['Name'].map(lambda x: x.split(',')[0])
data['Name'] = data['Name'].map(lambda x: x.split('/')[0])
data['Name'] = data['Name'].map(lambda x: x.split('*')[0])
data['Name'] = data['Name'].map(lambda x: x.split('.')[0])
data['Name'] = data['Name'].map(lambda x: x.split('+')[0])
data['Name'] = data['Name'].map(lambda x: x.split(';')[0])
data['Name'] = data['Name'].map(lambda x: x.split('-')[0])
data['Name'] = data['Name'].map(lambda x: x.split('(')[0])
data['Name'] = data['Name'].map(lambda x: x.split('#')[0])
data['Name'] = data['Name'].map(lambda x: x.split('1')[0])
data['Name'] = data['Name'].map(lambda x: x.split('2')[0])
data['Name'] = data['Name'].map(lambda x: x.split('3')[0])
data['Name'] = data['Name'].map(lambda x: x.split('4')[0])
data['Name'] = data['Name'].map(lambda x: x.split('5')[0])
data['Name'] = data['Name'].map(lambda x: x.split('6')[0])
data['Name'] = data['Name'].map(lambda x: x.split('7')[0])
print (data['Name'].unique())
# checking the null value

data.replace({'Unkown':np.nan,},inplace=True)

data.isnull().sum()

data.Bean_Type.fillna(data.Bean_Type.mode()[0], inplace=True)
data.Origin.fillna(data.Origin.mode()[0], inplace = True)

data.isnull().sum()


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data.columns
data["Company"] = label.fit_transform(data["Company"])
data["Company_Location"] = label.fit_transform(data['Company_Location'])
data['Bean_Type'] = label.fit_transform(data['Bean_Type'])
data['Origin'] = label.fit_transform(data['Origin'])
data['Name'] = label.fit_transform(data['Name'])

y = data['Rating']


# Normalization of the data 

def norm_fun(i):
    x= (i-i.min())/(i.max()-i.min())
    return(x)

data_norm = norm_fun(data)
data_norm.describe()



# spliiting the data into train and test datasets

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(data_norm,y , test_size = 0.2, random_state =10)

######################  Bagging ######################33

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

# 99% accuracy

# Evalution of the model on train dataset

confusion_matrix(Y_train, bag_clf.predict(X_train))
accuracy_score(Y_train, bag_clf.predict(X_train))

#  99.9% 

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
## 98% accuracy
# Evaluation on training datasets

confusion_matrix(Y_train, ada_clf.predict(X_train))
accuracy_score(Y_train, ada_clf.predict(X_train))

# 98 % accuracy

############################# Gradient boosting##############################333

from sklearn.ensemble import GradientBoostingClassifier

gd_clf = GradientBoostingClassifier(learning_rate = 0.02, n_estimators =400) 


gd_clf.fit(X_train, Y_train)

# Evaluate the model on test datasets
pd.crosstab(Y_test, gd_clf.predict(X_test))
accuracy_score(Y_test, gd_clf.predict(X_test))

# Hyperparameters
gd_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 100, max_depth = 2)
gd_clf2.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(Y_test, gd_clf2.predict(X_test))
accuracy_score(Y_test, gd_clf2.predict(X_test))

# 95% accuracy

# Evaluation on Training Data
accuracy_score(Y_train, gd_clf2.predict(X_train))

# 98% accuracy

######################### XGBoosting###############################

import xgboost as xgb


xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 100, learning_rate = 0.3, n_jobs = -1)

# Fit the model in the traiing datasets

xgb_clf.fit(X_train, Y_train)


# Evaluation on Testing Data
confusion_matrix(Y_test, xgb_clf.predict(X_test))
accuracy_score(Y_test, xgb_clf.predict(X_test))
# 99%
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
# 100 %
accuracy_score(Y_train, cv_xg_clf.predict(X_train))

# 99 % accuracy

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
# 98% accuracy

pred_train = clf.predict(X_train)
accuracy_score(Y_train,pred_train)
# 100%
#################################
 # another code of stacking
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

# 98 % accuracy

# Evaluation on Training Data
confusion_matrix(Y_train, model.predict(X_train))
accuracy_score(Y_train, model.predict(X_train))
# 100%
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
voting.fit(X_train,Y_train)

# predict the most voted class
 hard_prediction = voting.predict(X_test)

# Accuracy of the hard voting

print('Hard voting', accuracy_score(Y_test, hard_prediction))
# 97%
print('Hard Voting:', accuracy_score(Y_train, voting.predict(X_train)))
#  99%

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
# 96%

print('Soft Voting:', accuracy_score(Y_train, voting1.predict(X_train)))
# 98%
