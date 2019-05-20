print("My first Machine Learning example. Yah!")

#Taken from https://code.visualstudio.com/docs/editor/debugging#_launch-configurations

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Dimensions of the dataset.
# shape
print("Shape of dataset: ", dataset.shape)

# Display the first 20 rows of the dataset
# head
print(dataset.head(20))

# Statistical summary of all attributes.
# descriptions
print(dataset.describe())

#Breakdown of the data by the class variable
# class distribution
print(dataset.groupby('class').size())

# Univariate plots

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# histograms
dataset.hist()
# plt.show()

# scatter plot matrix
scatter_matrix(dataset)
# plt.show()

# Evaluate Some Algorithms
# Now it is time to create some models of the data and estimate their accuracy on unseen data.
# -- Separate out a validation dataset.
# -- Set-up the test harness to use 10-fold cross validation.
# -- Build 5 different models to predict species from flower measurements
# -- Select the best model.

# Split-out validation dataset; hold out 20% of the values as test data
# See https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# print(X)

# Test Harness
# We will use 10-fold cross validation to estimate accuracy. 
# This will split our dataset into 10 parts, train on 9 and test on 1 
# and repeat for all combinations of train-test splits.

# Test options and evaluation metric
# See https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# See https://machinelearningmastery.com/randomness-in-machine-learning/

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# See https://machinelearningmastery.com/make-predictions-scikit-learn/

# Some changes, to see what Git does!