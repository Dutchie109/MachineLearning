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
plt.show()
