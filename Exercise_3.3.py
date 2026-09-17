#Exercise 3.3 Python SVM Iris Scatter Plot - Sepal length and width
from sklearn import svm, datasets
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
print(df.shape)
print(df.head(10))
print(df.tail(10))
print(df.describe())

#count NAN
print(df.isna().sum().sum())
#drop NAN values
df = df.dropna()
print(df.groupby('species').size())

# Scatter plot of the first two features: sepal length and sepal width
pyplot.scatter(df['sepal_length'], df['sepal_width'])
pyplot.xlabel('Sepal Length')
pyplot.ylabel('Sepal Width')
pyplot.title('Scatter Plot: Sepal Length vs Sepal Width')
pyplot.show()

X = df.values[:,:2]
s = df['species']
d = dict([(y,x) for x,y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]
clf = svm.SVC()
clf.fit(X, y)
#Predict the flower for a given Sepal length and width
p = clf.predict([[5.4, 3.2]])
print(p)
