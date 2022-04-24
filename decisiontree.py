from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pandas as pd

iris = pd.read_csv('Iris.csv')
iris.drop('Id', axis=1,inplace=True)
iris.head()

x = iris[['PanjangSepalCm', 'LebarSepalCm','PanjangPetalCm','LebarPetalCm']]
y= iris['Spesies']

model = DecisionTreeClassifier()
model.fit(x,y)

model.predict([[6.2,3.4,5.4,2.3]])

