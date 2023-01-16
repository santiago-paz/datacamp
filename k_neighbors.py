from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

flowers = pd.read_csv('flowers.csv')

X = flowers.drop('species', axis=1).values
y = flowers['species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))
