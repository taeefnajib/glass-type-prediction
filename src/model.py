# importing all dependencies
from sklearn.neighbors import KNeighborsClassifier


# train and fit model
def fit_model(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    return knn.fit(X_train, y_train)
