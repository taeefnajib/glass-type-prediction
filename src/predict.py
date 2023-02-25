# importing all dependencies
from sklearn.metrics import accuracy_score


# predict test data
def predict(model, X_test):
    return model.predict(X_test)


# get accuracy
def get_acc(y_pred, y_true):
    return accuracy_score(y_true, y_pred)
