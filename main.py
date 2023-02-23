# importing all dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "data.csv"
    test_size: float = 0.2
    random_state: int = 6
    n_neighbors: int = 3

hp = Hyperparameters

# get data
def get_data(filepath):
    return pd.read_csv(filepath)

# clean data
def clean_data(df):
    df = df.drop(["Unnamed: 0.1","Unnamed: 0"], axis=1)
    return df

# split dataset
def split_data(df, test_size, random_state):
    X = df.drop(["Type"], axis=1)
    y = df["Type"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# train and fit model
def fit_model(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors)
    return knn.fit(X_train, y_train)


# define workflow
def run_wf(hp: Hyperparameters):
    train_df = get_data(hp.filepath)
    train_df = clean_data(df=train_df)
    X_train, X_test, y_train, y_test = split_data(df=train_df, test_size=hp.test_size, random_state=hp.random_state)
    return fit_model(X_train=X_train, y_train=y_train, n_neighbors=hp.n_neighbors)

if __name__=="__main__":
    run_wf(hp=hp)