# importing all dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

values_to_rename = {"'build wind non-float'":"build wind non-float",
                    "'build wind float'":"build wind float",
                    "headlamps":"headlamps",
                    "'vehic wind float'":"vehic wind float",
                    "containers":"containers",
                    "tableware":"tableware"
                   }

# read yaml file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# get data
def get_data(filepath):
    return pd.read_csv(filepath)


# clean data
def clean_data(df):
    df = df.drop(config["cols_to_drop"], axis=1)
    df["Type"] = df["Type"].replace(values_to_rename)
    return df


# split dataset
def split_data(df, test_size, random_state):
    X = df.drop(["Type"], axis=1)
    y = df["Type"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
