# importing all dependencies
import yaml
import pickle
from src.data import *
from src.model import *
from src.predict import *

# read yaml file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# define workflow
def run_wf(data_filepath, test_size, random_state, n_neighbors, model_filepath):
    train_df = get_data(data_filepath)
    train_df = clean_data(df=train_df)
    X_train, X_test, y_train, y_test = split_data(
        df=train_df, test_size=test_size, random_state=random_state
    )
    model = fit_model(X_train=X_train, y_train=y_train, n_neighbors=n_neighbors)
    pickle.dump(model, open(model_filepath, "wb"))
    return model


if __name__ == "__main__":
    run_wf(
        filepath=config["filepath"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        n_neighbors=config["n_neighbors"],
        model_filepath=config["model_filepath"],
    )
