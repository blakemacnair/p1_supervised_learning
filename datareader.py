import pickle

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

MAX_TRANSFORM_DATA = 100_000
RANDOM_STATE = 42

STUDY_PATH_TEMPLATE = "models/{}/{}_study"
PREPROCCESSING_PATH_TEMPLATE = "models/{}/{}_preprocessing"
OUTPUT_PATH_TEMPLATE = "output/{}/{}/{}.png"


def get_dataset_train_test(dataset_name,
                           train_size=0.6,
                           random_state=None,
                           data_preprocessor=None):
    if dataset_name == "heart":
        x, y = get_heart_xy(data_preprocessor)
    elif dataset_name == "credit_card":
        x, y = get_credit_card_xy(data_preprocessor)
    else:
        assert False

    return train_test_split(x,
                            y,
                            train_size=train_size,
                            random_state=random_state,
                            stratify=y)


def sample(x, y, n_rows_keep=None):
    if n_rows_keep is None:
        n_rows_keep = MAX_TRANSFORM_DATA // x.shape[1]

    ss = StratifiedShuffleSplit(train_size=n_rows_keep,
                                random_state=RANDOM_STATE)
    fit_ind, _ = list(ss.split(x, y))[0]
    return x[fit_ind], y[fit_ind]


def get_heart_xy(data_preprocessor=None) -> (np.ndarray, np.ndarray):
    heart_data = pd.read_csv('data/heart_cleveland_upload.csv')
    x = heart_data.drop(columns=['condition']).to_numpy()
    y = heart_data['condition'].to_numpy()
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    if data_preprocessor is not None:
        try:
            x = data_preprocessor.transform(x)
        except NotFittedError:
            x_fit, y_fit = x, y

            # Decrease sample size if required so we don't run out of machine memory
            if np.prod(x.shape) > MAX_TRANSFORM_DATA:
                x_fit, y_fit = sample(x, y)

            data_preprocessor.fit(x_fit, y_fit)
            x = data_preprocessor.transform(x)

    return x, y


def get_credit_card_xy(data_preprocessor=None) -> (np.ndarray, np.ndarray):
    credit_card_data = pd.read_csv('data/creditcard.csv')
    x = credit_card_data.drop(columns=['Time', 'Class']).to_numpy()
    y = credit_card_data['Class'].to_numpy()
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    x, y = sample(x, y, 100_000)

    if data_preprocessor is not None:
        try:
            x = data_preprocessor.transform(x)
        except NotFittedError:
            x_fit, y_fit = x, y

            # Decrease sample size if required so we don't run out of machine memory
            if np.prod(x.shape) > MAX_TRANSFORM_DATA:
                x_fit, y_fit = sample(x, y)

            data_preprocessor.fit(x_fit, y_fit)
            x = data_preprocessor.transform(x)

    return x, y


def save_study(study, clf_name, dataset_name):
    with open(STUDY_PATH_TEMPLATE.format(clf_name, dataset_name), "wb") as f:
        pickle.dump(study, f)


def save_preprocessor(preprocessor, clf_name, dataset_name):
    with open(PREPROCCESSING_PATH_TEMPLATE.format(clf_name, dataset_name), "wb") as f:
        pickle.dump(preprocessor, f)


def load_study(clf_name, dataset_name):
    with open(STUDY_PATH_TEMPLATE.format(clf_name, dataset_name), "rb") as f:
        study = pickle.load(f)

    return study


def load_preprocessor(clf_name, dataset_name):
    with open(PREPROCCESSING_PATH_TEMPLATE.format(clf_name, dataset_name), "rb") as f:
        study = pickle.load(f)

    return study


def save_figure(fig, clf_name, dataset_name, file_name):
    fig.savefig(OUTPUT_PATH_TEMPLATE.format(clf_name, dataset_name, file_name))
