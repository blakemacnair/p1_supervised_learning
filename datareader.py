import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_heart_train_test(train_size=0.6, random_state=None, data_preprocessor=None):
    x, y = get_heart_xy(data_preprocessor)

    return train_test_split(x,
                            y,
                            train_size=train_size,
                            random_state=random_state,
                            stratify=y)


def get_credit_card_train_test(train_size=0.6, random_state=None, data_preprocessor=None):
    x, y = get_credit_card_xy(data_preprocessor)

    return train_test_split(x,
                            y,
                            train_size=train_size,
                            random_state=random_state,
                            stratify=y)


def load_heart_data() -> pd.DataFrame:
    return pd.read_csv('data/heart_cleveland_upload.csv')


def get_heart_xy(data_preprocessor=None) -> (np.ndarray, np.ndarray):
    heart_data = load_heart_data()
    x = heart_data.drop(columns=['condition']).to_numpy()
    if data_preprocessor is not None:
        x = data_preprocessor(x)

    y = heart_data['condition'].to_numpy()
    return x, y


def load_credit_card_data() -> pd.DataFrame:
    return pd.read_csv('data/creditcard.csv')


def get_credit_card_xy(data_preprocessor=None) -> (np.ndarray, np.ndarray):
    credit_card_data = load_credit_card_data()
    x = credit_card_data.drop(columns=['Time', 'Class']).to_numpy()
    if data_preprocessor is not None:
        x = data_preprocessor(x)

    y = credit_card_data['Class'].to_numpy()
    return x, y
