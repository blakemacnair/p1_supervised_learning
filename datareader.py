import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_heart_train_test(train_size=0.6, random_state=None):
    x, y = get_heart_xy()

    return train_test_split(x,
                            y,
                            train_size=train_size,
                            random_state=random_state,
                            stratify=y)


def get_credit_card_train_test(train_size=0.6, random_state=None):
    x, y = get_credit_card_xy()

    return train_test_split(x,
                            y,
                            train_size=train_size,
                            random_state=random_state,
                            stratify=y)


def load_heart_data() -> pd.DataFrame:
    return pd.read_csv('data/heart_cleveland_upload.csv')


def get_heart_xy() -> (np.ndarray, np.ndarray):
    heart_data = load_heart_data()
    x = heart_data.drop(columns=['condition'])
    y = heart_data['condition']
    return x, y


def load_credit_card_data() -> pd.DataFrame:
    return pd.read_csv('data/creditcard.csv')


def get_credit_card_xy() -> (np.ndarray, np.ndarray):
    credit_card_data = load_credit_card_data()
    x = credit_card_data.drop(columns=['Time', 'Class'])
    y = credit_card_data['Class']
    return x, y
