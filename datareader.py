import pandas as pd


def load_heart_data() -> pd.DataFrame:
    return pd.read_csv('data/heart_cleveland_upload.csv')


def load_credit_card_data() -> pd.DataFrame:
    return pd.read_csv('data/creditcard.csv')
