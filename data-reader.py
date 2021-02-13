import pandas as pd


def load_heart_data():
    return pd.read_csv('data/heart_cleveland_upload.csv')


def load_credit_card_data():
    return pd.read_csv('data/creditcard.csv')
