import pickle

import optuna
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

TRAIN_SIZE = 0.6
RANDOM_STATE = 42
N_TRIALS = 100

from datareader import get_credit_card_train_test, get_heart_train_test


def generate_studies(objective_fn, clf_name, n_trials=N_TRIALS):
    generate_heart_study(objective_fn, clf_name, n_trials)
    generate_credit_card_study(objective_fn, clf_name, n_trials)


def generate_generic_study(objective_fn,
                           clf_name,
                           dataset_name,
                           n_trials=N_TRIALS,
                           scoring=make_scorer(roc_auc_score),
                           percent_sample=1.0,
                           data_preprocessor=None):
    x, y = [], []
    if dataset_name == "credit_card":
        x, _, y, _ = get_credit_card_train_test(train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    elif dataset_name == "heart":
        x, _, y, _ = get_heart_train_test(train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

    if percent_sample < 1:
        ss = StratifiedShuffleSplit(train_size=percent_sample,
                                    random_state=RANDOM_STATE)

        sample_ind, _ = list(ss.split(x, y))[0]
        x = x[sample_ind]
        y = y[sample_ind]

    def objective(trial):
        return objective_fn(trial, x, y, scoring)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with open("models/{}/{}_study".format(clf_name, dataset_name), "wb") as f:
        pickle.dump(study, f)


def generate_heart_study(objective_fn, clf_name, n_trials=N_TRIALS, scoring=make_scorer(roc_auc_score)):
    generate_generic_study(objective_fn, clf_name, "heart", n_trials, scoring)


def generate_credit_card_study(objective_fn,
                               clf_name,
                               n_trials=N_TRIALS,
                               scoring=make_scorer(roc_auc_score),
                               percent_sample=1.0):
    generate_generic_study(objective_fn, clf_name, "credit_card", n_trials, scoring, percent_sample)
