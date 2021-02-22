import pickle

import optuna
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from datareader import get_dataset_train_test, RANDOM_STATE, save_preprocessor, save_study

TRAIN_SIZE = 0.4
N_TRIALS = 100


def generate_generic_study(dataset_name,
                           objective_fn,
                           clf_name,
                           n_trials=N_TRIALS,
                           scoring=make_scorer(roc_auc_score),
                           train_size=TRAIN_SIZE,
                           data_preprocessor=None):
    x, _, y, _ = get_dataset_train_test(dataset_name,
                                        train_size=TRAIN_SIZE,
                                        random_state=RANDOM_STATE,
                                        data_preprocessor=data_preprocessor)

    def objective(trial):
        return objective_fn(trial, x, y, train_size, scoring)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    save_study(study, clf_name, dataset_name)

    if data_preprocessor is not None:
        save_preprocessor(data_preprocessor, clf_name, dataset_name)


def generate_heart_study(objective_fn,
                         clf_name,
                         n_trials=N_TRIALS,
                         scoring=make_scorer(roc_auc_score),
                         train_size=TRAIN_SIZE,
                         data_preprocessor=None):
    generate_generic_study("heart",
                           objective_fn,
                           clf_name,
                           n_trials,
                           scoring,
                           train_size,
                           data_preprocessor)


def generate_credit_card_study(objective_fn,
                               clf_name,
                               n_trials=N_TRIALS,
                               scoring=make_scorer(roc_auc_score),
                               train_size=TRAIN_SIZE,
                               data_preprocessor=None):
    generate_generic_study("credit_card",
                           objective_fn,
                           clf_name,
                           n_trials,
                           scoring,
                           train_size,
                           data_preprocessor)
