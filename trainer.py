import pickle

import optuna
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from datareader import get_dataset_train_test, RANDOM_STATE, save_preprocessor, save_study

TRAIN_SIZE = 0.4
N_TRIALS = 100


def generate_studies(objective_fn,
                     clf_name,
                     n_trials=N_TRIALS,
                     scoring=make_scorer(roc_auc_score),
                     heart_preprocessor=None,
                     credit_card_preprocessor=None):
    generate_heart_study(objective_fn,
                         clf_name,
                         n_trials,
                         scoring,
                         data_preprocessor=heart_preprocessor)
    generate_credit_card_study(objective_fn,
                               clf_name,
                               n_trials,
                               scoring,
                               data_preprocessor=credit_card_preprocessor)


def generate_generic_study(dataset_name,
                           objective_fn,
                           clf_name,
                           n_trials=N_TRIALS,
                           scoring=make_scorer(roc_auc_score),
                           percent_sample=1.0,
                           data_preprocessor=None):
    x, _, y, _ = get_dataset_train_test(dataset_name,
                                        train_size=TRAIN_SIZE,
                                        random_state=RANDOM_STATE,
                                        data_preprocessor=data_preprocessor)

    if percent_sample < 1:
        ss = StratifiedShuffleSplit(train_size=percent_sample,
                                    random_state=RANDOM_STATE)

        sample_ind, _ = list(ss.split(x, y))[0]
        x = x[sample_ind]
        y = y[sample_ind]

    def objective(trial):
        return objective_fn(trial, x, y, scoring)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    save_study(study, clf_name, dataset_name)

    if data_preprocessor is not None:
        save_preprocessor(data_preprocessor, clf_name, dataset_name)


def generate_heart_study(objective_fn,
                         clf_name,
                         n_trials=N_TRIALS,
                         scoring=make_scorer(roc_auc_score),
                         percent_sample=1.0,
                         data_preprocessor=None):
    generate_generic_study("heart",
                           objective_fn,
                           clf_name,
                           n_trials,
                           scoring,
                           percent_sample,
                           data_preprocessor)


def generate_credit_card_study(objective_fn,
                               clf_name,
                               n_trials=N_TRIALS,
                               scoring=make_scorer(roc_auc_score),
                               percent_sample=1.0,
                               data_preprocessor=None):
    generate_generic_study("credit_card",
                           objective_fn,
                           clf_name,
                           n_trials,
                           scoring,
                           percent_sample,
                           data_preprocessor)
