import pickle

import optuna
from sklearn.metrics import make_scorer, roc_auc_score

TRAIN_SIZE = 0.6
RANDOM_STATE = 42
N_TRIALS = 100

from datareader import get_credit_card_train_test, get_heart_train_test


def generate_studies(objective_fn, clf_name, n_trials=N_TRIALS):
    generate_heart_study(objective_fn, clf_name, n_trials)
    generate_credit_card_study(objective_fn, clf_name, n_trials)


def generate_generic_study(objective_fn, clf_name, dataset_name, n_trials=N_TRIALS):
    if dataset_name == "credit_card":
        x, x_t, y, y_t = get_credit_card_train_test(train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    elif dataset_name == "heart":
        x, x_t, y, y_t = get_heart_train_test(train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

    def objective(trial):
        return objective_fn(trial, x, y, make_scorer(roc_auc_score))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    with open("models/{}/{}_study".format(clf_name, dataset_name), "wb") as f:
        pickle.dump(study, f)


def generate_heart_study(objective_fn, clf_name, n_trials=N_TRIALS):
    generate_generic_study(objective_fn, clf_name, "heart", n_trials)


def generate_credit_card_study(objective_fn, clf_name, n_trials=N_TRIALS):
    generate_generic_study(objective_fn, clf_name, "credit_card", n_trials)
