import pickle

import optuna
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from datareader import get_credit_card_xy, get_heart_xy


def objective_heart(trial):
    x, y = get_heart_xy()
    return objective(trial, x, y, make_scorer(roc_auc_score))


def objective_credit_card(trial):
    x, y = get_credit_card_xy()
    return objective(trial, x, y, make_scorer(roc_auc_score))


def objective(trial, x, y, scoring=None):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 6, 32)
    min_samples_split = trial.suggest_float("min_samples_split", 0.01, 0.1)
    min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.01, 0.1)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0, 0.1)
    ccp_alpha = trial.suggest_float("ccp_alpha", 0, 0.1)

    dt = DecisionTreeClassifier(criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_impurity_decrease=min_impurity_decrease,
                                ccp_alpha=ccp_alpha,
                                class_weight="balanced",
                                random_state=42)

    score = cross_val_score(dt, x, y, n_jobs=-1, cv=5, scoring=scoring)
    return score.mean()


def generate_dt_studies():
    generate_dt_heart_study()
    generate_dt_credit_card_study()


def generate_dt_heart_study():
    heart_study = optuna.create_study(direction="maximize")
    heart_study.optimize(objective_heart, n_trials=100, show_progress_bar=True)

    with open("models/dt/heart_study", "wb") as f:
        pickle.dump(heart_study, f)


def generate_dt_credit_card_study():
    credit_card_study = optuna.create_study(direction="maximize")
    credit_card_study.optimize(objective_credit_card, n_trials=100, show_progress_bar=True)

    with open("models/dt/credit_card_study", "wb") as f:
        pickle.dump(credit_card_study, f)


def load_dt_models():
    return load_dt_heart_model(), load_dt_credit_card_model()


def load_dt_heart_model():
    with open("models/dt/heart_study", "rb") as f:
        study = pickle.load(f)

    heart_best_params = study.best_params
    return DecisionTreeClassifier(
        criterion=heart_best_params["criterion"],
        max_depth=heart_best_params["max_depth"],
        min_samples_split=heart_best_params["min_samples_split"],
        min_samples_leaf=heart_best_params["min_samples_leaf"],
        min_impurity_decrease=heart_best_params["min_impurity_decrease"],
        ccp_alpha=heart_best_params["ccp_alpha"],
        class_weight="balanced",
        random_state=42)


def load_dt_credit_card_model():
    with open("models/dt/credit_card_study", "rb") as f:
        study = pickle.load(f)

    heart_best_params = study.best_params
    return DecisionTreeClassifier(
        criterion=heart_best_params["criterion"],
        max_depth=heart_best_params["max_depth"],
        min_samples_split=heart_best_params["min_samples_split"],
        min_samples_leaf=heart_best_params["min_samples_leaf"],
        min_impurity_decrease=heart_best_params["min_impurity_decrease"],
        ccp_alpha=heart_best_params["ccp_alpha"],
        class_weight="balanced",
        random_state=42)


if __name__ == "__main__":
    # generate_dt_studies()
    heart_dt, credit_card_dt = load_dt_models()

