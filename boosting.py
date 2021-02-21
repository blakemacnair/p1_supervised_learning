import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from analysis import cross_validate_and_analyze
from datareader import get_dataset_train_test
from trainer import generate_credit_card_study, generate_heart_study, generate_studies, RANDOM_STATE, TRAIN_SIZE


def boosted_decision_tree(criterion,
                          max_depth,
                          min_samples_split,
                          min_samples_leaf,
                          min_impurity_decrease,
                          ccp_alpha,
                          n_estimators,
                          learning_rate):
    dt = DecisionTreeClassifier(criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_impurity_decrease=min_impurity_decrease,
                                ccp_alpha=ccp_alpha,
                                class_weight="balanced",
                                random_state=RANDOM_STATE)
    booster = AdaBoostClassifier(dt,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 random_state=RANDOM_STATE)
    return booster


def objective(trial, x, y, scoring=None):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_float("min_samples_split", 1e-3, 1e-1)
    min_samples_leaf = trial.suggest_float("min_samples_leaf", 1e-3, 1e-1)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 1e-10, 1e-1)
    ccp_alpha = trial.suggest_float("ccp_alpha", 1e-3, 5e-1)

    n_estimators = trial.suggest_int("n_estimators", 5, 50)
    learning_rate = trial.suggest_float("learning_rate", 1e-1, 1)

    clf = boosted_decision_tree(criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_impurity_decrease=min_impurity_decrease,
                                ccp_alpha=ccp_alpha,
                                n_estimators=n_estimators,
                                learning_rate=learning_rate)

    score = cross_val_score(clf, x, y, n_jobs=-1, cv=5, scoring=scoring)
    return score.mean()


def generate_boosting_studies():
    generate_studies(objective, "boosting")


def load_boosting_models():
    return load_boosting_heart_model(), load_boosting_credit_card_model()


def load_boosting_heart_model():
    with open("models/boosting/heart_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return boosted_decision_tree(criterion=best_params["criterion"],
                                 max_depth=best_params["max_depth"],
                                 min_samples_split=best_params["min_samples_split"],
                                 min_samples_leaf=best_params["min_samples_leaf"],
                                 min_impurity_decrease=best_params["min_impurity_decrease"],
                                 ccp_alpha=best_params["ccp_alpha"],
                                 n_estimators=best_params["n_estimators"],
                                 learning_rate=best_params["learning_rate"])


def load_boosting_credit_card_model():
    with open("models/boosting/credit_card_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return boosted_decision_tree(criterion=best_params["criterion"],
                                 max_depth=best_params["max_depth"],
                                 min_samples_split=best_params["min_samples_split"],
                                 min_samples_leaf=best_params["min_samples_leaf"],
                                 min_impurity_decrease=best_params["min_impurity_decrease"],
                                 ccp_alpha=best_params["ccp_alpha"],
                                 n_estimators=best_params["n_estimators"],
                                 learning_rate=best_params["learning_rate"])


if __name__ == "__main__":
    # generate_heart_study(objective, "boosting", 200)
    # generate_credit_card_study(objective, "boosting", 50, percent_sample=0.1)

    heart_boosting, credit_card_boosting = load_boosting_models()

    # h_x_train, h_x_test, h_y_train, h_y_test = get_dataset_train_test("heart",
    #     train_size=TRAIN_SIZE,
    #     random_state=RANDOM_STATE)
    # cross_validate_and_analyze(
    #     heart_boosting,
    #     h_x_train,
    #     h_x_test,
    #     h_y_train,
    #     h_y_test,
    #     name="Heart Disease",
    #     labels=["No Disease", "Disease"],
    #     scoring=make_scorer(roc_auc_score)
    # )

    cc_x_train, cc_x_test, cc_y_train, cc_y_test = get_dataset_train_test("credit_card",
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE)
    cross_validate_and_analyze(
        credit_card_boosting,
        cc_x_train,
        cc_x_test,
        cc_y_train,
        cc_y_test,
        name="Credit Card Fraud",
        labels=["No Fraud", "Fraud"],
        scoring=make_scorer(roc_auc_score)
    )
