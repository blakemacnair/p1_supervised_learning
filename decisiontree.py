import pickle
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, roc_auc_score, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier

from analysis import plot_clf_analysis
from trainer import generate_credit_card_study, generate_heart_study, generate_studies, RANDOM_STATE, TRAIN_SIZE

from datareader import get_credit_card_train_test, get_heart_train_test


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
                                random_state=RANDOM_STATE)

    score = cross_val_score(dt, x, y, n_jobs=-1, cv=5, scoring=scoring)
    return score.mean()


def generate_dt_studies():
    generate_studies(objective, "dt")


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
        random_state=RANDOM_STATE)


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
        random_state=RANDOM_STATE)


if __name__ == "__main__":
    # Generate studies with optimal hyper-parameters based on training split
    # generate_dt_studies()

    # Load models from optimized studies
    heart_dt, credit_card_dt = load_dt_models()

    # Train optimized models on train splits

    # Heart data
    h_x_train, h_x_test, h_y_train, h_y_test = get_heart_train_test(
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE)

    heart_train_scores = cross_validate(
        heart_dt,
        h_x_train,
        h_y_train,
        scoring=make_scorer(roc_auc_score),
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True)

    best_heart_dt = heart_train_scores['estimator'][heart_train_scores['test_score'].argmax()]

    # Credit card data
    cc_x_train, cc_x_test, cc_y_train, cc_y_test = get_credit_card_train_test(
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE)

    credit_card_train_scores = cross_validate(
        heart_dt,
        cc_x_train,
        cc_y_train,
        scoring=make_scorer(roc_auc_score),
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True)

    best_credit_card_dt = credit_card_train_scores['estimator'][credit_card_train_scores['test_score'].argmax()]

    # Score optimized models against test splits
    plot_clf_analysis(best_heart_dt,
                      h_x_test,
                      h_y_test,
                      name="Heart Disease",
                      labels=["No Disease", "Disease"])
    plot_clf_analysis(best_credit_card_dt,
                      cc_x_test,
                      cc_y_test,
                      name="Credit Card Fraud",
                      labels=["No Fraud", "Fraud"])
