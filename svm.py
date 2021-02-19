import pickle

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.svm import LinearSVC, SVC

from analysis import cross_validate_and_analyze, plot_clf_analysis
from datareader import get_credit_card_train_test, get_heart_train_test
from trainer import generate_credit_card_study, generate_heart_study, generate_studies, RANDOM_STATE, TRAIN_SIZE


def objective(trial, x, y, scoring=None):
    fold = StratifiedKFold(7)
    train_index, test_index = list(fold.split(x, y))[0]
    x = x.iloc[test_index]
    y = y.iloc[test_index]

    kernel = trial.suggest_categorical("kernel", ["poly", "rbf"])
    c = trial.suggest_float("c", 0.1, 2)
    degree = trial.suggest_int("degree", 1, 3)
    gamma = "scale"
    coef0 = trial.suggest_float("coef0", 0, 2)  # Only used in poly and sigmoid
    shrinking = trial.suggest_categorical("shrinking", [False, True])

    svc = SVC(C=c,
              kernel=kernel,
              degree=degree,
              gamma=gamma,
              coef0=coef0,
              shrinking=shrinking,
              class_weight="balanced",
              random_state=RANDOM_STATE)

    score = cross_val_score(svc, x, y, n_jobs=-1, cv=5, scoring=scoring)
    return score.mean()


def generate_svc_studies():
    generate_studies(objective, "svm")


def load_svc_models():
    return load_svc_heart_model(), load_svc_credit_card_model()


def load_svc_heart_model():
    with open("models/svm/heart_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return SVC(C=best_params["c"],
               kernel=best_params["kernel"],
               degree=best_params["degree"],
               gamma="scale",
               coef0=best_params["coef0"],
               shrinking=best_params["shrinking"],
               class_weight="balanced",
               random_state=RANDOM_STATE)


def load_svc_credit_card_model():
    with open("models/svm/credit_card_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return SVC(C=best_params["c"],
               kernel=best_params["kernel"],
               degree=best_params["degree"],
               gamma="scale",
               coef0=best_params["coef0"],
               shrinking=best_params["shrinking"],
               class_weight="balanced",
               random_state=RANDOM_STATE)


if __name__ == "__main__":
    # generate_svc_studies()

    heart_svc, cc_svc = load_svc_models()

    h_x_train, h_x_test, h_y_train, h_y_test = get_heart_train_test(
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE)
    cross_validate_and_analyze(
        heart_svc,
        h_x_train,
        h_x_test,
        h_y_train,
        h_y_test,
        name="Heart Disease",
        labels=["No Disease", "Disease"],
        scoring=make_scorer(roc_auc_score)
    )

    cc_x_train, cc_x_test, cc_y_train, cc_y_test = get_credit_card_train_test(
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE)
    cross_validate_and_analyze(
        cc_svc,
        cc_x_train,
        cc_x_test,
        cc_y_train,
        cc_y_test,
        name="Credit Card Fraud",
        labels=["No Fraud", "Fraud"],
        scoring=make_scorer(roc_auc_score)
    )
