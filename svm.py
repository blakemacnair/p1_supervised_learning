from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

from analysis import analyze_clf
from datareader import load_preprocessor, load_study
from trainer import generate_credit_card_study, generate_heart_study, RANDOM_STATE


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


def load_svm_model(params):
    return SVC(C=params["c"],
               kernel=params["kernel"],
               degree=params["degree"],
               gamma="scale",
               coef0=params["coef0"],
               shrinking=params["shrinking"],
               class_weight="balanced",
               random_state=RANDOM_STATE)


def load_svm_heart_model():
    study = load_study("svm", "heart")
    return load_svm_model(study.best_params)


def load_svm_credit_card_model():
    study = load_study("svm", "credit_card")
    return load_svm_model(study.best_params)


def load_svm_heart_preprocessor():
    return load_preprocessor("svm", "heart")


def load_svm_credit_card_preprocessor():
    return load_preprocessor("svm", "credit_card")


if __name__ == "__main__":
    # Study Heart Failure dataset with SVM
    generate_heart_study(objective, "svm", 300, data_preprocessor=None)

    clf = load_svm_heart_model()
    analyze_clf(dataset_name="heart",
                name="Heart Failure",
                labels=["Healthy", "Failure"],
                clf=clf,
                data_preprocessor=None)

    # Study Credit Card dataset with SVM
    generate_credit_card_study(objective, "svm", 75, data_preprocessor=None)

    clf = load_svm_credit_card_model()
    analyze_clf(dataset_name="credit_card",
                name="Credit Card Fraud",
                labels=["No Fraud", "Fraud"],
                clf=clf,
                data_preprocessor=None)
