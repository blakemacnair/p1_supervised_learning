import pickle

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier

from analysis import cross_validate_and_analyze, plot_clf_analysis
from datareader import get_heart_train_test
from trainer import generate_heart_study, generate_studies, RANDOM_STATE, TRAIN_SIZE


def objective(trial, x, y, scoring=None):
    n_neighbors = trial.suggest_int("n_neighbors", 1, 100)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    algorithm = trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"])
    leaf_size = trial.suggest_int("leaf_size", 10, 50)
    p = trial.suggest_int("p", 1, 4)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               p=p,
                               n_jobs=-1)
    score = cross_val_score(knn, x, y, n_jobs=-1, cv=3, scoring=scoring)
    return score.mean()


def generate_knn_studies():
    generate_studies(objective, "knn")


def load_knn_models():
    return load_knn_heart_model(), load_knn_credit_card_model()


def load_knn_heart_model():
    with open("models/knn/heart_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return KNeighborsClassifier(n_neighbors=best_params["n_neighbors"],
                                weights=best_params["weights"],
                                algorithm=best_params["algorithm"],
                                leaf_size=best_params["leaf_size"],
                                p=best_params["p"],
                                n_jobs=-1)


def load_knn_credit_card_model():
    with open("models/knn/credit_card_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return KNeighborsClassifier(n_neighbors=best_params["n_neighbors"],
                                weights=best_params["weights"],
                                algorithm=best_params["algorithm"],
                                leaf_size=best_params["leaf_size"],
                                p=best_params["p"],
                                n_jobs=-1)


if __name__ == "__main__":
    # generate_knn_studies()
    generate_heart_study(objective, "knn", 200)

    heart_knn = load_knn_heart_model()

    h_x_train, h_x_test, h_y_train, h_y_test = get_heart_train_test(
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE)

    cross_validate_and_analyze(
        heart_knn,
        h_x_train,
        h_x_test,
        h_y_train,
        h_y_test,
        name="Heart Disease",
        labels=["No Disease", "Disease"],
        scoring=make_scorer(roc_auc_score)
    )
