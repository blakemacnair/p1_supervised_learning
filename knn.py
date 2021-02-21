import pickle

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import analyze_clf, cross_validate_and_analyze
from datareader import get_dataset_train_test, load_preprocessor, load_study
from trainer import generate_credit_card_study, generate_heart_study, generate_studies, RANDOM_STATE, TRAIN_SIZE


def objective(trial, x, y, scoring=None):
    cv_folds = 3
    max_n_neighbors = min(x.shape[0] * (cv_folds - 1) // cv_folds, 150)

    n_neighbors = trial.suggest_int("n_neighbors", 3, max_n_neighbors)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    algorithm = trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree"])
    leaf_size = trial.suggest_int("leaf_size", 1, 100)
    p = trial.suggest_int("p", 1, 5)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               p=p,
                               # metric="infinity",
                               n_jobs=-1)
    # ['chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p']
    score = cross_val_score(knn, x, y, n_jobs=-1, cv=3, scoring=scoring)
    return score.mean()


def generate_knn_studies():
    generate_studies(objective, "knn")


def load_knn_model(params):
    return KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                weights=params["weights"],
                                # algorithm=params["algorithm"],
                                leaf_size=params["leaf_size"],
                                p=params["p"],
                                metric="infinity",
                                n_jobs=-1)


def load_knn_heart_model():
    study = load_study("knn", "heart")
    return load_knn_model(study.best_params)


def load_knn_credit_card_model():
    study = load_study("knn", "credit_card")
    return load_knn_model(study.best_params)


def load_knn_heart_preprocessor():
    return load_preprocessor("knn", "heart")


def load_knn_credit_card_preprocessor():
    return load_preprocessor("knn", "credit_card")


if __name__ == "__main__":
    # generate_heart_study(objective, "knn", 200)
    #
    # analyze_clf(dataset_name="heart",
    #             name="Heart Failure",
    #             labels=["Healthy", "Failure"],
    #             clf=load_knn_heart_model(),
    #             data_preprocessor=None)

    prep = make_pipeline(StandardScaler(),
                         PCA(n_components="mle", random_state=RANDOM_STATE))
    generate_credit_card_study(objective,
                               "knn",
                               100,
                               percent_sample=0.3,
                               data_preprocessor=prep)

    analyze_clf(dataset_name="credit_card",
                name="Credit Card Fraud",
                labels=["No Fraud", "Fraud"],
                clf=load_knn_credit_card_model(),
                data_preprocessor=load_knn_credit_card_preprocessor())
