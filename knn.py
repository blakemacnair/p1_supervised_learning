from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import analyze_clf
from datareader import load_preprocessor, load_study
from trainer import generate_credit_card_study, generate_heart_study, RANDOM_STATE, TRAIN_SIZE


def objective(trial, x, y, train_size=TRAIN_SIZE, scoring=None):
    cv_folds = 5
    ss = StratifiedShuffleSplit(train_size=train_size,
                                random_state=RANDOM_STATE,
                                n_splits=cv_folds)

    max_n_neighbors = min(int(x.shape[0] * train_size),
                          150)

    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, max_n_neighbors),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree"]),
        "leaf_size": trial.suggest_int("leaf_size", 1, 100),
        "p": trial.suggest_int("p", 1, 5)
    }
    knn = load_knn_model(params)
    score = cross_val_score(knn, x, y, n_jobs=-1, cv=ss, scoring=scoring)
    return score.mean()


def load_knn_model(params):
    return KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                weights=params["weights"],
                                algorithm=params["algorithm"],
                                leaf_size=params["leaf_size"],
                                p=params["p"],
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


def generate_knn():
    prep_h = make_pipeline(StandardScaler(),
                           PCA(n_components="mle", random_state=RANDOM_STATE))
    generate_heart_study(objective,
                         "knn",
                         200,
                         data_preprocessor=prep_h)

    prep_cc = make_pipeline(StandardScaler(),
                            PCA(n_components=8, random_state=RANDOM_STATE))
    generate_credit_card_study(objective,
                               "knn",
                               75,
                               train_size=0.1,
                               data_preprocessor=prep_cc)


def validate_knn():
    print("Validating KNN for Heart Failure")
    analyze_clf(dataset_name="heart",
                clf_name="knn",
                labels=["Healthy", "Failure"],
                clf=load_knn_heart_model(),
                data_preprocessor=load_knn_heart_preprocessor())

    print("Validating KNN for Credit Card Fraud")
    analyze_clf(dataset_name="credit_card",
                clf_name="knn",
                labels=["No Fraud", "Fraud"],
                clf=load_knn_credit_card_model(),
                data_preprocessor=load_knn_credit_card_preprocessor(),
                train_max=0.1)


if __name__ == "__main__":
    generate_knn()
    validate_knn()
