from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from analysis import analyze_clf
from datareader import load_preprocessor, load_study
from trainer import generate_credit_card_study, generate_heart_study, RANDOM_STATE


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


def load_dt_model(params):
    return DecisionTreeClassifier(
        criterion=params["criterion"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        min_impurity_decrease=params["min_impurity_decrease"],
        ccp_alpha=params["ccp_alpha"],
        class_weight="balanced",
        random_state=RANDOM_STATE)


def load_dt_heart_model():
    study = load_study("dt", "heart")
    return load_dt_model(study.best_params)


def load_dt_credit_card_model():
    study = load_study("dt", "credit_card")
    return load_dt_model(study.best_params)


def load_dt_heart_preprocessor():
    return load_preprocessor("dt", "heart")


def load_dt_credit_card_preprocessor():
    return load_preprocessor("dt", "credit_card")


if __name__ == "__main__":
    # Study Heart Failure dataset with DecisionTree
    nca_h = make_pipeline(StandardScaler(),
                          NeighborhoodComponentsAnalysis(n_components=12,
                                                         random_state=RANDOM_STATE))
    generate_heart_study(objective, "dt", 300, data_preprocessor=nca_h)

    clf = load_dt_heart_model()
    nca_h = load_dt_heart_preprocessor()

    analyze_clf(dataset_name="heart",
                name="Heart Failure",
                labels=["Healthy", "Failure"],
                clf=clf,
                data_preprocessor=nca_h)

    # Study Credit Card dataset with DecisionTree
    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=8,
                                                       random_state=RANDOM_STATE))
    generate_credit_card_study(objective, "dt", 75, data_preprocessor=nca)

    clf = load_dt_credit_card_model()
    nca = load_dt_credit_card_preprocessor()

    analyze_clf(dataset_name="credit_card",
                name="Credit Card Fraud",
                labels=["No Fraud", "Fraud"],
                clf=clf,
                data_preprocessor=nca)
