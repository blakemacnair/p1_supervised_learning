from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from analysis import analyze_clf
from datareader import load_preprocessor, load_study
from trainer import generate_credit_card_study, generate_heart_study, RANDOM_STATE


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
    max_depth = trial.suggest_int("max_depth", 3, 7)
    min_samples_split = trial.suggest_float("min_samples_split", 1e-3, 1e-1)
    min_samples_leaf = trial.suggest_float("min_samples_leaf", 1e-3, 1e-1)
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 1e-10, 1e-1)
    ccp_alpha = trial.suggest_float("ccp_alpha", 1e-3, 5e-1)

    n_estimators = trial.suggest_int("n_estimators", 15, 50)
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


def load_boosting_model(params):
    return boosted_decision_tree(criterion=params["criterion"],
                                 max_depth=params["max_depth"],
                                 min_samples_split=params["min_samples_split"],
                                 min_samples_leaf=params["min_samples_leaf"],
                                 min_impurity_decrease=params["min_impurity_decrease"],
                                 ccp_alpha=params["ccp_alpha"],
                                 n_estimators=params["n_estimators"],
                                 learning_rate=params["learning_rate"])


def load_boosting_heart_model():
    study = load_study("boosting", "heart")
    return load_boosting_model(study.best_params)


def load_boosting_credit_card_model():
    study = load_study("boosting", "credit_card")
    return load_boosting_model(study.best_params)


def load_dt_heart_preprocessor():
    return load_preprocessor("boosting", "heart")


def load_dt_credit_card_preprocessor():
    return load_preprocessor("boosting", "credit_card")


def generate_boosting():
    nca_h = make_pipeline(StandardScaler(),
                          NeighborhoodComponentsAnalysis(n_components=12,
                                                         random_state=RANDOM_STATE))
    generate_heart_study(objective,
                         "boosting",
                         200,
                         data_preprocessor=nca_h)

    nca_cc = make_pipeline(StandardScaler(),
                           NeighborhoodComponentsAnalysis(n_components=8,
                                                          random_state=RANDOM_STATE))
    generate_credit_card_study(objective,
                               "boosting",
                               75,
                               percent_sample=0.25,
                               data_preprocessor=nca_cc)


def validate_boosting():
    analyze_clf(dataset_name="heart",
                name="Heart Failure",
                labels=["Healthy", "Failure"],
                clf=load_boosting_heart_model(),
                data_preprocessor=load_dt_heart_preprocessor())

    analyze_clf(dataset_name="credit_card",
                name="Credit Card Fraud",
                labels=["No Fraud", "Fraud"],
                clf=load_boosting_credit_card_model(),
                data_preprocessor=load_dt_credit_card_preprocessor())


if __name__ == "__main__":
    generate_boosting()
    validate_boosting()
