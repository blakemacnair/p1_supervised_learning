import pickle

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from analysis import cross_validate_and_analyze
from datareader import get_dataset_train_test, load_preprocessor, load_study
from trainer import generate_credit_card_study, generate_studies, RANDOM_STATE, TRAIN_SIZE


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


def load_dt_models():
    return load_dt_heart_model(), load_dt_credit_card_model()


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


if __name__ == "__main__":
    # generate_dt_studies()

    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=7,
                                                       random_state=RANDOM_STATE))

    generate_credit_card_study(objective, "dt", 100, data_preprocessor=nca)

    heart_dt, credit_card_dt = load_dt_models()
    nca = load_preprocessor("dt", "credit_card")

    # h_x_train, h_x_test, h_y_train, h_y_test = get_dataset_train_test("heart",
    #                                                                   train_size=TRAIN_SIZE,
    #                                                                   random_state=RANDOM_STATE,
    #                                                                   data_preprocessor=nca)
    # cross_validate_and_analyze(
    #     heart_dt,
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
                                                                          random_state=RANDOM_STATE,
                                                                          data_preprocessor=nca)
    cross_validate_and_analyze(
        credit_card_dt,
        cc_x_train,
        cc_x_test,
        cc_y_train,
        cc_y_test,
        name="Credit Card Fraud",
        labels=["No Fraud", "Fraud"],
        scoring=make_scorer(roc_auc_score)
    )
