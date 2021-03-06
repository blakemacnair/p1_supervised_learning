from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
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

    hidden_layer_1 = trial.suggest_int("hidden_layer_1", 25, 500)
    hidden_layer_2 = trial.suggest_int("hidden_layer_2", 25, 250)
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["sgd", "adam"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2)
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-6, 1e-2)

    nn = MLPClassifier(hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
                       activation=activation,
                       solver=solver,
                       alpha=alpha,
                       learning_rate=learning_rate,
                       learning_rate_init=learning_rate_init,
                       random_state=RANDOM_STATE,
                       max_iter=1000)
    score = cross_val_score(nn, x, y, n_jobs=-1, cv=ss, scoring=scoring)
    return score.mean()


def load_nn_model(params):
    return MLPClassifier(hidden_layer_sizes=(params["hidden_layer_1"], params["hidden_layer_2"]),
                         activation=params["activation"],
                         solver=params["solver"],
                         alpha=params["alpha"],
                         learning_rate=params["learning_rate"],
                         learning_rate_init=params["learning_rate_init"],
                         random_state=RANDOM_STATE,
                         max_iter=1000)


def load_nn_heart_model():
    study = load_study("nn", "heart")
    return load_nn_model(study.best_params)


def load_nn_credit_card_model():
    study = load_study("nn", "credit_card")
    return load_nn_model(study.best_params)


def load_nn_heart_preprocessor():
    return load_preprocessor("nn", "heart")


def load_nn_credit_card_preprocessor():
    return load_preprocessor("nn", "credit_card")


def generate_nn():
    prep_h = make_pipeline(StandardScaler(),
                           PCA(n_components="mle", random_state=RANDOM_STATE))
    generate_heart_study(objective, "nn", 100, data_preprocessor=prep_h)

    prep_cc = make_pipeline(StandardScaler(),
                            PCA(n_components="mle", random_state=RANDOM_STATE))
    generate_credit_card_study(objective,
                               "nn",
                               75,
                               train_size=0.1,
                               data_preprocessor=prep_cc)


def validate_nn():
    print("Validating Neural Network for Heart Failure")
    analyze_clf(dataset_name="heart",
                clf_name="nn",
                labels=["Healthy", "Failure"],
                clf=load_nn_heart_model(),
                data_preprocessor=load_nn_heart_preprocessor())

    print("Validating Neural Network for Credit Card Fraud")
    analyze_clf(dataset_name="credit_card",
                clf_name="nn",
                labels=["No Fraud", "Fraud"],
                clf=load_nn_credit_card_model(),
                data_preprocessor=load_nn_credit_card_preprocessor(),
                train_max=0.5)


if __name__ == "__main__":
    generate_nn()
    validate_nn()
