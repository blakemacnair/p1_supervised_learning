import pickle

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier

from analysis import cross_validate_and_analyze, plot_clf_analysis
from datareader import get_heart_train_test
from trainer import generate_credit_card_study, generate_studies, RANDOM_STATE, TRAIN_SIZE


def objective(trial, x, y, scoring=None):
    hidden_layer_1 = trial.suggest_int("hidden_layer_1", 25, 1000)
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["sgd", "adam"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2)
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-6, 1e-2)

    nn = MLPClassifier(hidden_layer_sizes=(hidden_layer_1,),
                       activation=activation,
                       solver=solver,
                       alpha=alpha,
                       learning_rate=learning_rate,
                       learning_rate_init=learning_rate_init,
                       random_state=RANDOM_STATE,
                       max_iter=1000)
    score = cross_val_score(nn, x, y, n_jobs=-1, cv=3, scoring=scoring)
    return score.mean()


def generate_nn_studies():
    generate_studies(objective, "nn")


def load_nn_models():
    return load_nn_heart_model(), load_nn_credit_card_model()


def load_nn_heart_model():
    with open("models/nn/heart_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return MLPClassifier(hidden_layer_sizes=(best_params["hidden_layer_1"],),
                         activation=best_params["activation"],
                         solver=best_params["solver"],
                         alpha=best_params["alpha"],
                         learning_rate=best_params["learning_rate"],
                         learning_rate_init=best_params["learning_rate_init"],
                         random_state=RANDOM_STATE,
                       max_iter=1000)


def load_nn_credit_card_model():
    with open("models/nn/credit_card_study", "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params
    return MLPClassifier(hidden_layer_sizes=(best_params["hidden_layer_1"],),
                         activation=best_params["activation"],
                         solver=best_params["solver"],
                         alpha=best_params["alpha"],
                         learning_rate=best_params["learning_rate"],
                         learning_rate_init=best_params["learning_rate_init"],
                         random_state=RANDOM_STATE,
                       max_iter=1000)


if __name__ == "__main__":
    # generate_nn_studies()
    generate_credit_card_study(objective, "nn", 50)

    heart_knn = load_nn_heart_model()

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
