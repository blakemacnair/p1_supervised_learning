import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, plot_confusion_matrix, plot_roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate


def plot_clf_analysis(clf, x_test, y_test, name, labels):
    # Score the AUC-ROC
    y_test_pred = clf.predict(x_test)
    clf_roc_auc_score = roc_auc_score(y_test, y_test_pred, average='weighted')

    # Plot area under the ROC curve
    roc_curve_display = plot_roc_curve(clf,
                                       x_test,
                                       y_test,
                                       name=name)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("{} - Area Under the ROC Curve".format(name))
    plt.show()

    confusion_display = plot_confusion_matrix(clf,
                                              x_test,
                                              y_test,
                                              display_labels=labels,
                                              normalize="true",
                                              cmap=plt.get_cmap("Blues"))
    plt.title("{} - Confusion Matrix".format(name))
    plt.show()


def cross_validate_and_analyze(clf,
                               x_train,
                               x_test,
                               y_train,
                               y_test,
                               name,
                               labels,
                               scoring=make_scorer(roc_auc_score)):
    train_scores = cross_validate(
        clf,
        x_train,
        y_train,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True)

    best_clf = train_scores['estimator'][train_scores['test_score'].argmax()]

    plot_clf_analysis(best_clf,
                      x_test,
                      y_test,
                      name=name,
                      labels=labels)