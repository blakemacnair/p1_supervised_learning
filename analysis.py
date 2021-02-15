import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, roc_auc_score


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
