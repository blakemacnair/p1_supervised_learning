import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, make_scorer, plot_confusion_matrix, plot_roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold

from datareader import get_dataset_train_test, RANDOM_STATE
from trainer import TRAIN_SIZE
from datareader import save_figure


def plot_clf_confusion_mat(clf, x_test, y_test, clf_name, dataset_name, labels):
    fig, ax = plt.subplots()
    confusion_display = plot_confusion_matrix(clf,
                                              x_test,
                                              y_test,
                                              display_labels=labels,
                                              normalize="true",
                                              ax=ax,
                                              cmap=plt.get_cmap("Blues"))

    save_figure(fig, clf_name, dataset_name, "confusion_matrix")


def plot_cross_val_roc_curves(clf, x, y, clf_name, dataset_name):
    cv = StratifiedKFold(n_splits=6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(x, y)):
        clf.fit(x[train], y[train])
        viz = plot_roc_curve(clf, x[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.legend(loc="lower right")

    save_figure(fig, clf_name, dataset_name, "roc_auc_cv")


def cross_validate_and_analyze(clf,
                               x_train,
                               x_test,
                               y_train,
                               y_test,
                               clf_name,
                               dataset_name,
                               labels,
                               scoring=make_scorer(roc_auc_score)):
    plot_cross_val_roc_curves(clf, x_train, y_train, clf_name, dataset_name)

    train_scores = cross_validate(
        clf,
        x_train,
        y_train,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
        return_estimator=True)
    best_clf = train_scores['estimator'][train_scores['test_score'].argmax()]

    plot_clf_confusion_mat(best_clf,
                           x_test,
                           y_test,
                           clf_name=clf_name,
                           dataset_name=dataset_name,
                           labels=labels)


def analyze_clf(dataset_name,
                clf_name,
                labels,
                clf,
                train_size=TRAIN_SIZE,
                data_preprocessor=None):
    x_train, x_test, y_train, y_test = get_dataset_train_test(dataset_name,
                                                              train_size=train_size,
                                                              random_state=RANDOM_STATE,
                                                              data_preprocessor=data_preprocessor)
    cross_validate_and_analyze(
        clf,
        x_train,
        x_test,
        y_train,
        y_test,
        clf_name=clf_name,
        dataset_name=dataset_name,
        labels=labels,
        scoring=make_scorer(roc_auc_score)
    )
