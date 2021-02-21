import optuna.visualization as vis
import matplotlib.pyplot as plt

from datareader import load_study

if __name__ == "__main__":
    study = load_study("dt", "heart")
    vis.plot_contour(study).show()
    vis.plot_param_importances(study).show()
