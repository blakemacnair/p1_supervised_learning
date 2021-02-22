# Instructions

1. Prerequisites:
    1. Ensure you have a working installation of Anaconda `conda`.
    2. Have the [Kaggle API](https://www.kaggle.com/docs/api) set up, or download the datasets in `setup.sh` manually from kaggle.com.

2. Run `bash setup.sh` to download the datasets into the data directory and install the required conda environment from `env.yml`.

3. Run `conda activate ml-1-supervised-learning`.

4. To generate models for all requested model types, run `bash generate_models.sh`.

5. To run validation on all models and output graphs and metrics, run `bash validate_models.sh`.
