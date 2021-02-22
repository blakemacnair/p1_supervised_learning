# Download datasets
kaggle d download -p ./data --unzip -d cherngs/heart-disease-cleveland-uci
kaggle d download -p ./data --unzip -d mlg-ulb/creditcardfraud

# Install and switch to project conda environment
conda env create -f env.yml
conda activate ml-1-supervised-learning
