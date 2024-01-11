# road_accident_severity
The goal of this project is to predict the severity of corporeal injury of a road accident based on the data provided by the French Ministry of the Interior.

## Data
The data contains 4 files that can be found in the `data` folder. It is provided by the French Ministry of the Interior and can be found [here](https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation/).
- `caracteristiques-2022.csv` : contains the characteristics of the accident
- `lieux-2022.csv` : contains the location of the accident
- `usagers-2022.csv` : contains the information about the users involved in the accident
- `vehicules-2022.csv` : contains the information about the vehicles involved in the accident

## Notebooks
The notebooks are located in the `notebooks` folder. There are 5 of them:
- `feature_selection.ipynb` : contains the data exploration and the feature selection process
- `hp_search.ipynb` : contains the hyperparameter search for several models
- `severity_classification.ipynb` : contains the training of 5 models using the best hyperparameters found in the previous notebook. It compares the performance on several sets of features :
    - a set of **26 features** extracted and built from the data
    - a set of **5 principal components** computed with PCA
    - a set of the **5 most important features** found in the 5 principal components
- `severity_classification_binary.ipynb` : compares the performances of the models with the original target variable (4 classes) and a binary target variable (2 classes)
- `severity_classification_bin_smote.ipynb` : compares the performances of the models with a binary target variable and a binary target variable where the minority class has been oversampled using SMOTE
