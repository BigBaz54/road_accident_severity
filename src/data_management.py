import pandas as pd
from sklearn.decomposition import PCA


def load_raw_data(path):
    """
    Load raw data from a csv file
    :param path: path to the csv file
    :return: a pandas dataframe
    """
    return pd.read_csv(path, sep=';', decimal=',', low_memory=False)

def joined_data(caracteristiques, lieux, usagers, vehicules):
    """
    Join the 4 datasets into one dataframe
    :param caracteristiques: a pandas dataframe
    :param lieux: a pandas dataframe
    :param usagers: a pandas dataframe
    :param vehicules: a pandas dataframe
    :return: a pandas dataframe
    """
    # Rename column 'Accident_Id' to 'Num_Acc' in caracteristiques
    car = caracteristiques.rename(columns={'Accident_Id': 'Num_Acc'}, inplace=False)
    
    # Merge the 4 datasets
    veh = vehicules.drop(['Num_Acc'], axis=1, inplace=False)
    df = usagers.merge(veh, on='id_vehicule')
    df = df.merge(lieux, on='Num_Acc')
    df = df.merge(car, on='Num_Acc')
    return df

def process_data(df):
    """
    Process the data :
    - replace birth year by age
    - add column 'secu' containing the number of equipments used + 1
    - add column 'obscar' merging 'obs' and 'obsm' ('obms' if existing, else 'obs' + 10)
    - replace letter values by numbers in 'actp'
    - replace the one '#ERREUR' value in 'nbv' by -1
    - add 1 to 'infra' to have 'no infrastructure' counting as an actual information
    - add 1 to 'choc' to have 'no contact' counting as an actual information
    - add 1 to 'nbv' to have '0 route' counting as an actual information
    - add 1 to 'vosp' to have 'no reserved route' counting as an actual information

    :param df: a pandas dataframe
    :return: a pandas dataframe
    """
    df = df.copy()

    df['age'] = 2022 - df['an_nais']
    # Replace 'nan' values by -1
    df['age'].fillna(-1, inplace=True)

    # Replace letter values by numbers in 'actp'
    df['actp'] = df['actp'].apply(lambda x: 10 if x == 'A' else -1 if (x == 'B' or x == '-1' or x == ' -1') else int(x))

    # Replace the one '#ERREUR' value in 'plan' by -1
    df['nbv'] = df['nbv'].apply(lambda x: -1 if (x == '#ERREUR' or x == ' -1') else x)

    # Add 1 to 'infra' to have 'no infrastructure' counting as an actual information
    df['infra'] += 1

    # Add 1 to 'choc'
    df['choc'] += 1

    # Add 1 to 'nbv'
    df['nbv'].apply(lambda x: -1 if x == ' -1' else int(x) + 1 if int(x) > 0 else int(x))

    # Add 1 to 'vosp'
    df['vosp'] += 1

    
    # Create column 'secu' containing the number of equipments used + 1
    df['secu'] = 1
    df['secu'] += df['secu1'].apply(lambda x: 0 if x == 0 or x == -1 else 1)
    df['secu'] += df['secu2'].apply(lambda x: 0 if x == 0 or x == -1 else 1)
    df['secu'] += df['secu3'].apply(lambda x: 0 if x == 0 or x == -1 else 1)

    # Create column 'obscar' merging 'obs' and 'obsm' ('obms' if existing, else 'obs' + 9)
    df['obscar'] = df['obsm'].apply(lambda x: x + 1 if x > 0 else float('nan'))
    df['obscar'].fillna(df['obs'].apply(lambda x: 1 if x == 0 else x + 10 if x > 0 else -1), inplace=True)


    df.drop(['an_nais'], axis=1, inplace=True)

    return df

def select_features(df, features):
    """
    Return a dataframe with only the selected features
    :param df: a pandas dataframe
    :param features: a list of features
    :return: a pandas dataframe
    """
    return df[features].copy()

def remove_missing_values(df):
    """
    Remove rows with missing values
    :param df: a pandas dataframe
    :return: a pandas dataframe
    """
    values_to_remove = [float('nan'), '', -1, '.', 0]

    df = df.copy()

    for col in df.columns:
        df = df[~df[col].isin(values_to_remove)]
    
    return df

def data_available(df, feature):
    """
    Return the percentage of data available for a feature
    :param df: a pandas dataframe
    :param feature: a feature
    :return: a float
    """
    return 100 * (1 - df[feature].isin([float('nan'), '', -1, '.', 0]).sum() / len(df))

def workable_data(nb_pca=5):
    """
    Return a dataset containing the 27 features selected in the corresponding notebook and the 5 PCA components
    :return: a pandas dataframe
    """
    caracteristiques = load_raw_data('data/caracteristiques-2022.csv')
    lieux = load_raw_data('data/lieux-2022.csv')
    usagers = load_raw_data('data/usagers-2022.csv')
    vehicules = load_raw_data('data/vehicules-2022.csv')

    df = joined_data(caracteristiques, lieux, usagers, vehicules)
    df = process_data(df)

    selected_features = ['place', 'catu', 'grav', 'sexe', 'trajet', 'senc', 'catv', 'obscar', 'choc', 'manv','motor', 'catr', 'circ', 'vosp','prof',  'plan', 'surf', 'infra','situ', 'vma',  'lum', 'agg','int', 'atm', 'col', 'age', 'secu']
    df = select_features(df, selected_features)

    df = remove_missing_values(df)

    data_without_grav = df.drop(['grav'], axis=1, inplace=False)
    grav = df['grav']
    data_PCA = df.drop(['grav'], axis=1, inplace=False)
    data_PCA = PCA(n_components=nb_pca).fit_transform(data_PCA)

    data_PCA = pd.DataFrame(data_PCA, columns= ['PC' + str(i) for i in range(nb_pca)])
    
    return data_without_grav, grav, data_PCA


if __name__ == "__main__":
    caracteristiques = load_raw_data('data/caracteristiques-2022.csv')
    lieux = load_raw_data('data/lieux-2022.csv')
    usagers = load_raw_data('data/usagers-2022.csv')
    vehicules = load_raw_data('data/vehicules-2022.csv')

    df = joined_data(caracteristiques, lieux, usagers, vehicules)
    df = process_data(df)

    print(len(df))
    print(len(df['id_usager'].unique()))

    print(df.head(30))

    df = select_features(df, ['place', 'grav'])

    print(df.head(30))

    df = remove_missing_values(df)

    print(len(df))

    print(workable_data()[2].columns)
