import pandas as pd


def load_raw_data(path):
    """
    Load raw data from a csv file
    :param path: path to the csv file
    :return: a pandas dataframe
    """
    return pd.read_csv(path, sep=';')

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
    caracteristiques.rename(columns={'Accident_Id': 'Num_Acc'}, inplace=True)
    
    # Merge the 4 datasets
    vehicules.drop(['Num_Acc'], axis=1, inplace=True)
    df = usagers.merge(vehicules, on='id_vehicule')
    df = df.merge(lieux, on='Num_Acc')
    df = df.merge(caracteristiques, on='Num_Acc')
    return df

def process_data(df):
    """
    Process the data :
    - replace birth year by age
    - replace 'secu1', 'secu2', 'secu3' by 'secu' containing the number of equipments used

    :param df: a pandas dataframe
    :return: a pandas dataframe
    """
    df['age'] = 2022 - df['an_nais']
    # Replace 'nan' values by -1
    df['age'].fillna(-1, inplace=True)
    
    # Create column 'secu' containing the number of equipments used
    df['secu'] = df['secu1'].apply(lambda x: 0 if x == 0 or x == -1 else 1)
    df['secu'] += df['secu2'].apply(lambda x: 0 if x == 0 or x == -1 else 1)
    df['secu'] += df['secu3'].apply(lambda x: 0 if x == 0 or x == -1 else 1)

    df.drop(['an_nais'], axis=1, inplace=True)
    return df

def select_features(df, features):
    """
    Return a dataframe with only the selected features
    :param df: a pandas dataframe
    :param features: a list of features
    :return: a pandas dataframe
    """
    return df[features]

def remove_missing_values(df):
    """
    Remove rows with missing values
    :param df: a pandas dataframe
    :return: a pandas dataframe
    """
    values_to_remove = ['nan', '', -1, '.']
    for col in df.columns:
        df = df[~df[col].isin(values_to_remove)]
    return df


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
