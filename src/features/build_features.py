import pathlib
import pandas as pd
import numpy as np

def load_data(data_path):
    df = pd.read_parquet(data_path)
    return df

def save_data(train, test, validate,output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    validate.to_csv(output_path + '/validate.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

def preprocessing(train, validate, test):
    # Extract Dyula and French columns from the nested dictionary
    train[['dyu', 'fr']] = train['translation'].apply(pd.Series)
    validate[['dyu', 'fr']] = validate['translation'].apply(pd.Series)
    test[['dyu', 'fr']] = test['translation'].apply(pd.Series)

    # Drop the original 'translation' column
    train.drop(columns=['translation'], inplace=True)
    validate.drop(columns=['translation'], inplace=True)
    test.drop(columns=['translation'], inplace=True)

    # Convert text to lowercase
    train['dyu'] = train['dyu'].apply(lambda x: x.lower())
    train['fr'] = train['fr'].apply(lambda x: x.lower())

    validate['dyu'] = validate['dyu'].apply(lambda x: x.lower())
    validate['fr'] = validate['fr'].apply(lambda x: x.lower())

    test['dyu'] = test['dyu'].apply(lambda x: x.lower())

    return train, validate, test


if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    train_path = home_dir.as_posix() + '/data/raw/train-00000-of-00001.parquet'
    validate_path = home_dir.as_posix() + '/data/raw/validation-00000-of-00001.parquet'
    test_path = home_dir.as_posix() + '/data/raw/test-00000-of-00001.parquet'

    train = load_data(train_path)
    validate = load_data(validate_path)
    test = load_data(test_path)

    train, validate, test = preprocessing(train=train, validate=validate, test=test)

    save_data(train, validate, test, home_dir.as_posix() + '/data/processed')
