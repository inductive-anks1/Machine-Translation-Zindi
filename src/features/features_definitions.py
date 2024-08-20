import pandas as pd
import numpy as np

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


def merge(train, validate):
    df = pd.concat([train, validate], ignore_index=True)
    return df