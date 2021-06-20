import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COL = 'target'
DATA_FILE = 'data.csv'
TARGET_FILE = 'target.csv'
TRAIN_FILE = 'train.csv'
VALID_FILE = 'valid.csv'


@click.group()
def cli():
  pass


@cli.command(name='split')
@click.option('--input_dir')
@click.option('--output_dir')
@click.option('--val_size', type=float, default=0.2)
def split_train_val(input_dir: str, output_dir: str, val_size: float):
    data = pd.read_csv(os.path.join(input_dir, DATA_FILE),
                       usecols=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                'restecg', 'thalach', 'exang', 'oldpeak',
                                'scope', 'ca', 'thal'],
                       dtype={'age':int, 'sex':int, 'cp':int, 'trestbps':int, 'chol':int,
                              'fbs':int, 'restecg':int, 'thalach':int, 'exang':int,
                              'oldpeak':float, 'scope':int, 'ca':int, 'thal':int}
                       )
    target = pd.read_csv(os.path.join(input_dir, TARGET_FILE),
                         usecols=[TARGET_COL], dtype={TARGET_COL:int}
                         )

    data[TARGET_COL] = target[TARGET_COL]
    train_data, val_data = train_test_split(
        data, test_size=val_size, random_state=RANDOM_STATE, stratify=data[TARGET_COL],
    )

    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, TRAIN_FILE), index=False)
    val_data.to_csv(os.path.join(output_dir, VALID_FILE), index=False)


@cli.command()
@click.option('--input_dir')
@click.option('--output_dir')
def preprocess(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(input_dir, DATA_FILE))
    data.to_csv(os.path.join(output_dir, DATA_FILE), index=False)

    target = pd.read_csv(os.path.join(input_dir, TARGET_FILE))
    target.to_csv(os.path.join(output_dir, TARGET_FILE), index=False)


if __name__ == '__main__':
    cli()
