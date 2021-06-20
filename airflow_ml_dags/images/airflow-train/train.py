import os
import pickle
import json

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RANDOM_STATE = 42
TARGET_COL = 'target'


@click.group()
def cli():
  pass


@cli.command()
@click.option('--data_path', type=click.Path(exists=True), help='Input *.csv filepath')
@click.option('--model_path', help='Output *.pkl filepath')
def train(data_path: str, model_path: str):
    data = pd.read_csv(data_path)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(data.drop(columns=TARGET_COL), data[TARGET_COL])

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as fio:
        pickle.dump(model, fio)


@cli.command()
@click.option('--data_path', type=click.Path(exists=True), help='Input *.csv filepath')
@click.option('--model_path', type=click.Path(exists=True), help='Input *.pkl filepath')
@click.option('--metric_path', help='Output *.json filepath')
def validate(data_path: str, model_path: str, metric_path: str):

    data = pd.read_csv(data_path)

    with open(model_path, 'rb') as fio:
        model = pickle.load(fio)

    predictions = model.predict(data.drop(columns=TARGET_COL))

    metrics = {
        'accuracy': accuracy_score(data[TARGET_COL], predictions),
        'f1_score': f1_score(data[TARGET_COL], predictions),
        'precision': precision_score(data[TARGET_COL], predictions),
        'recall': recall_score(data[TARGET_COL], predictions),
    }

    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    with open(metric_path, 'w') as fio:
        json.dump(metrics, fio)


if __name__ == '__main__':
    cli()
