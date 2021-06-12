import os
import pickle

import click
import pandas as pd


@click.command()
@click.option('--data_path', type=click.Path(exists=True), help='Input *.csv filepath')
@click.option('--model_path', type=click.Path(exists=True), help='Input *.pkl filepath')
@click.option('--pred_path', help='Output *.csv filepath')
def predict(data_path: str, model_path: str, pred_path: str):

    with open(model_path, 'rb') as fio:
        model = pickle.load(fio)

    data = pd.read_csv(data_path)
    pred = model.predict(data)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pd.Series(pred).to_csv(pred_path, index=False)


if __name__ == '__main__':
    predict()
