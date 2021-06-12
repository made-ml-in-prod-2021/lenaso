import os, csv
import click
from pydantic import BaseModel, conint, confloat
import freddy


class DataModel(BaseModel):
    age: conint(ge=0, le=120)
    sex: conint(ge=0, le=1)
    cp: conint(ge=0, le=3)
    trestbps: conint(ge=0, le=300)
    chol: conint(ge=100, le=700)
    fbs: conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalach: conint(ge=0, le=300)
    exang: conint(ge=0, le=1)
    oldpeak: confloat(ge=0, le=10.0)
    scope: conint(ge=0, le=2)
    ca: conint(ge=0, le=4)
    thal: conint(ge=0, le=3)

    class Config:
        extra = 'forbid' #forbid option raise an exception when extra fields are determined


@click.command()
@click.option('--data_path', type=click.Path(), default='data/data.csv',  help='Data csv filepath')
@click.option('--target_path', type=click.Path(), default='data/target.csv',  help='Data csv filepath')
@click.option('--n_samples', type=int, default=1000)
def sample(data_path: str, target_path:str, n_samples=1000):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    target = []
    with open(data_path, 'w', newline='') as fout:
        fieldnames = list(DataModel.__fields__.keys())
        writer = csv.DictWriter(fout, fieldnames=fieldnames)

        writer.writeheader()
        for _ in range(n_samples):
            data = freddy.sample(DataModel)
            writer.writerow(data)
            target.append(int(data['age'] > 40 and data['chol'] > 200))

    with open(target_path, 'w') as fout:
        fout.write('target\n')
        fout.write('\n'.join(map(str, target)))


if __name__ == '__main__':
    sample()