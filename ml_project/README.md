ml_project
==============================

Example of ml classification project (https://www.kaggle.com/ronitf/heart-disease-uci)

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
cd ml_project
pip install -e .
~~~
Usage:
~~~
python -m src.run_pipeline do_train=true do_predict=false
Run settings are in config/config.yaml or as command arguments.
~~~

Test:
~~~
pytest tests/
~~~

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │
    ├── outputs            <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── entities       <- config dataclasses
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    │   │ 
    ├── tests               <- Pytest tests for scr.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
