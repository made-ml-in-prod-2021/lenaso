from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Elena Soldatenko soldatenkoes@gmail.com",
    install_requires=[
        "hydra-core==1.0.6",
        "importlib-resources==5.1.2",
        "python-dotenv>=0.5.1",
        "scikit-learn==0.24.1",
        "pyyaml==3.11",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.1.5",
    ],
    license="MIT",
)
