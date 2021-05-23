from setuptools import find_packages, setup

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="Example of ml project",
    author="Elena Soldatenko",
    author_email="soldatenkoes@gmail.com",
    python_requires='>=3.8',
    install_requires=[
        "hydra-core==1.0.6",
        "importlib-resources>=5.1.2",
        "scikit-learn==0.24.1",
        "pyyaml==5.4.1",
        "marshmallow-dataclass==8.3.0",
        "pandas==1.2.4",
        "pytest==6.2.1"
    ],
    license="MIT",
)
