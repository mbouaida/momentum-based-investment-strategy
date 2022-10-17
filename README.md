Momentum Based Investment Strategy
==============================

The objective of this project is to build a momentum based investment strategy applied to Defi assets.


Approach
------------
The technical and non-technical parts of the project are explained in the jupyter notebook which you can find in the Notebooks folder.

Project Organization
------------

    ├── LICENSE            <- MIT License
    ├── README.md          <- README of the project
    ├── notebooks          <- Jupyter notebooks 
    ├── requirements.txt   <- The requirements file contains all the necessary libs to run the project
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src/defi           <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── optimization.py  <- Scripts to optimize portfolio with constraints
        │
        ├── backtests.py   <- Scripts to backtest our two strategies
        │ 
        └── evaluation.py  <- Scripts to provide evaluation functions
